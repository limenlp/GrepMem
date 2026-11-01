"""Evaluate a harness on ContextBench (arXiv 2609.37725) against an OpenAI-compatible vLLM server.

  python cb_eval.py --agent summary_grep --base-url http://localhost:8000/v1 --out eval_cb_step0.jsonl

Runs the KV Store and Log Triage families (exact recall after streaming up to ~24x a 32K window) at the paper's
levels and seeds through agents/contextbench.py, graded by ContextBench's own graders (reward = fraction of the
24 queries answered exactly). Instances are generated locally; they are byte-identical to the released ones.
One line per instance goes to --out; re-running with the same --out resumes. When all instances are done, the
summary goes to <out>.summary.json (eval_cb_step<N>.jsonl -> eval_cb_step<N>.summary.json), which a training run
started by train.sh logs into its own wandb run as eval/cb/* (agents/eval_bridge.py). If instances are missing
(e.g. dropped LLM calls), no summary is written and the exit code is 2; re-run to retry them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import statistics
from pathlib import Path

from contextbench.cli import load_config
from contextbench.levels import iter_levels, level_knobs

from agents.contextbench import HARNESS_NOTES, CBAgent, generate_instance, run_instance
from agents.memory import COST_STATS, MASK_STATS, MEMGPT_STATS

DEFAULT_LEVELS = "kv_store:R520,R1000,R2000,R4000,R8000,R16000 log_triage:L640,L1600,L3200,L6400,L12800"
DIAG_KEYS = (
    (
        "reward_lookup",
        "reward_count",
        "kv_hash_dropped",
        "kv_wrong",
        "kv_unanswered",
        "kv_grepped",
        "kv_correct_grepped",
        "kv_correct_ungrepped",
        "grep_empty",
        "grep_chars",
        "llm_calls",
        "trimmed_turns",
    )
    + MASK_STATS
    + MEMGPT_STATS
    + COST_STATS
)


def build_instances(levels: str, seeds: list[int], cache: Path) -> list[dict]:
    """Generate (once) every family/level/seed instance under ``cache``."""
    want = {fam: lv.split(",") for fam, lv in (x.split(":") for x in levels.split())}
    out = []
    for fam, spec, entry in iter_levels(load_config()):
        if entry["name"] not in want.get(fam, []):
            continue
        for seed in seeds:
            d = cache / f"{fam}-{entry['name']}-seed{seed}"
            # Concurrent evaluations may generate the same instance; the first rename wins.
            generate_instance(fam, {**level_knobs(spec, entry), "seed": seed}, d)
            out.append(
                {
                    "id": f"{fam}/{entry['name']}/{seed}",
                    "family": fam,
                    "level": entry["name"],
                    "seed": seed,
                    "pressure": entry.get("figure_x"),
                    "dir": d,
                }
            )
    missing = {f"{f}:{lv}" for f, lvs in want.items() for lv in lvs} - {f"{i['family']}:{i['level']}" for i in out}
    if missing:
        raise SystemExit(f"unknown ContextBench levels: {sorted(missing)}")
    return out


def summarize(res: list[dict]) -> dict:
    s: dict = {"n": len(res)}
    fams = sorted({r["family"] for r in res})
    for fam in fams:
        rows = [r for r in res if r["family"] == fam]
        s[f"{fam}/reward"] = statistics.mean(r["reward"] for r in rows)
        over = [r["reward"] for r in rows if (r["pressure"] or 0) >= 1]
        if over:
            s[f"{fam}/reward_over_budget"] = statistics.mean(over)  # levels whose stream exceeds the window
        for lv in sorted(
            {r["level"] for r in rows}, key=lambda x: next(r["pressure"] for r in rows if r["level"] == x)
        ):
            lr = [r for r in rows if r["level"] == lv]
            s[f"{fam}/{lv}/reward"] = statistics.mean(r["reward"] for r in lr)
        s[f"{fam}/budget_wall"] = statistics.mean(r["terminated_by"] == "budget_wall" for r in rows)
        s[f"{fam}/stuck"] = statistics.mean(r["terminated_by"] == "stuck" for r in rows)
        s[f"{fam}/compactions"] = statistics.mean(r["compactions"] for r in rows)
        s[f"{fam}/grep_calls"] = statistics.mean(r["grep_calls"] for r in rows)
        for k in DIAG_KEYS:  # per-episode means (KV failure reasons, Log lookup vs count accuracy, grep result sizes)
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                s[f"{fam}/{k}"] = statistics.mean(vals)
    s["reward"] = statistics.mean(s[f"{fam}/reward"] for fam in fams)  # macro average over families
    for k in ("llm_calls",) + COST_STATS:  # per-instance means over all instances
        vals = [r[k] for r in res if r.get(k) is not None]
        if len(vals) == len(res):
            s[k] = statistics.mean(vals)
    return s


async def main(args):
    seeds = [int(x) for x in args.seeds.split(",")]
    insts = build_instances(args.levels, seeds, Path(args.instances_dir))
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r := json.loads(line))["id"]: r for line in f}
    todo = [i for i in insts if i["id"] not in done]
    print(f"{len(insts)} instances, {len(done)} already done, running {len(todo)} with agent={args.agent}", flush=True)
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sem, lock, out = asyncio.Semaphore(args.concurrency), asyncio.Lock(), open(args.out, "a")
    runs_dir = Path(args.runs_dir)

    async def one(inst, k):
        async with sem:
            run_dir = runs_dir / inst["id"].replace("/", "-")
            agent = CBAgent(args.agent, urls[k % len(urls)], args.model)
            try:
                result = await run_instance(agent, inst["dir"], run_dir)
            except Exception as e:  # noqa: BLE001  (dropped LLM call etc.: retried by the next run)
                print(f"{inst['id']} failed: {e!r}", flush=True)
                return
            rw = result["rewards"]
            rec = {k2: inst[k2] for k2 in ("id", "family", "level", "seed", "pressure")}
            rec.update(
                {
                    k2: result[k2]
                    for k2 in (
                        "reward",
                        "terminated_by",
                        "n_turns",
                        "peak_tokens",
                        "n_ops_delivered",
                        "n_ops_total",
                        "llm_calls",
                        "compactions",
                        "grep_calls",
                        "tool_errors",
                        "max_context",
                    )
                }
            )
            rec.update(
                {
                    k2: rw[k2]
                    for k2 in (
                        "n_answered",
                        "reward_attempted",
                        "reward_lookup",
                        "reward_count",
                        "token_method",
                        "graded_from",
                    )
                    if k2 in rw
                }
            )
            rec.update({k2: result[k2] for k2 in DIAG_KEYS if k2 in result})
            if not args.keep_snapshots:
                shutil.rmtree(run_dir / "context_snapshots", ignore_errors=True)
            async with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[inst["id"]] = rec
                print(
                    f"[{len(done)}/{len(insts)}] {inst['id']} reward={rec['reward']:.3f} {rec['terminated_by']} "
                    f"turns={rec['n_turns']} compactions={rec['compactions']} greps={rec['grep_calls']}",
                    flush=True,
                )

    # Longest streams first, so they do not start last and dominate the wall-clock time.
    todo.sort(key=lambda i: -(i["pressure"] or 0))
    await asyncio.gather(*(one(i, k) for k, i in enumerate(todo)))
    out.close()
    res = [done[i["id"]] for i in insts if i["id"] in done]
    summary = summarize(res) if res else {"n": 0}
    print(json.dumps(summary, indent=2))
    if len(res) < len(insts):
        print(f"incomplete: {len(insts) - len(res)} instances missing; re-run to retry them", flush=True)
        raise SystemExit(2)
    if args.out.endswith(".jsonl"):
        path = args.out[: -len(".jsonl")] + ".summary.json"
        with open(path + ".tmp", "w") as f:
            json.dump(
                {**summary, "agent": args.agent, "model": args.model, "levels": args.levels, "seeds": args.seeds},
                f,
                indent=2,
            )
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="summary_grep", choices=list(HARNESS_NOTES))
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="one or more comma-separated vLLM servers")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="served model name")
    ap.add_argument(
        "--levels",
        default=os.environ.get("CB_LEVELS", DEFAULT_LEVELS),
        help='"family:LEVEL,LEVEL family:..." (paper.yaml names)',
    )
    ap.add_argument("--seeds", default=os.environ.get("CB_SEEDS", "42,43,44,45"))
    ap.add_argument("--instances-dir", default=".cache/grepmem/cb_instances")
    ap.add_argument("--runs-dir", default="outputs/cb_runs/latest", help="per-instance snapshots, result.json")
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--keep-snapshots", action="store_true", help="keep per-turn context snapshots (large)")
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
