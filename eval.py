"""Evaluate a harness against an OpenAI-compatible vLLM server (no training stack needed).

  python eval.py --agent react --base-url http://localhost:8000/v1 --model Qwen/Qwen3.5-9B \
      --dataset browsecomp_plus --split test --out results.jsonl

Reports LLM-judge accuracy, token F1 and exact match;
re-running with the same --out resumes. When all questions are done, the summary is also written to
<out>.summary.json (eval_bcp_step<N>.jsonl -> eval_bcp_step<N>.summary.json); a training run started by
train.sh logs that file into its own wandb run (agents/eval_bridge.py). If questions are missing (e.g. dropped
LLM calls), no summary is written and the exit code is 2; re-run to retry them.

Multi-question datasets (browsecomp_plus_multi = BCP-Multi, eval_bcpm_step<N>.jsonl) are scored per question:
acc is the mean judge verdict over all questions, all_correct the fraction of episodes with every answer right.
Tool-use stats (search/document/grep calls, free grep turns, LLM calls) are averaged per episode.

With --wandb-run NAME --step N, the summary is also logged to a separate wandb run NAME at step N (older runs).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import uuid

from rllm.data.dataset import DatasetRegistry
from rllm.types import AgentConfig, Task

from agents.common import f1_evaluator, judge_evaluator
from agents.flows import FLOWS
from agents.judge import load_config

STAT_KEYS = (
    "all_correct",
    "llm_calls",
    "search_calls",
    "doc_calls",
    "grep_calls",
    "grep_turns",
    "grep_empty",
    "grep_chars",
    "max_context",
    "compactions",
    "mask_events",
    "masked",
    "mem_turns",
    "mem_warnings",
    "wc_edits",
    "archival_inserts",
    "archival_search_calls",
    "conv_search_calls",
    "conv_search_empty",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "wall_s",
)


async def main(args):
    if args.judge:
        load_config()
    rows = DatasetRegistry.load_dataset(args.dataset, args.split)
    if rows is None:
        raise SystemExit("dataset missing; run data/prepare_data.py")
    rows = [dict(r) for r in rows][: args.limit or None]
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r := json.loads(line))["id"]: r for line in f}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{len(rows)} questions, {len(done)} already done, running {len(todo)} with agent={args.agent}")

    use_judge = args.judge
    flow, sem, lock = FLOWS[args.agent], asyncio.Semaphore(args.concurrency), asyncio.Lock()
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out = open(args.out, "a")

    async def one(row, k):
        async with sem:
            cfg = AgentConfig(base_url=urls[k % len(urls)], model=args.model, session_uid=str(uuid.uuid4()))
            try:
                ep = await flow.arun(Task(id=row["id"], instruction=row["question"], metadata=row), cfg)
            except Exception as e:  # skip; a re-run with the same --out retries it
                print(f"{row['id']} failed: {e!r}", flush=True)
                return
            if ep.artifacts.get("termination") == "llm_error":  # server/connection failure: retry on the next run
                print(f"{row['id']} llm_error; not recorded", flush=True)
                return
            ev = (
                await asyncio.to_thread(judge_evaluator.evaluate, row, ep)
                if use_judge
                else f1_evaluator.evaluate(row, ep)
            )
            sig = {s.name: s.value for s in ev.signals}
            rec = {
                "id": row["id"],
                "gold": row["answer"],
                **ep.artifacts,
                "f1": sig.get("f1", 0.0),
                "em": sig.get("em", 0.0),
            }
            if use_judge:
                rec.update(judge=sig.get("judge", 0.0), judge_failed=sig.get("judge_failed", 0.0))
            if "multi_all_correct" in sig:
                rec["all_correct"] = sig["multi_all_correct"]
            async with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[row["id"]] = rec
                if len(done) % 20 == 0:
                    acc = f" acc={statistics.mean(r.get('judge', 0) for r in done.values()):.4f}" if use_judge else ""
                    print(
                        f"[{len(done)}/{len(rows)}] F1={statistics.mean(r['f1'] for r in done.values()):.4f}{acc}",
                        flush=True,
                    )

    await asyncio.gather(*(one(r, k) for k, r in enumerate(todo)))
    out.close()
    res = [done[r["id"]] for r in rows if r["id"] in done]
    if not res:
        print("incomplete: no questions completed; re-run to retry them", flush=True)
        raise SystemExit(2)
    summary = {
        "n": len(res),
        **(
            {"acc": statistics.mean(r["judge"] for r in res), "judge_failed": sum(r["judge_failed"] for r in res)}
            if res and all("judge" in r for r in res)
            else {}
        ),
        "f1": statistics.mean(r["f1"] for r in res),
        "em": statistics.mean(r["em"] for r in res),
        "answered": statistics.mean(r["answer"] is not None for r in res),
        "turns": statistics.mean(r["turns"] for r in res),
        "compactions": statistics.mean(r["compactions"] for r in res),
        **{k: statistics.mean(r[k] for r in res) for k in STAT_KEYS if res and all(k in r for r in res)},
        "termination": {t: sum(r["termination"] == t for r in res) for t in sorted({r["termination"] for r in res})},
    }
    print(json.dumps(summary, indent=2))
    if len(res) < len(rows):
        print(f"incomplete: {len(rows) - len(res)} questions missing; re-run to retry them", flush=True)
        raise SystemExit(2)
    if args.out.endswith(".jsonl"):
        path = args.out[: -len(".jsonl")] + ".summary.json"
        with open(path + ".tmp", "w") as f:
            json.dump({**summary, "agent": args.agent, "model": args.model}, f, indent=2)
        os.replace(path + ".tmp", path)
    if args.wandb_run:
        log_wandb(args, summary, res)


def log_wandb(args, summary: dict, res: list[dict]):
    import wandb

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project=os.environ.get("WANDB_PROJECT", "grepmem"),
        name=args.wandb_run,
        id=args.wandb_run.replace(".", "_"),
        resume="allow",
        job_type="eval",
        group="eval",
        config={"agent": args.agent, "dataset": args.dataset, "split": args.split, "model": args.model},
    )
    p = f"eval/{args.dataset}"
    metrics = {f"{p}/{k}": v for k, v in summary.items() if k != "termination"}
    metrics.update({f"{p}/termination/{t}": c / summary["n"] for t, c in summary["termination"].items()})
    run.log(metrics, step=args.step)
    run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="summary_grep", choices=list(FLOWS))
    ap.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="one or more comma-separated vLLM servers; questions are spread round-robin",
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="served model name")
    ap.add_argument("--dataset", default="browsecomp_plus_multi")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-judge", dest="judge", action="store_false", help="skip the LLM judge (report F1/EM only)")
    ap.add_argument(
        "--wandb-run", default="", help="separate wandb eval run name (older runs), e.g. qwen3.5-9b-react-bcp"
    )
    ap.add_argument("--step", type=int, default=0, help="training step of the evaluated checkpoint (wandb x-axis)")
    asyncio.run(main(ap.parse_args()))
