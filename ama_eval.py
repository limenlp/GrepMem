"""Evaluate a harness on AMA-Bench (arXiv 2602.22769, MIT license): memory over long agent trajectories.

  python ama_eval.py --agent summary_grep --protocol stream --base-url http://localhost:8000/v1 --out eval_ama_step0.jsonl
  python ama_eval.py --agent summary_grep --protocol file   --base-url http://localhost:8000/v1 --out eval_amafile_step0.jsonl

AMA-Bench's released test set (HF AMA-bench/AMA-bench, test/open_end_qa_set.jsonl) has 208 real agent trajectories
(six domains: web, software, text-to-SQL, games, embodied AI, open-world QA) with 12 open-ended questions each
(types A-D). Answers are judged yes/no by gpt-5.6-sol with AMA-Bench's judge prompt; acc is the fraction judged
correct. Two protocols:

  stream  The trajectory (steps of action + observation) is streamed into one conversation in chunks of about
          --chunk-chars characters, one chunk per user turn, through the same harnesses as ContextBench
          (agents/contextbench.CBAgent: compaction at COMPACT_AT, grep_history after the first compaction, react ends
          at its context limit). After the last chunk, each question is asked separately on a copy of the agent's
          state (AMA-Bench's per-query protocol). Default: trajectories of at most 128K tokens (189 episodes).
  file    The trajectory stays outside the context, like AMA-Bench's codex/claude_code runs: every harness can search
          it with grep_trajectory (the grep_history engine over the steps), and one session per episode answers all 12
          questions with numbered \\boxed{} answers (agents/flows.run_agent with a ToolEnv; AMA_FILE_TURNS turns).
          summary_grep additionally gets grep_history over its own raw context after the first compaction. All 208
          episodes.

One line per episode goes to --out; re-running with the same --out resumes. When all episodes are done, the summary
goes to <out>.summary.json (eval_ama_step<N> / eval_amafile_step<N>), which a training run started by train.sh logs
as eval/ama/* / eval/amafile/* (agents/eval_bridge.py). If episodes are missing (dropped LLM calls), no summary is
written and the exit code is 2; re-run to retry them.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import glob
import json
import os
import re
import statistics
import time
import uuid

from huggingface_hub.constants import HF_HUB_CACHE
from rllm.types import AgentConfig, Task

from agents.common import GREP_TOOL_SPEC, _judge, grep_call
from agents.contextbench import CBAgent
from agents.flows import InvalidCall, ToolEnv, run_agent
from agents.judge import load_config
from agents.memory import COST_STATS, MASK_STATS, MEMGPT_STATS

SYSTEM = "You are a careful assistant that reads long agent trajectories and answers questions about them."
INSTRUCTION = """You will read the trajectory of an AI agent working on a task, delivered in chunks of consecutive steps; each step shows the agent's action and the observation it received. After the last chunk you will be asked questions about the trajectory, for example what happened at a given step, why something happened, or how the state of the environment changed.

Task of the agent ({domain}): {task}

{note}

After each chunk, reply briefly (you may write down notes you want to keep); the next chunk then follows."""
NOTES = {
    "react": "Everything you read stays in your context; nothing is ever removed, and the run ends if your context window fills up.",
    "summary": (
        "When your context window fills up, it is compacted: you write a summary and the conversation restarts from these "
        "instructions plus your summary, so anything you need later must be in the conversation or in that summary."
    ),
    "summary_grep": (
        "When your context window fills up, it is compacted: you write a summary and the conversation restarts from these "
        "instructions plus your summary. After that, the full raw conversation (every chunk and reply, including what was "
        "removed from your context) is searchable with the grep_history tool."
    ),
    "mask": (
        "When your context window fills up, the oldest chunks in it are replaced by a short placeholder while your own "
        "replies are kept, so anything you need later must still be in the conversation or in your replies."
    ),
    "memgpt": (
        "You have memory tools. When your context window fills up, the oldest messages are removed from it and replaced by "
        "a summary (you are warned shortly before). Removed messages stay searchable with conversation_search, and you can "
        "keep notes in your working context (shown in the system prompt) and in archival memory."
    ),
}
SUMMARY_PROMPT = """Your context window is full and will be reset. Write a summary that lets you continue reading the trajectory from scratch with only the instructions and this summary, and later answer questions about the whole trajectory. Include:
1. Which steps you have read so far.
2. The key facts so far, with step numbers: actions, observations, state changes, errors and outcomes.
3. Anything else later questions may need, copied exactly.
Do not call any tools. Respond only with the summary."""
CHUNK = "[Trajectory chunk {k} of {n}]\n\n{text}"
QUESTION = """The trajectory is complete ({n_steps} steps). Please answer the following question based on the task description and the agent trajectory. Provide a direct and concise answer.

Question: {question}"""

# AMA-Bench's judge prompt (utils/evaluation_metrics.py: compute_llm_as_judge) with gpt-5.6-sol as the judge (the
# trailing "<think></think>" it adds for Qwen judges is dropped).
JUDGE_PROMPT = """You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context}

Question: {question}

Reference Answer: {gold}

Predicted Answer: {pred}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:"""


# ---------------------------------------------------------------- file protocol

FILE_TURNS = int(os.environ.get("AMA_FILE_TURNS", 60))
FILE_SYSTEM = """You answer questions about the trajectory of an AI agent. The trajectory is stored in a file that you can search with the tools below; it is not in your context.

Tools:
- grep_trajectory(pattern, max_matches, offset, context_lines, count): regex search over the trajectory file (every step's action and observation). Matches are grouped by step.
{extra_tools}
Guidelines:
- Look up the steps each question is about and check your answers against the trajectory.
- When you have answered all questions, stop calling tools and give your final answers in the requested format."""
FILE_PROMPT = """Task of the agent ({domain}): {task}
The trajectory has {n_steps} steps (Step {first} to Step {last}).

Please answer the following questions based on the task description and the agent trajectory. For each question, provide a direct and concise answer.

{questions}

When you have answered all of them, stop calling tools and give your final answers in this format, one line per question, in order (write your best guess if unsure):
{format}"""
FILE_SUMMARY_PROMPT = """Your context window is full and will be reset. Write a summary that lets you continue from scratch with only the questions and this summary. Include:
1. Your answer so far to each question (by number), with the steps that support it and how sure you are.
2. The questions not yet answered and what to look up for them.
Do not call any tools. Respond only with the summary."""
GREP_TRAJ_SPEC = copy.deepcopy(GREP_TOOL_SPEC)
GREP_TRAJ_SPEC["function"].update(
    name="grep_trajectory",
    description=(
        "Regex (Python re, case-insensitive) search over the agent trajectory file: every step's action and "
        "observation. Returns the number of matching lines and then the matches, in step order, grouped under "
        "their step. Page through many matches with offset; use count=true to get only the number of matching lines."
    ),
)


def file_env(ep: dict) -> tuple[ToolEnv, str]:
    """ToolEnv (grep_trajectory over the steps) and the question prompt for one episode."""
    steps = [
        f"[step {st.get('turn_idx')}]\nAction: {st.get('action', '')}\nObservation: {st.get('observation', '')}"
        for st in ep["trajectory"]
    ]

    async def grep_trajectory(args: dict) -> str:
        if not args.get("pattern"):
            raise InvalidCall
        return grep_call(steps, args)

    qs = [qa["question"] for qa in ep["qa_pairs"]]
    idx = [st.get("turn_idx") for st in ep["trajectory"]]
    prompt = FILE_PROMPT.format(
        domain=ep["domain"],
        task=ep["task"],
        n_steps=len(steps),
        first=idx[0] if idx else 0,
        last=idx[-1] if idx else 0,
        questions="\n\n".join(f"Question {i + 1}: {q}" for i, q in enumerate(qs)),
        format="\n".join(f"{i + 1}. \\boxed{{answer to question {i + 1}}}" for i in range(len(qs))),
    )
    env = ToolEnv(
        FILE_SYSTEM,
        [GREP_TRAJ_SPEC],
        {"grep_trajectory": (grep_trajectory, "trajectory_grep_calls")},
        summary_prompt=FILE_SUMMARY_PROMPT,
        resume_tail="Continue answering the questions.",
        max_turns=FILE_TURNS,
        n_questions=len(qs),
    )
    return env, prompt


async def run_file_episode(ep: dict, mode: str, base_url: str, model: str) -> dict:
    env, prompt = file_env(ep)
    cfg = AgentConfig(base_url=base_url, model=model, session_uid=str(uuid.uuid4()))
    res = await run_agent(
        Task(id=str(ep["episode_id"]), instruction=prompt, metadata={"question": prompt}), cfg, mode, env
    )
    a = res.artifacts
    if a["termination"] == "llm_error":
        raise RuntimeError("llm_error")
    preds = list(a.get("answers") or []) + [None] * len(ep["qa_pairs"])

    async def score(qa: dict, pred: str | None) -> dict:
        s = await asyncio.to_thread(judge, qa["question"], qa["answer"], pred, ep) if pred else 0.0
        return {"uuid": qa.get("question_uuid"), "type": qa.get("type"), "score": s, "answer": (pred or "")[:2000]}

    qas = await asyncio.gather(*(score(qa, p) for qa, p in zip(ep["qa_pairs"], preds)))
    failed = sum(q["score"] is None for q in qas)
    for q in qas:
        q["score"] = q["score"] or 0.0
    stat_keys = (
        "turns",
        "llm_calls",
        "compactions",
        "grep_calls",
        "grep_turns",
        "trajectory_grep_calls",
        "tool_errors",
        "max_context",
    ) + COST_STATS
    return {
        "id": str(ep["episode_id"]),
        "domain": ep["domain"],
        "task_type": ep["task_type"],
        "total_tokens": int(ep["total_tokens"]),
        "n_steps": len(ep["trajectory"]),
        "terminated_by": a["termination"],
        "acc": statistics.mean(q["score"] for q in qas),
        "answered": sum(p is not None for p in preds[: len(qas)]) / len(qas),
        "judge_failed": failed,
        **{k: a[k] for k in stat_keys},
        **{k: a[k] for k in MASK_STATS + MEMGPT_STATS if k in a},
        "qa": qas,
    }


# ---------------------------------------------------------------- stream protocol


class AMAAgent(CBAgent):
    summary_prompt = SUMMARY_PROMPT

    def harness_instruction(self, instruction: str) -> str:
        return instruction  # already contains the harness note


def chunk_trajectory(trajectory: list[dict], chunk_chars: int) -> list[str]:
    """Pack steps ("Step i:\\nAction: ...\\nObservation: ...") into chunks of about chunk_chars; longer steps are split."""
    pieces = []
    for st in trajectory:
        text = f"Step {st.get('turn_idx')}:\nAction: {st.get('action', '')}\nObservation: {st.get('observation', '')}\n"
        if len(text) <= chunk_chars:
            pieces.append(text)
            continue
        parts = [text[i : i + chunk_chars] for i in range(0, len(text), chunk_chars)]
        pieces += [f"Step {st.get('turn_idx')} (part {j + 1} of {len(parts)}):\n{p}\n" for j, p in enumerate(parts)]
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) > chunk_chars:
            chunks.append(cur)
            cur = ""
        cur += p + "\n"
    return chunks + ([cur] if cur else [])


def judge(question: str, gold: str, pred: str, ep: dict) -> float | None:
    context = f"Task Type: {ep['task_type']}\nEpisode ID: {ep['episode_id']}\nTask Context: {ep['task']}"
    try:
        client, model = _judge()
        resp = client.chat.completions.create(
            model=model,
            max_completion_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(context=context, question=question, gold=gold, pred=pred),
                }
            ],
        )
        text = (resp.choices[0].message.content or "").lower()
    except Exception as e:  # noqa: BLE001
        print(f"judge failed ({type(e).__name__})", flush=True)
        return None
    m = re.search(r"\b(yes|no)\b", text)
    return None if m is None else float(m.group(1) == "yes")


def default_data() -> str:
    hub = HF_HUB_CACHE
    paths = sorted(glob.glob(f"{hub}/datasets--AMA-bench--AMA-bench/snapshots/*/test/open_end_qa_set.jsonl"))
    return paths[-1] if paths else ""


async def run_episode(ep: dict, agent: AMAAgent, chunk_chars: int, qsem: asyncio.Semaphore) -> dict:
    chunks = chunk_trajectory(ep["trajectory"], chunk_chars)
    t0 = time.monotonic()
    agent.reset(SYSTEM, INSTRUCTION.format(domain=ep["domain"], task=ep["task"], note=NOTES[agent.mode]))
    for k, text in enumerate(chunks):
        await agent.on_user_turn(CHUNK.format(k=k + 1, n=len(chunks), text=text))
        if agent.exhausted:
            break
    wall = agent.exhausted
    stream_stats = dict(agent.stats)

    async def ask(qa: dict) -> dict:
        async with qsem:
            if wall:
                pred = ""
                stats = dict.fromkeys(agent.stats, 0)
            else:
                fork = agent.fork()
                pred = (
                    await fork.on_user_turn(QUESTION.format(n_steps=len(ep["trajectory"]), question=qa["question"]))
                ).strip()
                stats = fork.stats
        score = await asyncio.to_thread(judge, qa["question"], qa["answer"], pred, ep) if pred else 0.0
        return {
            "uuid": qa.get("question_uuid"),
            "type": qa.get("type"),
            "score": score,
            "answer": pred[:2000],
            "stats": stats,
        }

    qas = await asyncio.gather(*(ask(qa) for qa in ep["qa_pairs"]))
    wall_s = round(time.monotonic() - t0, 2)  # stream and questions (answered concurrently), not the judge
    q_stats = {k: sum(q["stats"][k] for q in qas) for k in stream_stats}
    q_stats["max_context"] = max(q["stats"]["max_context"] for q in qas) if qas else 0
    failed = sum(q["score"] is None for q in qas)
    for q in qas:
        q["score"] = q["score"] or 0.0
        del q["stats"]
    return {
        "id": str(ep["episode_id"]),
        "domain": ep["domain"],
        "task_type": ep["task_type"],
        "total_tokens": int(ep["total_tokens"]),
        "n_steps": len(ep["trajectory"]),
        "n_chunks": len(chunks),
        "terminated_by": "budget_wall" if wall else "done",
        "acc": statistics.mean(q["score"] for q in qas),
        "judge_failed": failed,
        **{
            k: stream_stats[k] + q_stats[k]
            for k in (
                "llm_calls",
                "compactions",
                "grep_calls",
                "tool_errors",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
            )
            + MASK_STATS
            + MEMGPT_STATS
            if k in stream_stats
        },
        "wall_s": wall_s,
        "question_grep_calls": q_stats["grep_calls"],
        "max_context": max(stream_stats["max_context"], q_stats["max_context"]),
        "qa": qas,
    }


def bucket(tokens: int) -> str:
    return "le32k" if tokens <= 32768 else ("32k_128k" if tokens <= 131072 else "gt128k")


def summarize(res: list[dict]) -> dict:
    qs = [(r, q) for r in res for q in r["qa"]]
    s = {"n": len(res), "n_qa": len(qs), "acc": statistics.mean(q["score"] for _, q in qs)}
    for name, key in (
        ("domain", lambda r, q: r["domain"]),
        ("type", lambda r, q: q["type"]),
        ("len", lambda r, q: bucket(r["total_tokens"])),
    ):
        for v in sorted({key(r, q) for r, q in qs}):
            s[f"{name}/{v}/acc"] = statistics.mean(q["score"] for r, q in qs if key(r, q) == v)
    s["judge_failed"] = sum(r["judge_failed"] for r in res)
    s["termination"] = {t: sum(r["terminated_by"] == t for r in res) for t in sorted({r["terminated_by"] for r in res})}
    for k in sorted({k for r in res for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}):
        if k not in ("total_tokens", "acc", "judge_failed") and all(k in r for r in res):
            s[k] = statistics.mean(r[k] for r in res)  # per-episode means: compactions, grep calls, LLM calls, ...
    return s


async def main(args):
    load_config()
    data = args.data or default_data()
    if not data:
        raise SystemExit("AMA-Bench data not found; run: hf download AMA-bench/AMA-bench --repo-type dataset")
    eps = [json.loads(line) for line in open(data)]
    max_tokens = args.max_tokens if args.max_tokens is not None else (131072 if args.protocol == "stream" else 0)
    eps = [e for e in eps if not max_tokens or int(e["total_tokens"]) <= max_tokens][: args.limit or None]
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {(r := json.loads(line))["id"]: r for line in f}
    todo = [e for e in eps if str(e["episode_id"]) not in done]
    print(
        f"{len(eps)} episodes, {len(done)} already done, running {len(todo)} with agent={args.agent} protocol={args.protocol}",
        flush=True,
    )
    urls = [u.strip() for u in args.base_url.split(",") if u.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sem, qsem, lock, out = (
        asyncio.Semaphore(args.concurrency),
        asyncio.Semaphore(4 * args.concurrency),
        asyncio.Lock(),
        open(args.out, "a"),
    )

    async def one(ep, k):
        async with sem:
            try:
                if args.protocol == "file":
                    rec = await run_file_episode(ep, args.agent, urls[k % len(urls)], args.model)
                else:
                    rec = await run_episode(
                        ep, AMAAgent(args.agent, urls[k % len(urls)], args.model), args.chunk_chars, qsem
                    )
            except Exception as e:  # noqa: BLE001  (dropped LLM call etc.: retried by the next run)
                print(f"{ep['episode_id']} failed: {e!r}", flush=True)
                return
            async with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                done[rec["id"]] = rec
                print(
                    f"[{len(done)}/{len(eps)}] {rec['id']} {rec['domain']} acc={rec['acc']:.3f} {rec['terminated_by']} "
                    f"llm_calls={rec['llm_calls']} compactions={rec['compactions']} greps={rec['grep_calls']}",
                    flush=True,
                )

    todo.sort(key=lambda e: -int(e["total_tokens"]))  # longest first
    await asyncio.gather(*(one(e, k) for k, e in enumerate(todo)))
    out.close()
    res = [done[str(e["episode_id"])] for e in eps if str(e["episode_id"]) in done]
    summary = summarize(res) if res else {"n": 0}
    print(json.dumps(summary, indent=2))
    if len(res) < len(eps):
        print(f"incomplete: {len(eps) - len(res)} episodes missing; re-run to retry them", flush=True)
        raise SystemExit(2)
    if args.out.endswith(".jsonl"):
        path = args.out[: -len(".jsonl")] + ".summary.json"
        with open(path + ".tmp", "w") as f:
            json.dump(
                {
                    **summary,
                    "agent": args.agent,
                    "model": args.model,
                    "protocol": args.protocol,
                    "max_tokens": max_tokens,
                    "chunk_chars": args.chunk_chars,
                    "file_turns": FILE_TURNS,
                },
                f,
                indent=2,
            )
        os.replace(path + ".tmp", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="summary_grep", choices=list(NOTES))
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="one or more comma-separated vLLM servers")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B", help="served model name")
    ap.add_argument("--data", default="", help="open_end_qa_set.jsonl (default: the HF cache under $HF_HOME)")
    ap.add_argument("--protocol", default="stream", choices=["stream", "file"])
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ["AMA_MAX_TOKENS"]) if os.environ.get("AMA_MAX_TOKENS") else None,
        help="skip longer trajectories (default: 131072 for stream, no limit for file; 0 = no limit)",
    )
    ap.add_argument("--chunk-chars", type=int, default=int(os.environ.get("AMA_CHUNK_CHARS", 12000)))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=64, help="episodes in flight")
    ap.add_argument("--out", required=True)
    asyncio.run(main(ap.parse_args()))
