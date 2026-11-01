"""ContextBench (arXiv 2609.37725) with five context management strategies.

Adapted from the Apache-2.0 ContextBench runner; see NOTICE.md for attribution.

ContextBench streams more data than the context window holds (up to 24x a 32K window) into one conversation,
one operation per user turn, and then asks exact-recall questions (KV Store: GET a random value; Log Triage:
look up a log line's message or count matching lines). The agent advances the stream by ending its reply with
READY_FOR_NEXT_OP, and is graded on the ANSWER blocks that appear in its per-turn context snapshots.

``run_instance`` is an async copy of ``contextbench.runner.run`` (same op rendering, readiness rules, refusal
texts, o200k budget wall, per-turn snapshots and official graders) so many episodes can share one vLLM server.
``CBAgent`` runs the harnesses with the settings of ``agents/flows.py`` (window, compaction threshold, turn
length, grep caps):

  react         no context management; the run ends at the context limit (budget wall).
  summary       Codex-style compaction at COMPACT_AT: the model writes a summary and the context restarts from
                [system, instructions + summary] (+ the operation being handled).
  summary_grep  summary + grep_history(pattern) over the full raw conversation, offered after the first
                compaction (before that, the history is identical to the context).
  mask          baseline: at COMPACT_AT the oldest streamed messages (user turns) are replaced by a placeholder;
                the model's replies are kept (agents/memory.py).
  memgpt        baseline: MemGPT-style memory tools, memory-pressure warning, FIFO flush with a recursive summary
                (agents/memory.py).

No method-specific strategy ("skill") is given, unlike the paper's arms: each harness gets one neutral sentence
describing what happens to its context, so the scores show what the model does on its own.

Every LLM call is recorded as a Step, so ``run_cb_episode`` can also serve as an rLLM training flow (dataset rows
with kind=cb, generated with fresh seeds by data/prepare_data.py). A run ends early ("stuck") after
BCRL_CB_STUCK consecutive replies that do not advance the stream.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

from contextbench import stream as proto
from contextbench.graders import grade_run
from contextbench.runner import DEFAULT_SYSTEM_PROMPT, instruction_for, load_instance
from contextbench.stream import render_op
from contextbench.tokens import count_tokens, record_snapshot
from openai import AsyncOpenAI
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

from . import memory as mem
from .common import GREP_TOOL_SPEC, grep_call, msg_to_dict, normalize_message, parse_args, row_extra, truncate
from .flows import (
    COMPACT_AT,
    CONTEXT_BUDGET,
    FIT_MARGIN,
    LLM_TIMEOUT,
    SAMPLING,
    TRAJ_NAME,
    TURN_TOKENS,
    _est_tokens,
    _fit,
    fit_compaction_prompt,
)

logger = logging.getLogger(__name__)

# LLM calls allowed per user turn (an op plus grep calls); a reply without the marker is nudged by the runner.
CALLS_PER_TURN = int(os.environ.get("BCRL_CB_CALLS_PER_TURN", 8))
# Consecutive replies that do not advance the stream (no marker, refused submit, ...) before the run ends as "stuck".
STUCK_TURNS = int(os.environ.get("BCRL_CB_STUCK", 5))

SUMMARY_PROMPT = """Your context window is full and will be reset. Write a summary that lets you continue the task from scratch with only the task instructions and this summary. Include:
1. What you have done so far (which operations you have handled).
2. Everything later operations may need, copied exactly.
3. What to do next.
Do not call any tools. Respond only with the summary."""

RESUME_TEMPLATE = """{instruction}

---
Your earlier context was compacted. Summary of your progress so far:

{summary}
{grep_note}"""

GREP_NOTE = "\n(The full raw history of this conversation, including what was removed from your context, is searchable with grep_history.)\n"
GREP_TOOL_PARAGRAPH = (
    "\n\nTools:\n- grep_history(pattern, max_matches, offset, context_lines, count): regex search over your full, "
    "uncompressed conversation history (every operation and reply so far), including parts removed from your "
    "context by summarization. Matches are labeled with the turn they came from; count=true only counts them."
)

# The reply-mode instruction says the agent has no tools and must keep everything in the conversation; each
# harness replaces that sentence with one describing what actually happens to its context.
NO_TOOLS_SENTENCE = (
    "You have no shell, no files and no tools in this setting: anything you want to\n"
    "recall later must still be in this conversation when you need it."
)
HARNESS_NOTES = {
    "react": NO_TOOLS_SENTENCE + " Nothing is ever removed from your context; the run ends when it is full.",
    "summary": (
        "You have no shell, no files and no tools in this setting. When your context fills up, it is compacted: you\n"
        "write a summary and the conversation restarts from these instructions plus your summary, so anything you\n"
        "want to recall later must be in this conversation or in that summary."
    ),
    "summary_grep": (
        "You have no shell and no files in this setting. When your context fills up, it is compacted: you write a\n"
        "summary and the conversation restarts from these instructions plus your summary. After that, the full raw\n"
        "conversation (every operation and reply, including what was removed from your context) is searchable\n"
        "with the grep_history tool."
    ),
    "mask": (
        "You have no shell, no files and no tools in this setting. When your context fills up, the oldest messages\n"
        "you received are replaced by a short placeholder while your own replies are kept, so anything you want to\n"
        "recall later must still be in this conversation or in your replies."
    ),
    "memgpt": (
        "You have no shell and no files in this setting, but you have memory tools. When your context fills up, the\n"
        "oldest messages are removed from it and replaced by a summary (you are warned shortly before). Removed\n"
        "messages stay searchable with conversation_search, and you can keep notes in your working context (shown\n"
        "in the system prompt) and in archival memory."
    ),
}
MEMORY_TOOL_PARAGRAPH = "\n\nTools:\n" + mem.MEMORY_TOOL_DOC.rstrip("\n")


def harness_instruction(instruction: str, mode: str) -> str:
    if NO_TOOLS_SENTENCE not in instruction:
        raise ValueError("ContextBench instruction changed: the no-tools sentence was not found")
    return instruction.replace(NO_TOOLS_SENTENCE, HARNESS_NOTES[mode])


class CBAgent:
    """One harness episode on a ContextBench stream (the runner's ContextAgent protocol, async)."""

    summary_prompt = SUMMARY_PROMPT

    def __init__(self, mode: str, base_url: str, model: str, max_retries: int = 3):
        assert mode in HARNESS_NOTES, mode
        self.mode, self.model = mode, model
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=LLM_TIMEOUT, max_retries=max_retries)
        self.stats = {
            "llm_calls": 0,
            "compactions": 0,
            "grep_calls": 0,
            "grep_empty": 0,
            "grep_chars": 0,
            "tool_errors": 0,
            "max_context": 0,
            "trimmed_turns": 0,
            **dict.fromkeys(mem.COST_STATS, 0),
        }
        self.stats.update(
            dict.fromkeys(mem.MASK_STATS if mode == "mask" else mem.MEMGPT_STATS if mode == "memgpt" else (), 0)
        )
        self.memory: mem.MemGPTMemory | None = None
        self.exhausted = False  # react hit its context limit
        self.steps: list[Step] = []  # one per LLM call (rLLM maps them positionally to gateway traces)
        self.grep_patterns: list[str] = []
        self.turn = 0

    def reset(self, system: str, instruction: str) -> None:
        self.base_system = system
        self.instruction = self.harness_instruction(instruction)
        self.grep_on = False
        self.memory = mem.MemGPTMemory() if self.mode == "memgpt" else None
        self.warned = False  # memgpt: memory-pressure warning given since the last flush
        self.messages: list[dict] = [self._system(), {"role": "user", "content": self.instruction}]
        self.history: list[str] = []
        self.ctx = _est_tokens(system + self.instruction) + 200
        self.scored_tokens, self.n_scored = 0, 0

    def harness_instruction(self, instruction: str) -> str:
        return harness_instruction(instruction, self.mode)

    def fork(self) -> "CBAgent":
        """A copy of the current conversation state with fresh stats (e.g. one per question)."""
        a = copy.copy(self)
        a.messages, a.history = list(self.messages), list(self.history)
        a.memory = self.memory.copy() if self.memory is not None else None
        a.stats, a.steps, a.grep_patterns = dict.fromkeys(self.stats, 0), [], []
        return a

    def _system(self) -> dict:
        if self.memory is not None:  # the working-context block is (re)rendered here: at reset and at each flush
            return {"role": "system", "content": self.base_system + MEMORY_TOOL_PARAGRAPH + self.memory.block()}
        return {"role": "system", "content": self.base_system + (GREP_TOOL_PARAGRAPH if self.grep_on else "")}

    def context(self) -> list[dict]:
        # Qwen3.5's chat template drops the reasoning of earlier turns, so between turns it is not in the live context.
        return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in self.messages]

    async def _call(self, msgs: list[dict], use_tools: bool = True):
        tools = [GREP_TOOL_SPEC] if self.grep_on else (mem.MEMORY_TOOL_SPECS if self.memory is not None else None)
        tools = tools if use_tools else None
        kwargs = {"tools": tools} if tools else {}
        resp = await self.client.chat.completions.create(model=self.model, messages=msgs, **SAMPLING, **kwargs)
        self.stats["llm_calls"] += 1
        self.steps.append(Step(chat_completions=list(msgs), model_response=resp.choices[0].message.content or ""))
        usage = resp.usage
        mem.add_usage(self.stats, usage)
        prompt = usage.prompt_tokens if usage else self.ctx
        total = prompt + (usage.completion_tokens if usage else 0)
        self.stats["max_context"] = max(self.stats["max_context"], total)
        return resp.choices[0], total, prompt

    async def _compact(self, current_op: dict | None) -> None:
        budget = COMPACT_AT - _est_tokens(self.summary_prompt) - 64
        msgs = self.messages
        if current_op is not None:  # summarize what came before the op being handled; the op itself is kept
            msgs = msgs[: next(i for i in range(len(msgs) - 1, -1, -1) if msgs[i] is current_op)]
        fitted, trimmed = (
            fit_compaction_prompt(msgs, budget, self.scored_tokens, self.n_scored) if self.n_scored else (msgs, 0)
        )
        self.stats["trimmed_turns"] += trimmed
        choice, _, _ = await self._call(fitted + [{"role": "user", "content": self.summary_prompt}])
        cmsg = normalize_message(choice.message)
        summary = mem.reply_text(cmsg)
        self.stats["compactions"] += 1
        self.history.append(f"[compaction {self.stats['compactions']} summary]\n{summary}")
        if self.mode == "summary_grep":
            self.grep_on = True
        resume = RESUME_TEMPLATE.format(
            instruction=self.instruction, summary=summary, grep_note=GREP_NOTE if self.grep_on else ""
        )
        self.messages = [self._system(), {"role": "user", "content": resume}] + (
            [current_op] if current_op is not None else []
        )
        self.ctx = (
            _est_tokens(self.messages[0]["content"] + resume)
            + 200
            + (_est_tokens(current_op["content"]) if current_op else 0)
        )
        self.n_scored = 0

    def _index(self, op: dict | None) -> int:
        """Position of the operation being handled (len(messages) if it is not in the context yet)."""
        return next((i for i in range(len(self.messages) - 1, -1, -1) if self.messages[i] is op), len(self.messages))

    def _mask(self, current_op: dict | None) -> None:
        scale = mem.token_scale(self.messages, self.scored_tokens, self.n_scored)
        self.messages, self.ctx = mem.make_room_by_masking(
            self.messages,
            self.ctx,
            COMPACT_AT,
            TURN_TOKENS,
            scale,
            self._index(current_op),
            ("user",),
            mem.MSG_PLACEHOLDER,
            "user",
            self.stats,
        )
        self.scored_tokens, self.n_scored = int(scale * mem.ctx_chars(self.messages)), len(self.messages)

    async def _flush(self, current_op: dict | None) -> None:
        """MemGPT queue flush: evict the oldest messages (whole operations) and write a recursive summary of them."""
        scale = mem.token_scale(self.messages, self.scored_tokens, self.n_scored)
        cut = mem.flush_cut(self.messages, scale, mem.KEEP * COMPACT_AT, "user", self._index(current_op))
        if cut is None:
            return
        fitted, trimmed = _fit(self.messages[:cut], COMPACT_AT - _est_tokens(mem.SUMMARY_PROMPT) - 64, scale)
        self.stats["trimmed_turns"] += trimmed
        choice, _, _ = await self._call(fitted + [{"role": "user", "content": mem.SUMMARY_PROMPT}], use_tools=False)
        summary = mem.reply_text(normalize_message(choice.message))
        self.stats["compactions"] += 1
        self.history.append(f"[memory summary {self.stats['compactions']}]\n{summary}")
        head = mem.HEAD_TEMPLATE.format(task=self.instruction, n=cut - 2, summary=summary)
        self.messages = [self._system(), {"role": "user", "content": head}] + self.messages[cut:]
        self.ctx = int(scale * mem.ctx_chars(self.messages)) + 200
        self.n_scored = 0
        self.warned = False

    async def _make_room(self, current_op: dict | None) -> None:
        if self.mode == "mask":
            self._mask(current_op)
        elif self.mode == "memgpt":
            scale = mem.token_scale(self.messages, self.scored_tokens, self.n_scored)
            await self._flush(current_op)
            if current_op is not None and self.ctx > COMPACT_AT:
                # The flush evicts only what came before the operation being handled; its own chain of tool calls
                # (up to CALLS_PER_TURN replies, reasoning included) can still overflow the window.
                self.messages, self.ctx, dropped = mem.trim_turn(
                    self.messages,
                    self.ctx,
                    COMPACT_AT - TURN_TOKENS,
                    COMPACT_AT - FIT_MARGIN,
                    scale,
                    self._index(current_op) + 1,
                )
                self.stats["trimmed_turns"] += dropped
                self.scored_tokens, self.n_scored = int(scale * mem.ctx_chars(self.messages)), len(self.messages)
        else:
            await self._compact(current_op)

    async def on_user_turn(self, text: str) -> str:
        op = {"role": "user", "content": text}
        op_tokens = _est_tokens(text) + 10
        if self.mode != "react" and self.ctx + op_tokens > COMPACT_AT and len(self.messages) > 2:
            await self._make_room(None)
        if self.memory is not None and not self.warned and self.ctx + op_tokens > mem.WARN * COMPACT_AT:
            op["content"] = (
                text + "\n\n" + mem.WARNING.format(pct=round(100 * (self.ctx + op_tokens) / COMPACT_AT))
            )  # not in the history
            op_tokens += 60
            self.warned = True
            self.stats["mem_warnings"] += 1
        self.messages.append(op)
        self.turn += 1
        self.history.append(f"[turn {self.turn} user]\n{text}")
        self.ctx += op_tokens
        content = ""
        for _ in range(CALLS_PER_TURN):
            if self.mode == "react" and self.ctx + TURN_TOKENS > CONTEXT_BUDGET:
                self.exhausted = True
                return ""
            if self.mode != "react" and self.ctx > COMPACT_AT:
                await self._make_room(op)
            choice, total, prompt = await self._call(self.messages)
            msg = normalize_message(choice.message)
            self.messages.append(msg_to_dict(msg))
            self.scored_tokens, self.n_scored = total, len(self.messages)
            content = msg.content or ""
            self.history.append(f"[turn {self.turn} assistant]\n{content}")
            if not msg.tool_calls:
                # The next user turn drops this reply's reasoning from the prompt (chat template).
                self.ctx = prompt + _est_tokens(content) + 20
                return content
            new_tokens = 0
            for tc in msg.tool_calls:
                args = parse_args(tc.function.arguments)
                if tc.function.name == "grep_history" and self.grep_on and args.get("pattern"):
                    out = grep_call(self.history, args)
                    self.grep_patterns.append(str(args["pattern"]))
                    self.stats["grep_calls"] += 1
                    self.stats["grep_empty"] += out.startswith("No matches")
                    self.stats["grep_chars"] += len(out)
                elif (
                    self.memory is not None
                    and (res := self.memory.call(tc.function.name, args, self.history)) is not None
                ):
                    out, key = res
                    self.stats[key] += 1
                    self.stats["conv_search_empty"] += key == "conv_search_calls" and out.startswith("No messages")
                else:
                    out = f"Error: invalid tool call {tc.function.name}({tc.function.arguments})."
                    self.stats["tool_errors"] += 1
                out = truncate(out)
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
                self.history.append(f"[turn {self.turn} tool {tc.function.name}({tc.function.arguments})]\n{out}")
                new_tokens += _est_tokens(out) + 20
            self.ctx = total + new_tokens
        return content  # too many calls in one turn; the runner nudges for the marker


async def run_instance(
    agent: CBAgent,
    instance_dir: str | Path,
    out_dir: str | Path,
    *,
    budget: int = 32768,
    reserve: int = 2048,
    max_turns: int | None = None,
) -> dict:
    """Async ``contextbench.runner.run``: deliver ops, enforce the budget wall, snapshot, grade."""
    instance = load_instance(instance_dir)
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    usable = budget - reserve
    t0 = time.monotonic()
    agent.reset(DEFAULT_SYSTEM_PROMPT, instruction_for(instance, delivery="reply"))
    total_ops, delivered, step, peak = instance.n_ops, 0, 0, 0
    max_turns = max_turns or 4 * total_ops + 50
    terminated_by = None
    stalled = 0  # consecutive replies that did not advance the stream

    def deliver(i: int) -> str:
        return render_op(instance.ops[i], remaining=total_ops - (i + 1))

    pending = deliver(0) if total_ops else None
    delivered = 1 if total_ops else 0
    while pending is not None and terminated_by is None:
        reply = await agent.on_user_turn(pending) or ""
        step += 1
        messages = agent.context()
        tokens, _ = count_tokens(messages)
        peak = max(peak, tokens)
        record_snapshot(
            out,
            step,
            messages,
            {"n_ops_delivered": delivered, "n_ops_total": total_ops, "budget": budget, "reserve": reserve},
        )
        if tokens > usable or agent.exhausted:
            terminated_by = "budget_wall"
            break
        stalled += 1
        if proto.is_submit_line(reply):
            if delivered < total_ops:
                pending = proto.REFUSE_SUBMIT_TEXT
            else:
                terminated_by = "submit"
        elif proto.is_ready_line(reply):
            if delivered < total_ops:
                pending, delivered, stalled = deliver(delivered), delivered + 1, 0
            else:
                terminated_by = "stream_exhausted"
        elif proto.mentions_advance_marker(reply):
            pending = proto.EXPLAIN_NON_STANDALONE_TEXT
        else:
            pending = proto.NO_MARKER_TEXT
        if terminated_by is None and step >= max_turns:
            terminated_by = "max_steps"
        if terminated_by is None and stalled >= STUCK_TURNS:
            terminated_by = "stuck"
    terminated_by = terminated_by or "stream_exhausted"
    agent.stats["wall_s"] = round(time.monotonic() - t0, 2)
    record = {"n_ops_delivered": delivered, "n_ops_total": total_ops, "terminated_by": terminated_by}
    (out / "interactive_stream.json").write_text(json.dumps(record, indent=2) + "\n")
    rewards = await asyncio.to_thread(grade_run, out, instance.gold)
    diag = (
        await asyncio.to_thread(kv_diagnostics, out, instance.gold, agent.grep_patterns)
        if instance.family == "kv_store"
        else {}
    )
    result = {
        "task_family": instance.family,
        "n_turns": step,
        "peak_tokens": peak,
        "reward": rewards.get("reward", 0.0),
        "rewards": rewards,
        **record,
        **agent.stats,
        **diag,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def kv_diagnostics(run_dir: Path, gold: dict, grep_patterns: list[str]) -> dict:
    """Why KV queries failed: answered without the value's #hash tag, wrong value, or unanswered; and accuracy for
    keys the agent did / did not grep for (answering from memory instead of looking the key up)."""
    from contextbench.graders.base import normalize, read_context
    from contextbench.graders.kv import KVStoreGrader

    answers = KVStoreGrader._collect(read_context(run_dir))
    n = {
        "kv_hash_dropped": 0,
        "kv_wrong": 0,
        "kv_unanswered": 0,
        "kv_grepped": 0,
        "kv_correct_grepped": 0,
        "kv_correct_ungrepped": 0,
    }
    for q in gold.get("queries") or []:
        key, want = normalize(str(q["key"])), normalize(str(q["value"]))
        seen = answers.get(key) or set()
        ok = any(want in a for a in seen)
        grepped = any(str(q["key"]).lower() in p.lower() for p in grep_patterns)
        n["kv_grepped"] += grepped
        n["kv_correct_grepped" if grepped else "kv_correct_ungrepped"] += ok
        if ok:
            continue
        core = want.rsplit("#", 1)[0].strip()
        if not seen:
            n["kv_unanswered"] += 1
        elif core and any(core in a for a in seen):
            n["kv_hash_dropped"] += 1
        else:
            n["kv_wrong"] += 1
    return n


# ---------------------------------------------------------------- training flow (dataset rows with kind=cb)

CB_TRAIN_DIR = os.environ.get("BCRL_CB_TRAIN_DIR", str(Path(".cache/grepmem/cb_train").resolve()))


def generate_instance(family: str, knobs: dict, dest: Path) -> Path:
    """Generate one ContextBench instance into ``dest`` (once; concurrent callers race safely)."""
    if not (dest / "meta.json").exists():
        mod = importlib.import_module(f"contextbench.tasks.{family}.generate")
        tmp = dest.with_name(f"{dest.name}.tmp{os.getpid()}-{uuid.uuid4().hex[:6]}")
        shutil.rmtree(tmp, ignore_errors=True)
        mod.write_outputs(tmp, mod.generate(**knobs))
        try:
            os.rename(tmp, dest)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            if not (dest / "meta.json").exists():
                raise
    return dest


async def run_cb_episode(task: Task, config: AgentConfig, mode: str) -> Episode:
    """One ContextBench stream as an rLLM episode; the reward (official grader score) goes to artifacts["cb_reward"]."""
    meta = task.metadata or {}
    x = row_extra(meta)
    root = Path(CB_TRAIN_DIR)
    root.mkdir(parents=True, exist_ok=True)
    inst = await asyncio.to_thread(
        generate_instance, x["family"], x["knobs"], root / "instances" / str(meta["id"]).replace("/", "-")
    )
    agent = CBAgent(
        mode, config.base_url, config.model, max_retries=1
    )  # as in flows.py: retries can add gateway traces
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    try:
        res = await run_instance(agent, inst, run_dir, max_turns=x.get("max_turns"))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    stats = {k: v for k, v in res.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    return Episode(
        trajectories=[Trajectory(name=TRAJ_NAME, steps=agent.steps)],
        artifacts={"answer": None, "termination": res["terminated_by"], "cb_reward": float(res["reward"]), **stats},
        metrics={f"cb_{k}": float(v) for k, v in stats.items() if k != "reward"},
    )
