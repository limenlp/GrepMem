"""Deep-research harnesses sharing one tool-calling loop.

  react         plain ReAct; the episode ends (unanswered) when the context budget is hit.
  summary       Codex-style compaction: at COMPACT_AT tokens the model writes a progress
                summary and the context restarts from [system, question + summary].
  summary_grep  summary + the full raw history is kept outside the context and exposed
                through a grep_history(pattern) tool, offered only after the first compaction
                (before that, the history is identical to the context).
  mask          baseline: observation masking -- at COMPACT_AT the oldest tool outputs are replaced
                by a placeholder (agents/memory.py).
  memgpt        baseline: MemGPT-style memory -- working context, archival memory, conversation_search,
                memory-pressure warning, FIFO flush with a recursive summary (agents/memory.py).

The harnesses other than react also see how many of their MAX_TURNS turns are left (BCRL_SHOW_BUDGET). Replies that
only call grep_history (memgpt: only memory tools) do not count toward MAX_TURNS (at most BCRL_GREP_FREE_TURNS of them
per episode), so looking something up in the history does not cost a research turn. An episode that uses up its turns without a
final answer, or whose final reply is cut off at TURN_TOKENS, gets one more call without tools that asks for the
(best-guess) answer now (termination "forced_answer"; BCRL_FORCE_ANSWER=0 turns this off). react still ends
unanswered at its context limit.

Rows with kind=multi hold several questions answered together at the end (numbered \\boxed{} answers, budget
BCRL_MULTI_TURNS_PER_Q turns per question); rows with kind=cb are ContextBench streams (agents/contextbench.py).
Numeric episode stats (tool calls, grep use, compactions) go to Episode.metrics, which rLLM logs as episode/*.

Each LLM call is one Step (steps map positionally to gateway traces). With
``rllm.gateway.cumulative_token_mode=true`` contiguous turns merge into one training
row; a compaction starts a new row and all rows share the episode's advantage.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import rllm
from openai import AsyncOpenAI
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

from . import memory as mem
from .common import (
    GREP_TOOL_DOC,
    GREP_TOOL_SPEC,
    SYSTEM_PROMPT,
    TOOL_SPECS,
    extract_answers,
    extract_boxed,
    get_document,
    grep_call,
    msg_to_dict,
    normalize_message,
    parse_args,
    row_extra,
    search,
    truncate,
)

logger = logging.getLogger(__name__)

MAX_TURNS = int(os.environ.get("BCRL_MAX_TURNS", 30))
MULTI_TURNS_PER_Q = int(os.environ.get("BCRL_MULTI_TURNS_PER_Q", 20))  # kind=multi rows: turn budget per question
GREP_FREE_TURNS = int(os.environ.get("BCRL_GREP_FREE_TURNS", 10))  # grep-only replies not counted toward the turn cap
FORCE_ANSWER = os.environ.get("BCRL_FORCE_ANSWER", "1") == "1"  # final call without tools when the turns run out
TURN_TOKENS = int(os.environ.get("BCRL_TURN_TOKENS", 4096))  # max generated tokens per call
CONTEXT_BUDGET = int(os.environ.get("BCRL_CONTEXT_BUDGET", 32768))  # react: prompt + generation
COMPACT_AT = int(os.environ.get("BCRL_COMPACT_AT", 28672))  # summary: compact when prompt would exceed this
# rLLM logs reward/<name>/*, batch/<name>/*, ...; one shared name keeps wandb keys identical across harnesses.
TRAJ_NAME = os.environ.get("BCRL_TRAJ_NAME", "agent")
# Training uses a longer timeout; scripts/evaluate.sh lowers it to retry dropped requests sooner.
LLM_TIMEOUT = float(os.environ.get("BCRL_LLM_TIMEOUT", 3600))
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": TURN_TOKENS,
}  # the rLLM gateway overrides these during training

SUMMARY_PROMPT = """Your context window is full and will be reset. Write a summary that lets you continue the research from scratch with only the original question and this summary. Include:
1. Key findings so far, with the docids/urls that support them.
2. Candidate answers and how well each is verified.
3. Clues already resolved and approaches/queries that did not work.
4. Concrete next steps.
Do not call any tools. Respond only with the summary."""

RESUME_TEMPLATE = """{question}

---
Your earlier context was compacted. Summary of your progress so far:

{summary}
{grep_note}{budget}
{tail}"""

GREP_NOTE = "\n(The full raw history of the earlier context is searchable with grep_history.)\n"

FINAL_LEAD = {
    "max_turns": "You have used all your turns.",
    "truncated": "Your last reply was cut off at the length limit.",
}
FINAL_PROMPT = (
    "{lead} Do not call any more tools. Based on what you have found so far, give your final answer now as a short "
    "phrase inside \\boxed{{}}; give your best guess if you are unsure."
)
FINAL_PROMPT_MULTI = (
    "{lead} Do not call any more tools. Based on what you have found so far, give your final answers now, one "
    "line per question, in order (write your best guess if unsure):\n{format}"
)

# Turn-capped harnesses (summary, summary_grep) see how many turns remain; react ends at its context limit instead.
SHOW_BUDGET = os.environ.get("BCRL_SHOW_BUDGET", "1") == "1"
BUDGET_GUIDE = "\n- You have at most {max_turns} turns (assistant replies). The number of turns left is shown after each tool result; give your final answer before it runs out."
GREP_FREE_GUIDE = f" Replies that only call grep_history do not count as turns (up to {GREP_FREE_TURNS} such replies)."
MEMORY_FREE_GUIDE = (
    f" Replies that only call memory tools do not count as turns (up to {GREP_FREE_TURNS} such replies)."
)
MASK_GUIDE = (
    "\n- Your context window is limited: when it fills up, the outputs of your oldest tool calls are replaced by a short "
    "placeholder (your own messages and tool calls are kept)."
)
MEMORY_GUIDE = (
    "\n- Your context window is limited: when it fills up, the oldest messages are removed from it and replaced by a "
    "summary (you are warned shortly before). Removed messages remain searchable with conversation_search; keep what "
    "you need in your working context or archival memory."
)
MODES = ("react", "summary", "summary_grep", "mask", "memgpt")
FIT_MARGIN = 1024  # mask/memgpt: fit target below COMPACT_AT, so the re-estimated context (+500) is under it


def _turns_left(used: int, max_turns: int = MAX_TURNS) -> str:
    left = max_turns - used
    if left == 1:
        return f"[1 of {max_turns} turns left: this is your last turn. You must give your final answer in \\boxed{{}} now; do not call any more tools.]"
    return f"[{left} of {max_turns} turns left]"


class InvalidCall(Exception):
    """A tool call with missing or malformed arguments."""


@dataclass
class ToolEnv:
    """The task's tools and prompts for run_agent (default: deep research over a corpus, ``research_env``).
    ``tools`` maps a tool name to (async fn(args) -> output, stats key); fn raises InvalidCall for bad arguments."""

    system_prompt: str  # with an {extra_tools} slot for the grep_history line
    tool_specs: list[dict]
    tools: dict[str, tuple[Callable[[dict], Awaitable[str]], str]] = field(default_factory=dict)
    summary_prompt: str = SUMMARY_PROMPT
    resume_tail: str = "Continue the research."
    max_turns: int = 0  # 0: MAX_TURNS (or MULTI_TURNS_PER_Q per question for kind=multi rows)
    n_questions: int = 0  # > 0: the final reply holds numbered \boxed{} answers (see common.extract_answers)


def research_env(corpus: str) -> ToolEnv:
    async def _search(args: dict) -> str:
        if not args.get("query"):
            raise InvalidCall
        return await search(corpus, str(args["query"]))

    async def _doc(args: dict) -> str:
        if args.get("docid") is None:
            raise InvalidCall
        return await get_document(corpus, str(args["docid"]))

    return ToolEnv(
        SYSTEM_PROMPT, TOOL_SPECS, {"search": (_search, "search_calls"), "get_document": (_doc, "doc_calls")}
    )


def _system(
    grep_on: bool,
    show_budget: bool,
    max_turns: int = MAX_TURNS,
    base: str = SYSTEM_PROMPT,
    mode: str = "",
    memory: mem.MemGPTMemory | None = None,
) -> dict:
    content = base.format(extra_tools=GREP_TOOL_DOC if grep_on else (mem.MEMORY_TOOL_DOC if memory is not None else ""))
    content += MASK_GUIDE if mode == "mask" else (MEMORY_GUIDE if memory is not None else "")
    if show_budget:
        content += BUDGET_GUIDE.format(max_turns=max_turns)
        if GREP_FREE_TURNS > 0:
            content += GREP_FREE_GUIDE if grep_on else (MEMORY_FREE_GUIDE if memory is not None else "")
    if memory is not None:
        content += memory.block()
    return {"role": "system", "content": content}


def _fit(messages: list[dict], budget: float, scale: float) -> tuple[list[dict], int]:
    """fit_compaction_prompt calibrated with a tokens-per-character scale."""
    return fit_compaction_prompt(messages, int(budget), int(scale * sum(_chars(m) for m in messages)), len(messages))


def _est_tokens(text: str) -> int:
    return len(text) // 3 + 1


def _chars(m: dict) -> int:
    n = len(m.get("content") or "") + len(m.get("reasoning_content") or "")
    for tc in m.get("tool_calls") or []:
        n += len(tc["function"]["name"]) + len(tc["function"]["arguments"]) + 30
    return n + 16


def fit_compaction_prompt(
    messages: list[dict], budget: int, scored_tokens: int, n_scored: int
) -> tuple[list[dict], int]:
    """Codex-style fit of the compaction prompt into ``budget`` tokens.

    Drops the oldest turns (keeping the system/first user message and the latest turn), then
    shortens the latest tool outputs. Tokens are estimated from characters, calibrated on
    ``scored_tokens`` = the actual token count of ``messages[:n_scored]``.
    """
    scale = scored_tokens / max(1, sum(_chars(m) for m in messages[:n_scored]))
    est = lambda ms: scale * sum(_chars(m) for m in ms)  # noqa: E731
    msgs, trimmed = list(messages), 0
    while est(msgs) > budget:
        starts = [i for i in range(2, len(msgs)) if msgs[i]["role"] == "assistant"]
        if len(starts) < 2:
            break
        del msgs[2 : starts[1]]
        trimmed += 1
    over = est(msgs) - budget
    for i in range(len(msgs) - 1, 1, -1):
        if over <= 0:
            break
        m = msgs[i]
        if m["role"] != "tool":
            continue
        keep = max(0, len(m["content"]) - int(over / scale) - 200)
        over -= (len(m["content"]) - keep) * scale
        msgs[i] = {**m, "content": m["content"][:keep] + "\n...(truncated to fit the context window)"}
    return msgs, trimmed


def _final_answer(content: str, n_questions: int) -> tuple[str | None, list | None]:
    """(answer, per-question answers) from a final reply; multi-question answers are joined with " | "."""
    if not n_questions:
        return extract_boxed(content), None
    answers = extract_answers(content, n_questions)
    return (" | ".join(a or "-" for a in answers) if any(answers) else None), answers


async def run_agent(task: Task, config: AgentConfig, mode: str, env: ToolEnv | None = None) -> Episode:
    assert mode in MODES, mode
    meta = task.metadata or {}
    if meta.get("kind") == "cb":
        from .contextbench import run_cb_episode  # imports this module

        return await run_cb_episode(task, config, mode)
    question = str(meta.get("question") or task.instruction)
    env = env or research_env(meta.get("corpus", "openresearcher"))
    n_questions = env.n_questions or (len(row_extra(meta).get("questions") or []) if meta.get("kind") == "multi" else 0)
    max_turns = env.max_turns or (MULTI_TURNS_PER_Q * n_questions if n_questions else MAX_TURNS)
    use_grep = mode == "summary_grep"
    show_budget = SHOW_BUDGET and mode != "react"
    # grep_history is offered only once a compaction has dropped something from the context.
    grep_on = False
    memory = mem.MemGPTMemory() if mode == "memgpt" else None
    tools = env.tool_specs + (mem.MEMORY_TOOL_SPECS if memory is not None else [])
    system = _system(grep_on, show_budget, max_turns, env.system_prompt, mode, memory)
    warned = False  # memgpt: memory-pressure warning given since the last flush
    # mask/memgpt make room at most once per call: if the context is still over afterwards (one turn's outputs alone
    # exceed the window), the next call goes ahead with what fits rather than looping without an LLM call.
    room_made = False

    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY", timeout=LLM_TIMEOUT, max_retries=1)
    messages: list[dict] = [system, {"role": "user", "content": question}]
    history: list[str] = [f"[question]\n{question}"]
    steps: list[Step] = []
    answer, answers, termination = None, None, "max_turns"
    ctx_est = _est_tokens(system["content"] + question) + 500
    stats = {
        "turns": 0,
        "tool_calls": 0,
        "compactions": 0,
        "max_context": 0,
        "tool_errors": 0,
        "trimmed_turns": 0,
        "llm_calls": 0,
        "search_calls": 0,
        "doc_calls": 0,
        "grep_calls": 0,
        "grep_empty": 0,
        "grep_chars": 0,
        "grep_turns": 0,
        "forced_answer": 0,
    }
    stats.update(
        {key: 0 for key in (mem.MASK_STATS if mode == "mask" else mem.MEMGPT_STATS if mode == "memgpt" else ())}
    )
    stats.update({key: 0 for _, key in env.tools.values()})
    stats.update(dict.fromkeys(mem.COST_STATS, 0))
    t0 = time.monotonic()
    scored_tokens, n_scored = 0, 0  # actual token count of messages[:n_scored] (from the last call's usage)

    async def call(msgs: list[dict], use_tools: bool = True):
        kwargs = {"tools": tools} if use_tools else {}
        resp = await client.chat.completions.create(model=config.model, messages=msgs, **kwargs, **SAMPLING)
        stats["llm_calls"] += 1
        usage = resp.usage
        mem.add_usage(stats, usage)
        total = (usage.prompt_tokens + usage.completion_tokens) if usage else ctx_est
        stats["max_context"] = max(stats["max_context"], total)
        return resp.choices[0], total

    while stats["turns"] < max_turns:
        if mode == "react" and ctx_est + TURN_TOKENS > CONTEXT_BUDGET:
            termination = "context_limit"
            break
        if mode == "mask" and ctx_est > COMPACT_AT and not room_made:
            # Mask the oldest tool outputs the model has already seen (not the latest turn's), then old reasoning, then
            # whole turns; only if the prompt still does not fit, shorten the newest tool outputs (not seen yet).
            scale = mem.token_scale(messages, scored_tokens, n_scored)
            protect = max(i for i, m in enumerate(messages) if m["role"] != "tool")
            messages, ctx_est = mem.make_room_by_masking(
                messages,
                ctx_est,
                COMPACT_AT,
                TURN_TOKENS,
                scale,
                protect,
                ("tool",),
                mem.OBS_PLACEHOLDER,
                "assistant",
                stats,
            )
            if ctx_est > COMPACT_AT:  # shorten the newest outputs; FIT_MARGIN covers the +500 of the estimate
                messages, _ = _fit(messages, COMPACT_AT - FIT_MARGIN, scale)
                ctx_est = scale * mem.ctx_chars(messages) + 500
            scored_tokens, n_scored = int(scale * mem.ctx_chars(messages)), len(messages)
            room_made = True
            continue
        if mode == "memgpt" and ctx_est > COMPACT_AT and not room_made:
            # Flush the queue: evict the oldest messages (whole turns) and write a recursive summary of them.
            scale = mem.token_scale(messages, scored_tokens, n_scored)
            protect = max((i for i, m in enumerate(messages) if m["role"] == "assistant"), default=len(messages))
            cut = mem.flush_cut(messages, scale, mem.KEEP * COMPACT_AT, "assistant", protect)
            summary = None
            if cut is not None:
                budget = COMPACT_AT - _est_tokens(mem.SUMMARY_PROMPT) - 64
                fitted, trimmed = _fit(messages[:cut], budget, scale)
                stats["trimmed_turns"] += trimmed
                msgs = fitted + [{"role": "user", "content": mem.SUMMARY_PROMPT}]
                try:
                    choice, _ = await call(msgs, use_tools=False)
                except Exception as e:  # noqa: BLE001
                    logger.warning("task %s memory flush failed: %s", task.id, e)
                    termination = "llm_error"
                    break
                cmsg = normalize_message(choice.message)
                summary = mem.reply_text(cmsg)
                steps.append(
                    Step(chat_completions=msgs + [msg_to_dict(cmsg)], model_response=summary, action="summary")
                )
                stats["compactions"] += 1
                history.append(f"[memory summary {stats['compactions']}]\n{summary}")
            system = _system(grep_on, show_budget, max_turns, env.system_prompt, mode, memory)
            head = (
                mem.HEAD_TEMPLATE.format(task=question, n=cut - 2, summary=summary)
                if cut is not None
                else messages[1]["content"]
            )
            messages = [system, {"role": "user", "content": head}] + messages[cut if cut is not None else 2 :]
            ctx_est = scale * mem.ctx_chars(messages) + 500
            if ctx_est > COMPACT_AT:  # the kept turn alone is too long: shorten its tool outputs to fit
                messages, trimmed = _fit(messages, COMPACT_AT - FIT_MARGIN, scale)
                stats["trimmed_turns"] += trimmed
                ctx_est = scale * mem.ctx_chars(messages) + 500
            scored_tokens, n_scored = int(scale * mem.ctx_chars(messages)), len(messages)
            warned = False
            room_made = True
            continue
        if mode in ("summary", "summary_grep") and ctx_est > COMPACT_AT:
            # Keep the compaction call within the 32K context: prompt <= COMPACT_AT, generation <= TURN_TOKENS.
            budget = COMPACT_AT - _est_tokens(env.summary_prompt) - 64
            fitted, trimmed = (
                fit_compaction_prompt(messages, budget, scored_tokens, n_scored) if n_scored else (messages, 0)
            )
            stats["trimmed_turns"] += trimmed
            msgs = fitted + [{"role": "user", "content": env.summary_prompt}]
            try:
                choice, _ = await call(msgs)
            except Exception as e:  # noqa: BLE001
                logger.warning("task %s compaction failed: %s", task.id, e)
                termination = "llm_error"
                break
            cmsg = normalize_message(choice.message)
            summary = mem.reply_text(cmsg)
            steps.append(Step(chat_completions=msgs + [msg_to_dict(cmsg)], model_response=summary, action="summary"))
            stats["compactions"] += 1
            history.append(f"[compaction {stats['compactions']} summary]\n{summary}")
            if use_grep and not grep_on:
                grep_on, tools = True, env.tool_specs + [GREP_TOOL_SPEC]
                system = _system(grep_on, show_budget, max_turns, env.system_prompt, mode)
            resume = RESUME_TEMPLATE.format(
                question=question,
                summary=summary,
                grep_note=GREP_NOTE if grep_on else "",
                budget=f"\n{_turns_left(stats['turns'], max_turns)}\n" if show_budget else "",
                tail=env.resume_tail,
            )
            messages = [system, {"role": "user", "content": resume}]
            ctx_est = _est_tokens(system["content"] + resume) + 500
            continue

        try:
            choice, ctx_tokens = await call(messages)
            room_made = False
        except Exception as e:  # noqa: BLE001
            logger.warning("task %s turn %d: LLM call failed: %s", task.id, stats["turns"], e)
            termination = "llm_error"
            break
        msg = normalize_message(choice.message)
        free_tools = {"grep_history"} if grep_on else (mem.MEMORY_TOOLS if memory is not None else ())
        free_key = "mem_turns" if memory is not None else "grep_turns"
        free = bool(free_tools and msg.tool_calls) and all(tc.function.name in free_tools for tc in msg.tool_calls)
        if free and stats[free_key] < GREP_FREE_TURNS:
            stats[free_key] += 1
        else:
            stats["turns"] += 1
        turn = (
            stats["turns"] + stats["grep_turns"] + stats.get("mem_turns", 0)
        )  # history label: every reply, counted or not
        messages.append(msg_to_dict(msg))
        scored_tokens, n_scored = ctx_tokens, len(messages)
        content = msg.content or ""
        steps.append(Step(chat_completions=list(messages), model_response=content, action=content))
        history.append(f"[turn {turn} assistant]\n{content}")

        if not msg.tool_calls:
            answer, answers = _final_answer(content, n_questions)
            termination = "answered" if answer else ("truncated" if choice.finish_reason == "length" else "no_answer")
            break

        new_tokens = 0
        for tc in msg.tool_calls:
            args = parse_args(tc.function.arguments)
            name = tc.function.name
            try:
                if name == "grep_history" and grep_on and args.get("pattern"):
                    out = grep_call(history, args)
                    stats["grep_calls"] += 1
                    stats["grep_empty"] += out.startswith("No matches")
                    stats["grep_chars"] += len(out)
                elif memory is not None and name in mem.MEMORY_TOOLS:
                    res = memory.call(name, args, history)
                    if res is None:
                        raise InvalidCall
                    out, key = res
                    stats[key] += 1
                    stats["conv_search_empty"] += key == "conv_search_calls" and out.startswith("No messages")
                elif name in env.tools:
                    fn, key = env.tools[name]
                    out = await fn(args)
                    stats[key] += 1
                else:
                    raise InvalidCall
            except InvalidCall:
                out = f"Error: invalid tool call {name}({tc.function.arguments})."
                stats["tool_errors"] += 1
            except Exception as e:  # noqa: BLE001
                out = f"Error: tool {name} failed: {type(e).__name__}: {e}"
                stats["tool_errors"] += 1
            out = truncate(out)
            stats["tool_calls"] += 1
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            history.append(f"[turn {turn} tool {name}({tc.function.arguments})]\n{out}")
            new_tokens += _est_tokens(out) + 20
        if show_budget and messages[-1]["role"] == "tool":  # shown to the model only, not added to the grep history
            messages[-1]["content"] += f"\n\n{_turns_left(stats['turns'], max_turns)}"
            new_tokens += 12
        ctx_est = ctx_tokens + new_tokens
        if (
            memory is not None
            and not warned
            and mem.WARN * COMPACT_AT < ctx_est <= COMPACT_AT
            and messages[-1]["role"] == "tool"
        ):
            messages[-1]["content"] += "\n\n" + mem.WARNING.format(
                pct=round(100 * ctx_est / COMPACT_AT)
            )  # not in the history
            ctx_est += 60
            warned = True
            stats["mem_warnings"] += 1

    if termination in ("max_turns", "truncated") and FORCE_ANSWER:
        # Out of turns (or the final reply was cut off): one more call without tools, fitted into the window like a
        # compaction call.
        lead = FINAL_LEAD[termination]
        prompt = (
            FINAL_PROMPT_MULTI.format(
                lead=lead,
                format="\n".join(f"{i + 1}. \\boxed{{answer to question {i + 1}}}" for i in range(n_questions)),
            )
            if n_questions
            else FINAL_PROMPT.format(lead=lead)
        )
        limit = (CONTEXT_BUDGET - TURN_TOKENS) if mode == "react" else COMPACT_AT
        fitted = messages
        if ctx_est > limit and n_scored:
            fitted, _ = fit_compaction_prompt(messages, limit - _est_tokens(prompt) - 64, scored_tokens, n_scored)
        msgs = fitted + [{"role": "user", "content": prompt}]
        try:
            choice, _ = await call(msgs, use_tools=False)
            fmsg = normalize_message(choice.message)
            content = fmsg.content or ""
            steps.append(Step(chat_completions=msgs + [msg_to_dict(fmsg)], model_response=content, action=content))
            history.append(f"[final answer]\n{content}")
            stats["forced_answer"] = 1
            answer, answers = _final_answer(content, n_questions)
            if answer:
                termination = "forced_answer"
        except Exception as e:  # noqa: BLE001
            logger.warning("task %s forced answer failed: %s", task.id, e)
            termination = "llm_error"

    stats["wall_s"] = round(time.monotonic() - t0, 2)  # the agent's own time (LLM and tool calls), not the judge
    return Episode(
        trajectories=[Trajectory(name=TRAJ_NAME, steps=steps)],
        artifacts={
            "answer": answer,
            "termination": termination,
            **({"answers": answers} if n_questions else {}),
            **stats,
        },
        metrics={k: float(v) for k, v in stats.items()},
    )


@rllm.rollout(name="react")
async def react_flow(task: Task, config: AgentConfig) -> Episode:
    return await run_agent(task, config, "react")


@rllm.rollout(name="summary")
async def summary_flow(task: Task, config: AgentConfig) -> Episode:
    return await run_agent(task, config, "summary")


@rllm.rollout(name="summary_grep")
async def summary_grep_flow(task: Task, config: AgentConfig) -> Episode:
    return await run_agent(task, config, "summary_grep")


@rllm.rollout(name="mask")
async def mask_flow(task: Task, config: AgentConfig) -> Episode:
    return await run_agent(task, config, "mask")


@rllm.rollout(name="memgpt")
async def memgpt_flow(task: Task, config: AgentConfig) -> Episode:
    return await run_agent(task, config, "memgpt")


FLOWS = {
    "react": react_flow,
    "summary": summary_flow,
    "summary_grep": summary_grep_flow,
    "mask": mask_flow,
    "memgpt": memgpt_flow,
}
