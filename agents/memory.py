"""Two context-management baselines shared by agents/flows.py (research, BCP-Multi, AMA file) and
agents/contextbench.py (ContextBench, AMA stream).

mask    Observation masking (Lindenbauer et al. 2025, "The Complexity Trap", arXiv 2508.21433): old observations
        (tool outputs; in the streaming tasks, the streamed user messages) are replaced by a short placeholder, while
        the agent's own messages and tool calls are kept. The paper keeps the last M observations (M=10, tuned per
        scaffold); with a fixed 32K window we mask by budget instead: when the prompt would exceed COMPACT_AT, the
        oldest unmasked observations are masked until it is under MASK_TARGET * COMPACT_AT. If that is not enough
        (Qwen3.5 keeps the reasoning of every turn after the last user message, i.e. all of a research episode's),
        the reasoning of the oldest turns is stripped too, keeping their actions (the chat template itself drops
        earlier turns' reasoning in multi-turn chats); only if the prompt still does not fit (COMPACT_AT - TURN_TOKENS)
        are the oldest whole turns dropped. No extra LLM call. Masking in batches rather than one observation per
        turn keeps the prompt prefix stable between masking events, as compaction does.

memgpt  MemGPT-style virtual context (Packer et al. 2023, arXiv 2310.08560), single agent:
        - main context = system prompt with an editable working-context block (working_context_append / _replace,
          WC_CHARS characters) + a FIFO message queue headed by the task and a recursive summary;
        - a memory-pressure warning when the prompt passes WARN * COMPACT_AT ("save what you need now");
        - at COMPACT_AT the queue is flushed: the oldest messages (about half the window, KEEP) are evicted and the
          model writes a new recursive summary of the old summary + the evicted messages (a call without tools, as
          MemGPT's summarizer; offered tools, the model often "saves" with a memory tool instead of summarizing);
        - recall storage: conversation_search(query, page), case-insensitive substring search over every past message
          (MemGPT's text search), PAGE results per page;
        - archival storage: archival_memory_insert(content) / archival_memory_search(query, page), ranked by BM25
          (in place of MemGPT's embedding search); long passages are shown around the first query term.
        Replies that only call memory tools do not count toward the turn cap (same allowance as grep_history). The
        working-context block in the system prompt is re-rendered when the queue is flushed; between flushes, edits
        are visible in the results of the working_context_* calls (same information, stable prompt prefix).
"""

from __future__ import annotations

import copy
import math
import os
import re
from collections import Counter

from .common import GREP_CHARS, _as_int

MASK_TARGET = float(os.environ.get("BCRL_MASK_TARGET", 0.5))  # mask down to this fraction of COMPACT_AT
WARN = float(os.environ.get("BCRL_MEMGPT_WARN", 0.7))  # memory-pressure warning at this fraction of COMPACT_AT
KEEP = float(os.environ.get("BCRL_MEMGPT_KEEP", 0.5))  # a flush keeps the newest messages up to this fraction
WC_CHARS = int(os.environ.get("BCRL_MEMGPT_WC_CHARS", 4000))  # working-context size limit
PAGE = 5  # results per conversation_search / archival_memory_search page
SNIPPET_CHARS = 1000  # conversation_search shows this much of each matching message, around the first match

MASK_STATS = ("mask_events", "masked", "reasoning_stripped")
MEMGPT_STATS = (
    "mem_warnings",
    "mem_turns",
    "wc_edits",
    "wc_errors",
    "archival_inserts",
    "archival_search_calls",
    "conv_search_calls",
    "conv_search_empty",
)

OBS_PLACEHOLDER = "[Old tool output removed from the context ({n} characters).]"
MSG_PLACEHOLDER = "[Old message removed from the context ({n} characters).]"
_MASKED = re.compile(r"^\[Old (tool output|message) removed from the context \(\d+ characters\)\.\]$")


# ---------------------------------------------------------------- token estimates


# Per-episode cost: tokens summed over every LLM call (summaries, flushes and forced answers included); agent wall time.
COST_STATS = ("prompt_tokens", "completion_tokens", "cached_tokens", "wall_s")


def add_usage(stats: dict, usage) -> None:
    """Add one call's token usage to the episode stats (cached_tokens needs vLLM --enable-prompt-tokens-details)."""
    if usage is None:
        return
    stats["prompt_tokens"] += usage.prompt_tokens or 0
    stats["completion_tokens"] += usage.completion_tokens or 0
    details = getattr(usage, "prompt_tokens_details", None)
    stats["cached_tokens"] += (getattr(details, "cached_tokens", None) or 0) if details is not None else 0


def ctx_chars(messages: list[dict]) -> int:
    """Characters of the rendered prompt. Qwen3.5's chat template keeps the reasoning only of assistant messages after
    the last user message."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    n = 0
    for i, m in enumerate(messages):
        n += len(m.get("content") or "") + 16
        if i > last_user:
            n += len(m.get("reasoning_content") or "")
        for tc in m.get("tool_calls") or []:
            n += len(tc["function"]["name"]) + len(tc["function"]["arguments"]) + 30
    return n


def token_scale(messages: list[dict], scored_tokens: int, n_scored: int) -> float:
    """Tokens per character, calibrated on the actual token count of messages[:n_scored] (else 1/3)."""
    if not n_scored or not scored_tokens:
        return 1 / 3
    return scored_tokens / max(1, ctx_chars(messages[:n_scored]))


# ---------------------------------------------------------------- observation masking


def mask_observations(
    messages: list[dict],
    est: float,
    target: float,
    scale: float,
    protect: int,
    roles: tuple[str, ...],
    placeholder: str,
) -> tuple[float, int]:
    """Mask, oldest first, the messages[2:protect] with a role in ``roles`` until ``est`` (the prompt's estimated
    tokens) is at most ``target``. Edits ``messages`` in place; returns (new estimate, number masked)."""
    n = 0
    for i in range(2, min(protect, len(messages))):
        if est <= target:
            break
        m = messages[i]
        content = m.get("content") or ""
        if m["role"] not in roles or _MASKED.match(content):
            continue
        new = placeholder.format(n=len(content))
        if len(new) >= len(content):
            continue
        messages[i] = {**m, "content": new}
        est -= (len(content) - len(new)) * scale
        n += 1
    return est, n


def strip_reasoning(messages: list[dict], est: float, target: float, scale: float, protect: int) -> tuple[float, int]:
    """Drop, oldest first, the reasoning of assistant messages[2:protect] that the chat template renders (those after the
    last user message) until ``est`` is at most ``target``; their content and tool calls stay. In place; returns (new
    estimate, number stripped)."""
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
    n = 0
    for i in range(max(2, last_user + 1), min(protect, len(messages))):
        if est <= target:
            break
        m = messages[i]
        if m["role"] == "assistant" and m.get("reasoning_content"):
            messages[i] = {k: v for k, v in m.items() if k != "reasoning_content"}
            est -= len(m["reasoning_content"]) * scale
            n += 1
    return est, n


def make_room_by_masking(
    messages: list[dict],
    est: float,
    compact_at: int,
    slack: int,
    scale: float,
    protect: int,
    roles: tuple[str, ...],
    placeholder: str,
    boundary: str,
    stats: dict,
) -> tuple[list[dict], float]:
    """The mask harness's step when the prompt would exceed ``compact_at``: (1) mask the oldest seen observations down to
    MASK_TARGET * compact_at; (2) if still over, strip the reasoning of the oldest turns (actions stay); (3) only if the
    prompt is still over ``compact_at - slack`` (at least the target), drop the oldest whole turns down to that. Returns (messages, estimate);
    the caller shortens the newest outputs if even that is not enough."""
    target = MASK_TARGET * compact_at
    est, n = mask_observations(messages, est, target, scale, protect, roles, placeholder)
    est, k = strip_reasoning(messages, est, target, scale, protect)
    messages, est, dropped = drop_oldest_turns(messages, est, max(target, compact_at - slack), scale, boundary)
    stats["mask_events"] += 1
    stats["masked"] += n
    stats["reasoning_stripped"] += k
    stats["trimmed_turns"] += dropped
    return messages, est


def drop_oldest_turns(
    messages: list[dict], est: float, target: float, scale: float, boundary: str
) -> tuple[list[dict], float, int]:
    """Drop whole turns after messages[1] (a turn starts at a message with role ``boundary``), oldest first and never
    the latest, until ``est`` is at most ``target``. Returns (messages, new estimate, turns dropped)."""
    offset = est - scale * ctx_chars(messages)  # calibration margin of the caller's estimate
    msgs, n = list(messages), 0
    while scale * ctx_chars(msgs) + offset > target:
        starts = [i for i in range(2, len(msgs)) if msgs[i]["role"] == boundary]
        if len(starts) < 2:
            break
        del msgs[2 : starts[1]]
        n += 1
    return msgs, scale * ctx_chars(msgs) + offset, n


def trim_turn(
    messages: list[dict], est: float, target: float, hard: float, scale: float, start: int
) -> tuple[list[dict], float, int]:
    """Make room inside the turn being handled, i.e. messages[start:], the replies and tool outputs that follow the
    latest user message (which stays), for when evicting what came before it is not enough. Strip the reasoning of all
    but the newest reply, drop the oldest (reply, tool outputs) groups down to ``target`` (never the newest), and shorten
    the newest tool outputs down to ``hard``. Returns (messages, new estimate, groups dropped)."""
    msgs = list(messages)
    replies = [i for i in range(start, len(msgs)) if msgs[i]["role"] == "assistant"]
    est, _ = strip_reasoning(msgs, est, target, scale, replies[-1] if replies else len(msgs))
    offset = est - scale * ctx_chars(msgs)
    n = 0
    while scale * ctx_chars(msgs) + offset > target:
        replies = [i for i in range(start, len(msgs)) if msgs[i]["role"] == "assistant"]
        if len(replies) < 2:
            break
        del msgs[replies[0] : replies[1]]
        n += 1
    over = scale * ctx_chars(msgs) + offset - hard
    for i in range(len(msgs) - 1, start - 1, -1):
        if over <= 0:
            break
        m = msgs[i]
        if m["role"] != "tool":
            continue
        keep = max(0, len(m["content"]) - int(over / scale) - 200)
        over -= (len(m["content"]) - keep) * scale
        msgs[i] = {**m, "content": m["content"][:keep] + "\n...(truncated to fit the context window)"}
    return msgs, scale * ctx_chars(msgs) + offset, n


# ---------------------------------------------------------------- MemGPT


def flush_cut(messages: list[dict], scale: float, keep_tokens: float, boundary: str, protect: int) -> int | None:
    """Where a MemGPT flush cuts the queue: messages[2:cut] are evicted, messages[cut:] kept (cut is a message with
    role ``boundary``, or len(messages)). The earliest cut <= protect that keeps at most ``keep_tokens``; if none
    does, the latest one. None if nothing can be evicted."""
    cands = [
        i for i in range(3, min(protect, len(messages)) + 1) if i == len(messages) or messages[i]["role"] == boundary
    ]
    for i in cands:
        if ctx_chars(messages[i:]) * scale <= keep_tokens:
            return i
    return cands[-1] if cands else None


MEMORY_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "working_context_append",
            "description": f"Append text to your working context, a notepad of up to {WC_CHARS} characters that is shown in the system prompt and never removed from your context.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "Text to append."}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "working_context_replace",
            "description": "Replace text in your working context (exact match); an empty new_content deletes it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_content": {"type": "string", "description": "Exact text currently in the working context."},
                    "new_content": {"type": "string", "description": "Replacement text (may be empty)."},
                },
                "required": ["old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_insert",
            "description": "Store a passage in archival memory (unlimited, outside your context; retrieve it with archival_memory_search).",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string", "description": "Passage to store."}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": f"Keyword search (BM25) over archival memory; returns the best-matching passages, {PAGE} per page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "page": {"type": "integer", "description": "0-based page of results (default 0)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conversation_search",
            "description": (
                "Case-insensitive text search over every earlier message and tool output of this conversation, "
                f"including those removed from your context; returns matching messages in order, {PAGE} per page."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Exact text to look for (case-insensitive)."},
                    "page": {"type": "integer", "description": "0-based page of results (default 0)."},
                },
                "required": ["query"],
            },
        },
    },
]
MEMORY_TOOLS = frozenset(s["function"]["name"] for s in MEMORY_TOOL_SPECS)

MEMORY_TOOL_DOC = (
    f"- working_context_append(content) / working_context_replace(old_content, new_content): edit your working context, a "
    f"notepad of up to {WC_CHARS} characters shown at the end of this system prompt; it is never removed from your context.\n"
    "- archival_memory_insert(content) / archival_memory_search(query, page): store passages in your unlimited archival memory "
    "and search them (keyword search).\n"
    "- conversation_search(query, page): case-insensitive text search over every earlier message and tool output, including "
    "those removed from your context.\n"
)
WARNING = (
    "[System message] Memory pressure: your context is about {pct}% full. When it is full, the oldest messages will be "
    "removed from it and replaced by a summary. Save anything you will need later to your working context or archival "
    "memory now."
)
SUMMARY_PROMPT = """Your context window is full. The messages above, after the task, will be removed from your context (they remain searchable with conversation_search); the most recent messages are kept. Write a summary of the messages that will be removed, so that you can continue the task with your working context, this summary and the kept messages. If the messages above begin with an earlier summary, merge it into the new one. Keep the key facts, findings and progress, copied exactly where later steps may need them.
Do not call any tools. Respond only with the summary."""
HEAD_TEMPLATE = """{task}

---
[Summary of the {n} oldest messages of this conversation, which were removed from your context (conversation_search finds them):]
{summary}"""


# A compaction reply without text (e.g. a tool call) falls back to its reasoning. Through the rLLM gateway (training) the
# reasoning arrives as raw text that normalize_message parses into ``reasoning_content``; a plain vLLM server (eval)
# returns it as ``reasoning``, which the harnesses ignored before Oct 24, so such compactions left an empty summary in
# eval only. BCRL_REASONING_FALLBACK=0 reproduces that old eval behavior.
REASONING_FALLBACK = os.environ.get("BCRL_REASONING_FALLBACK", "1") == "1"


def reply_text(msg) -> str:
    """A summarizer reply's text, or the end of its reasoning if it has none (vLLM: ``reasoning``; parsed: ``reasoning_content``)."""
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or (getattr(msg, "reasoning", None) if REASONING_FALLBACK else None)
        or ""
    )
    return (msg.content or "").strip() or reasoning[-8000:]


def _terms(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of the docs sharing a term with the query, best first."""
    toks = [_terms(d) for d in docs]
    if not toks:
        return []
    avgdl = sum(map(len, toks)) / len(toks) or 1.0
    df = Counter(t for ts in toks for t in set(ts))
    q = set(_terms(query))
    scores = []
    for ts in toks:
        tf, s = Counter(ts), 0.0
        for t in q:
            if tf[t]:
                idf = math.log(1 + (len(toks) - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(ts) / avgdl))
        scores.append(s)
    return sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: (-scores[i], i))


def _page(items: list[str], page: int, what: str) -> str:
    pages = max(1, math.ceil(len(items) / PAGE))
    page = min(max(page, 0), pages - 1)
    shown = items[page * PAGE : (page + 1) * PAGE]
    out = f"{len(items)} {what}; showing {page * PAGE + 1}-{page * PAGE + len(shown)} (page {page} of pages 0-{pages - 1}).\n\n"
    out += "\n\n".join(shown)
    return out if len(out) <= GREP_CHARS else out[:GREP_CHARS] + "\n...(cut off)"


def _window(text: str, i: int) -> str:
    """SNIPPET_CHARS of ``text`` around position i (-1: from the start)."""
    start = max(0, i - SNIPPET_CHARS // 3) if i >= 0 else 0
    return f"{'...' if start else ''}{text[start : start + SNIPPET_CHARS]}{'...' if start + SNIPPET_CHARS < len(text) else ''}"


def _snippet(entry: str, query: str) -> str:
    head, _, body = entry.partition("\n")
    return f"{head}\n{_window(body, body.lower().find(query))}"


def _passage_snippet(passage: str, query: str) -> str:
    """A long passage is shown around the first occurrence of a query term."""
    low = passage.lower()
    found = [i for i in (low.find(t) for t in _terms(query)) if i >= 0]
    return _window(passage, min(found) if found else -1)


def _is_memory_output(entry: str) -> bool:
    head = entry.split("\n", 1)[0]
    return any(f"{name}(" in head for name in MEMORY_TOOLS)


class MemGPTMemory:
    """Working context and archival storage of one episode; recall storage is the harness's raw history."""

    def __init__(self) -> None:
        self.working = ""
        self.archival: list[str] = []

    def copy(self) -> "MemGPTMemory":
        return copy.deepcopy(self)

    def block(self) -> str:
        return (
            f"\n\nWorking context ({len(self.working)}/{WC_CHARS} characters; updated here when old messages are removed "
            f"from your context, and in the results of your working_context_* calls in between):\n"
            f"<working_context>\n{self.working}\n</working_context>"
        )

    def call(self, name: str, args: dict, history: list[str]) -> tuple[str, str] | None:
        """Run a memory tool call: (output, stats key), or None for a missing/malformed argument."""
        if name == "working_context_append":
            if not isinstance(args.get("content"), str) or not args["content"].strip():
                return None
            new = (self.working + "\n" + args["content"]).strip("\n")
            return self._set_working(new)
        if name == "working_context_replace":
            old, new = args.get("old_content"), args.get("new_content", "")
            if not isinstance(old, str) or not old or not isinstance(new, str):
                return None
            if old not in self.working:
                return "Error: old_content was not found in the working context (it must match exactly).", "wc_errors"
            return self._set_working(self.working.replace(old, new, 1))
        if name == "archival_memory_insert":
            if not isinstance(args.get("content"), str) or not args["content"].strip():
                return None
            self.archival.append(args["content"].strip())
            return (
                f"Stored as archival passage {len(self.archival)} ({len(self.archival)} in total).",
                "archival_inserts",
            )
        query = str(args.get("query") or "").strip()
        if name not in ("archival_memory_search", "conversation_search") or not query:
            return None
        page = _as_int(args.get("page", 0), 0)
        if name == "archival_memory_search":
            hits = [
                f"[passage {i + 1}]\n{_passage_snippet(self.archival[i], query)}"
                for i in bm25_rank(query, self.archival)
            ]
            return (
                _page(hits, page, "matching passage(s)") if hits else "No matching passages."
            ), "archival_search_calls"
        q = query.lower()
        hits = [_snippet(h, q) for h in history if q in h.lower() and not _is_memory_output(h)]
        return (
            _page(hits, page, "matching message(s)") if hits else f"No messages contain {query!r}."
        ), "conv_search_calls"

    def _set_working(self, new: str) -> tuple[str, str]:
        if len(new) > WC_CHARS:
            return (
                f"Error: the working context would have {len(new)} characters, over its limit of {WC_CHARS}; shorten it "
                "with working_context_replace first.",
                "wc_errors",
            )
        self.working = new
        return f"Working context updated ({len(new)}/{WC_CHARS} characters). It now reads:\n{new}", "wc_edits"
