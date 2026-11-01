"""Shared pieces: retrieval tools, prompts, answer extraction and the F1 evaluator."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import string
import time
from collections import Counter

import httpx

try:  # re-compatible engine with a per-search timeout (re has none)
    import regex as _regex
except ImportError:  # pragma: no cover
    _regex = None

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.types import Episode, Task

from .judge import get_judge as _judge

logger = logging.getLogger(__name__)

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8100")
MAX_TOOL_CHARS = int(os.environ.get("BCRL_TOOL_CHARS", 40_000))  # safety cap on a single tool output (~10K tokens)
# get_document cap. The retrieval server returns at most 8192 tokens; smaller caps (for smaller context windows)
# are applied here at ~4 characters per token.
DOC_TOKENS = int(os.environ.get("BCRL_DOC_TOKENS", 8192))
# grep_history output caps. Results are paged (max_matches/offset) instead of cut, so one call stays small.
GREP_CHARS = int(os.environ.get("BCRL_GREP_CHARS", 6000))  # cap on one grep_history output
GREP_MAX_MATCHES = int(os.environ.get("BCRL_GREP_MAX_MATCHES", 30))  # upper bound for the max_matches argument
GREP_LINE_CHARS = int(os.environ.get("BCRL_GREP_LINE_CHARS", 300))  # long lines are clipped around the match
GREP_MAX_CONTEXT = 3
# Model-written patterns such as "a.*b.*c" can backtrack for hours on multi-MB lines (AMA-Bench JSON observations),
# blocking the whole event loop; one grep call gets this many seconds of matching, then returns an error.
GREP_TIMEOUT = float(os.environ.get("BCRL_GREP_TIMEOUT", 10))

SYSTEM_PROMPT = (
    """You are a deep-research agent. Answer the user's question by researching a fixed document corpus with the tools provided.

Tools:
- search(query): dense retrieval over the corpus; returns the top 10 documents (docid, url, first ~512 tokens).
- get_document(docid): returns the full text of a document (up to """
    + str(DOC_TOKENS)
    + """ tokens).
{extra_tools}
Guidelines:
- Questions often require combining several clues; break them down, search iteratively, and verify candidates against the evidence.
- Prefer short, specific search queries. Open documents when snippets are insufficient.
- When you are confident, stop calling tools and give your final answer as a short phrase inside \\boxed{{}}, e.g. \\boxed{{Marie Curie}}."""
)

GREP_TOOL_DOC = (
    "- grep_history(pattern, max_matches, offset, context_lines, count): regex search over your full, uncompressed "
    "interaction history (all earlier tool outputs and messages), including parts removed from your context by "
    "summarization. Each match is labeled with the turn and tool call it came from.\n"
)

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the document corpus. Returns the top 10 documents with docid, url and a snippet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Fetch the full text of a document by docid (truncated to " + str(DOC_TOKENS) + " tokens).",
            "parameters": {
                "type": "object",
                "properties": {"docid": {"type": "string", "description": "Document id from search results."}},
                "required": ["docid"],
            },
        },
    },
]

GREP_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "grep_history",
        "description": (
            "Regex (Python re, case-insensitive) search over the full raw interaction history, including content dropped "
            "by summarization (earlier grep_history results are not searched). Returns the number of matching lines and "
            "then the matches, oldest first, grouped under the turn/tool call they came from. Works best with exact "
            "strings you saw before: names, numbers, dates, IDs, keys. Page through many matches with offset; use "
            "count=true to get only the number of matching lines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression."},
                "max_matches": {
                    "type": "integer",
                    "description": f"Matching lines to show (default 10, at most {GREP_MAX_MATCHES}).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip this many matching lines (for the next page; default 0).",
                },
                "context_lines": {
                    "type": "integer",
                    "description": f"Lines of context around each match (default 1, at most {GREP_MAX_CONTEXT}).",
                },
                "count": {"type": "boolean", "description": "Only count matching lines (default false)."},
            },
            "required": ["pattern"],
        },
    },
}


def truncate(s: str, n: int = MAX_TOOL_CHARS) -> str:
    return s if len(s) <= n else s[:n] + f"\n...(truncated {len(s) - n} chars)"


async def search(corpus: str, query: str, k: int = 10) -> str:
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{RETRIEVAL_URL}/search", json={"corpus": corpus, "query": query, "k": k})
        r.raise_for_status()
    results = r.json()["results"]
    if not results:
        return "No results."
    return "\n\n".join(f"[{i + 1}] docid: {d['docid']}\nurl: {d['url']}\n{d['snippet']}" for i, d in enumerate(results))


async def get_document(corpus: str, docid: str) -> str:
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post(f"{RETRIEVAL_URL}/open", json={"corpus": corpus, "docid": str(docid)})
    if r.status_code == 404:
        return f"Error: document {docid!r} not found."
    r.raise_for_status()
    d = r.json()
    text = d["text"]
    if DOC_TOKENS < 8192 and len(text) > DOC_TOKENS * 4:
        text = text[: DOC_TOKENS * 4] + f"\n...(document truncated to ~{DOC_TOKENS} tokens)"
    return f"docid: {d['docid']}\nurl: {d['url']}\n\n{text}"


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _source_kind(header: str) -> str | None:
    """What produced a history entry ("[turn 3 tool search(...)]" -> tool outputs); None for grep_history output."""
    h = header.lower()
    if "grep_history(" in h:
        return None
    if h.startswith("[step"):
        return "trajectory steps"
    if " tool " in h or h.startswith("[tool"):
        return "tool outputs"
    if "summary" in h:
        return "summaries"
    if "assistant" in h:
        return "your replies"
    return "user messages"


def _clip(line: str, m: re.Match | None) -> str:
    if len(line) <= GREP_LINE_CHARS:
        return line
    start = max(0, min(m.start() - GREP_LINE_CHARS // 3, len(line) - GREP_LINE_CHARS)) if m else 0
    return (
        ("..." if start else "")
        + line[start : start + GREP_LINE_CHARS]
        + ("..." if start + GREP_LINE_CHARS < len(line) else "")
    )


def grep_history(history: list[str], pattern: str, max_matches=10, offset=0, context_lines=1, count=False) -> str:
    """grep over history entries ("[header]\\nbody"). Matches are numbered oldest first and paged with
    max_matches/offset; each group is labeled with its entry header. Earlier grep outputs are skipped (they
    would repeat earlier matches)."""
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return f"Error: invalid regex: {e}"
    try:
        timed = _regex.compile(pattern, _regex.I) if _regex is not None else None
    except _regex.error:
        timed = None
    deadline = time.monotonic() + GREP_TIMEOUT

    def search(line: str):
        if timed is None:
            return rx.search(line)
        return timed.search(line, timeout=max(0.01, deadline - time.monotonic()))

    try:
        return _grep(history, pattern, search, max_matches, offset, context_lines, count)
    except TimeoutError:
        return (
            f"Error: /{pattern}/ took over {GREP_TIMEOUT:g}s to search (several '.*' on very long lines); "
            "use a simpler, more specific pattern."
        )


def _grep(history, pattern, search, max_matches, offset, context_lines, count) -> str:
    max_matches = max(1, min(_as_int(max_matches, 10), GREP_MAX_MATCHES))
    offset = max(0, _as_int(offset, 0))
    context_lines = max(0, min(_as_int(context_lines, 1), GREP_MAX_CONTEXT))
    if isinstance(count, str):
        count = count.strip().lower() in ("true", "1", "yes")
    entries, hits = [], []
    for ei, entry in enumerate(history):
        header, _, body = entry.partition("\n")
        kind = _source_kind(header)
        lines = body.splitlines() if kind else []
        entries.append((header[:200], lines, kind))
        hits += [(ei, li) for li, line in enumerate(lines) if search(line)]
    if not hits:
        return f"No matches for /{pattern}/."
    by_kind = Counter(entries[ei][2] for ei, _ in hits)
    n_entries = len({ei for ei, _ in hits})
    head = (
        f"{len(hits)} matching line(s) in {n_entries} history entr{'y' if n_entries == 1 else 'ies'} ("
        + ", ".join(f"{c} in {k}" for k, c in by_kind.most_common())
        + ")."
    )
    if count:
        return head
    if offset >= len(hits):
        return f"{head}\nNo matches at offset={offset}."
    out, size, shown, cur, last_end = [], len(head), 0, None, 0
    for ei, li in hits[offset : offset + max_matches]:
        header, lines, _ = entries[ei]
        part = []
        if ei != cur:
            part.append(f"== {header}")
            cur, last_end = ei, 0
        s, e = max(0, li - context_lines, last_end), min(len(lines), li + context_lines + 1)
        if s >= e:  # already shown as context of the previous match
            shown += 1
            continue
        if last_end and s > last_end:
            part.append("  --")
        for j in range(s, e):
            m = search(lines[j])
            part.append(f"  {j + 1}{':' if m else '-'} {_clip(lines[j], m)}")
        text = "\n".join(part)
        if shown and size + len(text) > GREP_CHARS:
            break
        out.append(text)
        size, shown, last_end = size + len(text) + 1, shown + 1, max(last_end, e)
    end = offset + shown
    foot = f"(showing matches {offset + 1}-{end} of {len(hits)}" + (
        f"; call again with offset={end} for more)" if end < len(hits) else ")"
    )
    return truncate("\n".join([head, *out, foot]), GREP_CHARS + 1000)


def grep_call(history: list[str], args: dict) -> str:
    """Run a grep_history tool call with the model's (possibly sloppy) arguments."""
    return grep_history(
        history,
        str(args["pattern"]),
        args.get("max_matches", 10),
        args.get("offset", 0),
        args.get("context_lines", 1),
        args.get("count", False),
    )


def msg_to_dict(msg) -> dict:
    """OpenAI message -> dict, keeping reasoning so the chat template can replay it."""
    d: dict = {"role": "assistant", "content": msg.content or ""}
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    if reasoning:
        d["reasoning_content"] = reasoning
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


def normalize_message(msg):
    """Parse reasoning/tool calls out of raw text when the server did not.

    In ``cumulative_token_mode`` the rLLM gateway serves turns >= 2 through
    ``/v1/completions`` and returns the raw generation (``...</think>\\n<tool_call>
    <function=...>``) as ``content`` without running vLLM's reasoning/tool parsers.
    """
    from types import SimpleNamespace as NS

    content = msg.content or ""
    if msg.tool_calls or ("</think>" not in content and "<tool_call>" not in content):
        return msg
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    if "</think>" in content:
        think, content = content.split("</think>", 1)
        reasoning = (reasoning + think.replace("<think>", "")).strip()
    calls = []
    for i, block in enumerate(_TOOL_CALL_RE.findall(content)):
        m = _FUNC_RE.search(block)
        if not m:
            continue
        args = {}
        for k, v in _PARAM_RE.findall(m.group(2)):
            try:
                args[k] = json.loads(v)
            except json.JSONDecodeError:
                args[k] = v
        calls.append(NS(id=f"call_{i}", type="function", function=NS(name=m.group(1), arguments=json.dumps(args))))
    content = _TOOL_CALL_RE.sub("", content).strip()
    return NS(role="assistant", content=content, reasoning_content=reasoning or None, tool_calls=calls or None)


def parse_args(raw: str | None) -> dict:
    try:
        args = json.loads(raw or "{}")
        return args if isinstance(args, dict) else {}
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------- answers / reward


def _boxed_spans(text: str) -> list[tuple[int, str]]:
    """(start, content) of every brace-balanced \\boxed{...}."""
    spans, pos = [], 0
    while (start := text.find("\\boxed{", pos)) >= 0:
        i, depth, end = start + len("\\boxed{"), 1, None
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        spans.append((start, text[i:end].strip()))
        pos = end + 1
    return spans


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (brace-balanced)."""
    spans = _boxed_spans(text)
    return spans[-1][1] if spans else None


MULTI_PROMPT = """Answer all {k} questions below. They are unrelated; research them in any order.

{questions}

When you have answered all of them, stop calling tools and give your final answers in this format, one line per question, in order (write your best guess if unsure):
{format}"""
_NUMBERED_LINE = re.compile(r"^\W*(?:question\s*)?(\d+)\W", re.I)


def multi_question_prompt(questions: list[str]) -> str:
    qs = "\n\n".join(f"Question {i + 1}: {q}" for i, q in enumerate(questions))
    fmt = "\n".join(f"{i + 1}. \\boxed{{answer to question {i + 1}}}" for i in range(len(questions)))
    return MULTI_PROMPT.format(k=len(questions), questions=qs, format=fmt)


def extract_answers(text: str, k: int) -> list[str | None]:
    """Answers to k questions from "i. \\boxed{...}" lines; without numbering, exactly k boxes are taken in order."""
    spans = _boxed_spans(text)
    answers: list[str | None] = [None] * k
    numbered = False
    for start, content in spans:
        line = text[text.rfind("\n", 0, start) + 1 : start]
        m = _NUMBERED_LINE.match(line)
        if m and 1 <= int(m.group(1)) <= k:
            answers[int(m.group(1)) - 1] = content or None
            numbered = True
    if not numbered and len(spans) == k:
        answers = [c or None for _, c in spans]
    return answers


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_em(pred: str, gold: str) -> tuple[float, float]:
    p, g = normalize(pred).split(), normalize(gold).split()
    em = float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0, em
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec), em


def _meta(task: Task | dict) -> dict:
    return (task.metadata or {}) if isinstance(task, Task) else task


def row_extra(meta: dict) -> dict:
    """The row's ``extra`` field (a JSON string, so every dataset row has the same columns)."""
    x = meta.get("extra")
    if isinstance(x, dict):
        return x
    try:
        return json.loads(x) if x else {}
    except (TypeError, json.JSONDecodeError):
        return {}


@rllm.evaluator
def f1_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    if (_meta(task).get("kind") or "single") != "single":
        return _evaluate(task, episode, use_judge=False)
    gold = str(_meta(task).get("answer", ""))
    pred = (episode.artifacts or {}).get("answer")
    if not pred:
        return EvalOutput(
            reward=0.0,
            is_correct=False,
            signals=[Signal(name="f1", value=0.0), Signal(name="em", value=0.0), Signal(name="answered", value=0.0)],
        )
    f1, em = f1_em(pred, gold)
    return EvalOutput(
        reward=f1,
        is_correct=em > 0,
        signals=[Signal(name="f1", value=f1), Signal(name="em", value=em), Signal(name="answered", value=1.0)],
    )


# ---------------------------------------------------------------- LLM judge
JUDGE_PROMPT = """Judge whether the [response] to the [question] is correct, based on the precise and unambiguous [correct_answer].

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Focus only on whether the response and the correct answer refer to the same thing; do not solve the problem yourself. Allow differences in formatting, spelling variants, abbreviations, extra or missing honorifics, and a small margin of error for numbers. Answer 'no' if the response is wrong, ambiguous, lists several candidates, or is only partly correct.

Reply with exactly one line: "correct: yes" or "correct: no"."""


@functools.lru_cache(maxsize=100_000)
def judge_answer(question: str, gold: str, pred: str) -> bool | None:
    """True/False from the LLM judge, or None if it could not be reached."""
    try:
        client, model = _judge()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": JUDGE_PROMPT.format(question=question, response=pred, correct_answer=gold)}
            ],
            max_completion_tokens=2048,
        )
        text = (resp.choices[0].message.content or "").lower()
    except Exception as e:  # noqa: BLE001
        logger.warning("judge failed (%s)", type(e).__name__)
        return None
    m = re.search(r"correct:\s*(yes|no)", text)
    return None if m is None else m.group(1) == "yes"


@rllm.evaluator
def judge_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    """Binary reward from the LLM judge; F1/EM are logged as signals. Falls back to F1 >= 0.8 if the judge is unreachable.
    Multi-question rows get the mean over their questions, ContextBench rows the official grader's score."""
    meta = _meta(task)
    if (meta.get("kind") or "single") != "single":
        return _evaluate(task, episode, use_judge=True)
    gold = str(meta.get("answer", ""))
    pred = (episode.artifacts or {}).get("answer")
    if not pred:
        zero = [Signal(name=n, value=0.0) for n in ("judge", "f1", "em", "answered", "judge_failed", "single_reward")]
        return EvalOutput(reward=0.0, is_correct=False, signals=zero)
    ok, f1, em, failed = _score(str(meta.get("question", "")), gold, str(pred), True)
    signals = [
        Signal(name="judge", value=ok),
        Signal(name="f1", value=f1),
        Signal(name="em", value=em),
        Signal(name="answered", value=1.0),
        Signal(name="judge_failed", value=float(failed)),
        Signal(name="single_reward", value=ok),
    ]
    return EvalOutput(reward=ok, is_correct=bool(ok), signals=signals)


def _score(question: str, gold: str, pred: str, use_judge: bool) -> tuple[float, float, float, bool]:
    """(score, f1, em, judge_failed); score is the judge verdict, or F1 without the judge."""
    f1, em = f1_em(pred, gold)
    if not use_judge:
        return f1, f1, em, False
    ok = judge_answer(question, gold, pred)
    if ok is None:
        return float(f1 >= 0.8), f1, em, True
    return float(ok), f1, em, False


def _evaluate(task: Task | dict, episode: Episode, use_judge: bool) -> EvalOutput:
    """Multi-question (kind=multi) and ContextBench (kind=cb) rows; signals carry a per-kind reward for wandb."""
    meta, art = _meta(task), episode.artifacts or {}
    kind = meta.get("kind")
    if kind == "cb":
        r = float(art.get("cb_reward") or 0.0)
        return EvalOutput(reward=r, is_correct=r >= 1.0, signals=[Signal(name="cb_reward", value=r)])
    if kind != "multi":
        raise ValueError(f"unknown row kind {kind!r}")
    x = row_extra(meta)
    qs, golds = x["questions"], x["answers"]
    preds = list(art.get("answers") or [])
    preds += [None] * (len(qs) - len(preds))
    res = [_score(q, str(g), str(p), use_judge) if p else (0.0, 0.0, 0.0, False) for q, g, p in zip(qs, golds, preds)]
    mean = lambda i: sum(r[i] for r in res) / len(res)  # noqa: E731
    reward = mean(0)
    signals = [
        Signal(name="judge" if use_judge else "f1", value=reward),
        Signal(name="f1", value=mean(1)),
        Signal(name="em", value=mean(2)),
        Signal(name="answered", value=sum(p is not None for p in preds[: len(qs)]) / len(qs)),
        Signal(name="judge_failed", value=float(any(r[3] for r in res))),
        Signal(name="multi_reward", value=reward),
        Signal(name="multi_all_correct", value=float(all(r[0] >= 1.0 for r in res))),
    ]
    return EvalOutput(reward=reward, is_correct=all(r[0] >= 1.0 for r in res), signals=signals)


EVALUATORS = {"f1": f1_evaluator, "judge": judge_evaluator}
