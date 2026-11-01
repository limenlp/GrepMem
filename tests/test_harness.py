"""grep_history tool, multi-question answers/rewards and the harness's free grep turns (no vLLM, no retrieval).

python3 -m pytest tests/test_harness.py -q      (needs rllm and openai installed)
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace as NS

from rllm.types import AgentConfig, Episode, Task

from agents import common, flows
from agents import memory as mem
from agents.common import extract_answers, f1_evaluator, grep_history, multi_question_prompt

HISTORY = [
    "[question]\nWho founded Acme Corp?",
    '[turn 1 tool search({"query": "Acme"})]\n'
    + "\n".join(f"line {i} filler" for i in range(30))
    + "\nAcme was founded by Jane Roe in 1901.",
    "[turn 2 assistant]\nAcme: founder Jane Roe?",
    '[turn 3 tool get_document({"docid": "7"})]\n' + "x" * 1000 + " Acme " + "y" * 1000,
    '[turn 4 tool grep_history({"pattern": "Acme"})]\n== [turn 1 ...]\n  31: Acme was founded by Jane Roe in 1901.',
]


def test_grep_labels_counts_and_skips_grep_output():
    out = grep_history(HISTORY, "acme", max_matches=10, context_lines=0)
    head = out.splitlines()[0]
    # question, search result, reply, document; the earlier grep output is not searched
    assert (
        head.startswith("4 matching line(s) in 4 history entries")
        and "2 in tool outputs" in head
        and "1 in your replies" in head
    )
    assert "== [turn 1 tool search" in out and "== [turn 3 tool get_document" in out and "grep_history" not in out
    assert "  31: Acme was founded by Jane Roe in 1901." in out
    long_line = next(line for line in out.splitlines() if "Acme " in line and "x" in line)
    assert len(long_line) < common.GREP_LINE_CHARS + 20 and long_line.count("...") == 2  # clipped around the match
    assert out.endswith("(showing matches 1-4 of 4)")


def test_grep_count_paging_context_and_errors():
    assert grep_history(HISTORY, "filler", count=True) == "30 matching line(s) in 1 history entry (30 in tool outputs)."
    assert grep_history(HISTORY, "filler", count="true").startswith("30 matching")
    page = grep_history(HISTORY, "filler", max_matches=10, offset=10, context_lines=0)
    assert "  11: line 10 filler" in page and "  20: line 19 filler" in page and "line 20 filler" not in page
    assert page.endswith("(showing matches 11-20 of 30; call again with offset=20 for more)")
    assert grep_history(HISTORY, "filler", max_matches=1000).count("filler") >= common.GREP_MAX_MATCHES  # clamped
    ctx = grep_history(HISTORY, r"line 5 ", max_matches=1, context_lines=1)
    assert "  5- line 4 filler" in ctx and "  6: line 5 filler" in ctx and "  7- line 6 filler" in ctx
    assert grep_history(HISTORY, "nothing here").startswith("No matches")
    assert grep_history(HISTORY, "filler", offset=99).endswith("No matches at offset=99.")
    assert grep_history(HISTORY, "(").startswith("Error: invalid regex")


def test_grep_slow_pattern_times_out(monkeypatch):
    # A backtracking pattern on one multi-MB line (as in AMA-Bench observations) used to hang the event loop.
    monkeypatch.setattr(common, "GREP_TIMEOUT", 0.5)
    line = '{"id":"1A","region":"x","date":"2020","value":1.5},' * 60000
    t = time.monotonic()
    out = grep_history([f"[step 3]\n{line}"], '"id":"1A".*date.*value.*ARB')
    assert out.startswith("Error:") and "took over 0.5s" in out and time.monotonic() - t < 5
    assert grep_history([f"[step 3]\n{line[:2000]}"], "region").startswith("1 matching line")


def test_grep_output_cap_pages_instead_of_cutting(monkeypatch):
    monkeypatch.setattr(common, "GREP_CHARS", 400)
    out = grep_history(HISTORY, "filler", max_matches=30, context_lines=0)
    assert len(out) < 600 and "call again with offset=" in out


def test_extract_answers():
    text = "Done.\n1. \\boxed{Paris}\n2) \\boxed{}\n**3.** \\boxed{1901}"
    assert extract_answers(text, 3) == ["Paris", None, "1901"]
    assert extract_answers("\\boxed{a} and \\boxed{b}", 2) == ["a", "b"]  # unnumbered: in order when exactly k
    assert extract_answers("\\boxed{a}", 2) == [None, None]
    assert "Question 2: Q2" in multi_question_prompt(["Q1", "Q2"])


def test_multi_reward_is_mean_over_questions():
    row = {
        "id": "m",
        "kind": "multi",
        "question": "...",
        "answer": "Paris | 1901",
        "extra": json.dumps({"questions": ["capital?", "year?"], "answers": ["Paris", "1901"]}),
    }
    ep = Episode(artifacts={"answer": "Paris | 1900", "answers": ["Paris", "1900"]})
    out = f1_evaluator.evaluate(row, ep)
    sig = {s.name: s.value for s in out.signals}
    assert out.reward == 0.5 and not out.is_correct and sig["multi_reward"] == 0.5 and sig["answered"] == 1.0
    cb = f1_evaluator.evaluate({"id": "c", "kind": "cb"}, Episode(artifacts={"cb_reward": 0.75}))
    assert cb.reward == 0.75


class FakeClient:
    """Scripted chat completions: search, (compaction), 3 grep-only replies, then the numbered answers."""

    def __init__(self, *a, **k):
        self.chat = NS(completions=NS(create=self.create))
        self.n = 0

    async def create(self, model, messages, tools, **kw):
        last = messages[-1]
        if last["role"] == "user" and last["content"] == flows.SUMMARY_PROMPT:
            content, calls = "Found Paris for Q1.", None
        elif not any(m["role"] == "tool" for m in messages) and "compacted" not in messages[1]["content"]:
            content, calls = "", [("search", {"query": "capital"})]
        elif self.n < 3:
            self.n += 1
            content, calls = "", [("grep_history", {"pattern": "Paris|1901", "count": self.n == 1})]
        else:
            content, calls = "1. \\boxed{Paris}\n2. \\boxed{1901}", None
        tcs = [
            NS(id=f"c{i}", type="function", function=NS(name=n, arguments=json.dumps(a)))
            for i, (n, a) in enumerate(calls or [])
        ]
        msg = NS(role="assistant", content=content, reasoning_content=None, tool_calls=tcs or None)
        prompt = sum(len(m.get("content") or "") for m in messages) // 3
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=prompt, completion_tokens=20))


def test_grep_only_turns_are_free(monkeypatch):
    async def fake_search(corpus, query, k=10):
        return "Paris is the capital. 1901 was the year.\n" + "filler words " * 4000  # forces a compaction

    monkeypatch.setattr(flows, "AsyncOpenAI", FakeClient)
    monkeypatch.setattr(flows, "search", fake_search)
    monkeypatch.setattr(flows, "COMPACT_AT", 8000)
    monkeypatch.setattr(flows, "MULTI_TURNS_PER_Q", 1)  # 2 questions -> 2 counted turns
    row = {
        "id": "m",
        "kind": "multi",
        "question": multi_question_prompt(["capital?", "year?"]),
        "answer": "",
        "corpus": "x",
        "extra": json.dumps({"questions": ["capital?", "year?"], "answers": ["Paris", "1901"]}),
    }
    ep = asyncio.run(
        flows.run_agent(
            Task(id="m", instruction="", metadata=row),
            AgentConfig(base_url="x", model="m", session_uid="s"),
            "summary_grep",
        )
    )
    a = ep.artifacts
    assert a["termination"] == "answered" and a["answers"] == ["Paris", "1901"], a
    assert a["turns"] == 2 and a["grep_turns"] == 3 and a["grep_calls"] == 3 and a["compactions"] == 1
    assert ep.metrics["grep_turns"] == 3.0 and a["llm_calls"] == len(ep.trajectories[0].steps) == 6


def test_episode_cost_sums_usage_over_every_call(monkeypatch):
    clients = []

    class UsageClient(FakeClient):
        def __init__(self, *a, **k):
            super().__init__()
            self.prompts = []
            clients.append(self)

        async def create(self, model, messages, tools, **kw):
            r = await super().create(model, messages, tools, **kw)
            self.prompts.append(r.usage.prompt_tokens)
            r.usage.prompt_tokens_details = NS(cached_tokens=100)
            return r

    async def fake_search(corpus, query, k=10):
        return "Paris is the capital. 1901 was the year.\n" + "filler words " * 4000

    monkeypatch.setattr(flows, "AsyncOpenAI", UsageClient)
    monkeypatch.setattr(flows, "search", fake_search)
    monkeypatch.setattr(flows, "COMPACT_AT", 8000)
    row = {
        "id": "m",
        "kind": "multi",
        "question": multi_question_prompt(["capital?", "year?"]),
        "answer": "",
        "corpus": "x",
        "extra": json.dumps({"questions": ["capital?", "year?"], "answers": ["Paris", "1901"]}),
    }
    ep = asyncio.run(
        flows.run_agent(
            Task(id="m", instruction="", metadata=row),
            AgentConfig(base_url="x", model="m", session_uid="s"),
            "summary_grep",
        )
    )
    a, n = ep.artifacts, ep.artifacts["llm_calls"]
    assert a["compactions"] == 1 and len(clients[0].prompts) == n  # the compaction call is counted too
    assert (
        a["prompt_tokens"] == sum(clients[0].prompts)
        and a["completion_tokens"] == 20 * n
        and a["cached_tokens"] == 100 * n
    )
    assert a["wall_s"] >= 0 and "wall_s" in ep.metrics


class SearchForeverClient:
    """Always searches while tools are offered; answers only when called without tools."""

    def __init__(self, *a, **k):
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, model, messages, tools=None, **kw):
        if tools:
            calls = [
                NS(id="c0", type="function", function=NS(name="search", arguments=json.dumps({"query": "capital"})))
            ]
            msg = NS(role="assistant", content="", reasoning_content=None, tool_calls=calls)
        else:
            assert messages[-1]["content"].startswith("You have used all your turns")
            msg = NS(role="assistant", content="Best guess: \\boxed{Paris}", reasoning_content=None, tool_calls=None)
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=500, completion_tokens=20))


def test_forced_answer_when_turns_run_out(monkeypatch):
    async def fake_search(corpus, query, k=10):
        return "Paris"

    monkeypatch.setattr(flows, "AsyncOpenAI", SearchForeverClient)
    monkeypatch.setattr(flows, "search", fake_search)
    monkeypatch.setattr(flows, "MAX_TURNS", 2)
    row = {"id": "s", "kind": "single", "question": "capital?", "answer": "Paris", "corpus": "x", "extra": "{}"}
    ep = asyncio.run(
        flows.run_agent(
            Task(id="s", instruction="", metadata=row), AgentConfig(base_url="x", model="m", session_uid="s"), "summary"
        )
    )
    a = ep.artifacts
    assert (
        a["termination"] == "forced_answer" and a["answer"] == "Paris" and a["turns"] == 2 and a["forced_answer"] == 1
    )
    assert a["llm_calls"] == len(ep.trajectories[0].steps) == 3
    monkeypatch.setattr(flows, "FORCE_ANSWER", False)
    ep = asyncio.run(
        flows.run_agent(
            Task(id="s", instruction="", metadata=row), AgentConfig(base_url="x", model="m", session_uid="s"), "summary"
        )
    )
    assert ep.artifacts["termination"] == "max_turns" and ep.artifacts["answer"] is None


# ---------------------------------------------------------------- baselines: mask, memgpt (agents/memory.py)


def test_memgpt_memory_tools():
    m = mem.MemGPTMemory()
    hist = [
        "[question]\nWho founded Acme?",
        "[turn 1 tool search({})]\n" + "pad " * 400 + "ACME was founded by Jane Roe.",
        "[turn 2 tool working_context_append({})]\nWorking context updated: acme",
        "[turn 3 assistant]\nnothing here",
    ]
    out, key = m.call("conversation_search", {"query": "acme"}, hist)
    assert key == "conv_search_calls" and out.startswith("2 matching message(s); showing 1-2")
    assert "ACME was founded" in out and "working_context_append" not in out  # memory tool outputs are not searched
    assert out.count("pad") < 400 and "\n..." in out  # long messages are clipped around the match
    page, _ = m.call(
        "conversation_search", {"query": "note", "page": 2}, [f"[turn {i} assistant]\nnote {i}" for i in range(12)]
    )
    assert (
        page.startswith("12 matching message(s); showing 11-12 (page 2 of pages 0-2)")
        and "note 10" in page
        and "note 9" not in page
    )
    assert m.call("conversation_search", {"query": "zzz"}, hist)[0].startswith("No messages")
    assert m.call("working_context_append", {"content": "a" * 10}, [])[1] == "wc_edits"
    assert (
        m.call("working_context_append", {"content": "b" * mem.WC_CHARS}, [])[1] == "wc_errors"
        and m.working == "a" * 10
    )
    assert m.call("working_context_replace", {"old_content": "zz", "new_content": ""}, [])[1] == "wc_errors"
    out, _ = m.call("working_context_replace", {"old_content": "a" * 10, "new_content": "x"}, [])
    assert (
        m.working == "x"
        and out.endswith("It now reads:\nx")
        and "<working_context>\nx\n</working_context>" in m.block()
    )
    for p in ("Paris Hilton is a celebrity", "the capital of France is Paris", "Acme was founded in 1901"):
        m.call("archival_memory_insert", {"content": p}, [])
    out, key = m.call("archival_memory_search", {"query": "Paris capital"}, [])
    assert (
        key == "archival_search_calls"
        and out.startswith("2 matching passage(s)")
        and out.index("[passage 2]") < out.index("[passage 1]")
    )
    assert m.call("archival_memory_search", {"query": " "}, []) is None and m.call("search", {"query": "x"}, []) is None
    m.call("archival_memory_insert", {"content": "\n".join(f"K{i:05d} = value {i}" for i in range(2000))}, [])
    out, _ = m.call("archival_memory_search", {"query": "K01999"}, [])  # a long passage is shown around the match
    assert "K01999 = value 1999" in out and len(out) < 2 * mem.SNIPPET_CHARS


def _reply(content="", calls=(), reasoning=None):
    tcs = [
        NS(id=f"c{i}", type="function", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(calls)
    ]
    return NS(role="assistant", content=content, reasoning_content=reasoning, tool_calls=tcs or None)


class ScriptClient:
    """Fake server that records every prompt; ``script(messages) -> message`` decides the reply. Tokens = chars / 3."""

    prompts: list = []

    def __init__(self, *a, **k):
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, model, messages, tools=None, **kw):
        type(self).prompts.append([dict(m) for m in messages])
        type(self).tools = tools
        prompt = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in messages) // 3
        return NS(
            choices=[NS(message=type(self).script(messages), finish_reason="stop")],
            usage=NS(prompt_tokens=prompt, completion_tokens=20),
        )


def _run(monkeypatch, client, mode, n_search_chars=6000):
    async def fake_search(corpus, query, k=10):
        return f"Paris is the capital ({query}). " + "filler " * (n_search_chars // 7)

    client.prompts = []
    monkeypatch.setattr(flows, "AsyncOpenAI", client)
    monkeypatch.setattr(flows, "search", fake_search)
    monkeypatch.setattr(flows, "COMPACT_AT", 8000)
    row = {"id": "s", "kind": "single", "question": "capital?", "answer": "Paris", "corpus": "x", "extra": "{}"}
    return asyncio.run(
        flows.run_agent(
            Task(id="s", instruction="", metadata=row), AgentConfig(base_url="x", model="m", session_uid="s"), mode
        )
    )


def test_compaction_tool_call_falls_back_to_vllm_reasoning(monkeypatch):
    class Client(ScriptClient):
        @staticmethod
        def script(messages):
            if messages[-1]["role"] == "user" and messages[-1]["content"] == flows.SUMMARY_PROMPT:
                msg = _reply(calls=[("search", {"query": "more"})])  # ignores "do not call tools"
                msg.reasoning_content, msg.reasoning = None, "Progress: the capital is Paris."  # vLLM's field name
                return msg
            if "compacted" in messages[1]["content"]:
                return _reply("\\boxed{Paris}")
            return _reply(calls=[("search", {"query": "q"})])

    ep = _run(monkeypatch, Client, "summary", n_search_chars=13500)
    assert ep.artifacts["compactions"] == 1 and "Progress: the capital is Paris." in Client.prompts[-1][1]["content"]


def test_mask_replaces_old_tool_outputs(monkeypatch):
    class Client(ScriptClient):
        @staticmethod
        def script(messages):
            n = sum(1 for m in messages if m.get("tool_calls"))
            return _reply(calls=[("search", {"query": f"q{n}"})]) if n < 10 else _reply("\\boxed{Paris}")

    ep = _run(monkeypatch, Client, "mask")
    a = ep.artifacts
    assert a["termination"] == "answered" and a["turns"] == 11 and a["compactions"] == 0 and a["trimmed_turns"] == 0
    assert (
        a["mask_events"] >= 2 and a["masked"] >= 6 and a["llm_calls"] == len(ep.trajectories[0].steps) == 11
    )  # no summary calls
    assert max(sum(len(m["content"] or "") for m in p) // 3 for p in Client.prompts) <= 8000
    final = Client.prompts[-1]
    tools = [m["content"] for m in final if m["role"] == "tool"]
    assert len(tools) == 10 and sum(t.startswith("[Old tool output removed") for t in tools) == a["masked"]
    assert tools[-1].startswith("Paris is the capital (q9)")  # the newest output is never masked
    assert "outputs of your oldest tool calls are replaced" in final[0]["content"]


def test_mask_drops_old_turns_but_never_cuts_unseen_outputs(monkeypatch):
    class Client(ScriptClient):
        n = 0

        @staticmethod
        def script(messages):
            Client.n += 1
            return _reply(calls=[("search", {"query": "q"})]) if Client.n <= 6 else _reply("\\boxed{Paris}")

    ep = _run(monkeypatch, Client, "mask", n_search_chars=13500)  # one output > the mask target (half of COMPACT_AT)
    a = ep.artifacts
    assert a["termination"] == "answered" and a["masked"] > 0 and a["trimmed_turns"] > 0, a
    assert not any("truncated to fit" in (m["content"] or "") for p in Client.prompts for m in p)


def test_mask_strips_old_reasoning_but_keeps_actions(monkeypatch):
    class Client(ScriptClient):
        n = 0

        @staticmethod
        def script(messages):
            Client.n += 1
            if Client.n <= 12:
                return _reply(calls=[("search", {"query": f"q{Client.n}"})], reasoning="think " * 300)
            return _reply("\\boxed{Paris}")

    ep = _run(monkeypatch, Client, "mask", n_search_chars=3000)
    a = ep.artifacts
    assert a["termination"] == "answered" and a["reasoning_stripped"] > 0 and a["trimmed_turns"] == 0, a
    final = Client.prompts[-1]
    assert [m["tool_calls"][0]["function"]["arguments"] for m in final if m.get("tool_calls")] == [
        json.dumps({"query": f"q{i}"}) for i in range(1, 13)
    ]
    assert not final[2].get("reasoning_content") and final[-2].get("reasoning_content")  # oldest stripped, newest kept


def test_one_turn_over_the_window_does_not_loop(monkeypatch):
    # The first reply's outputs alone exceed the window: nothing old to mask or evict. This used to loop forever
    # (making room never got the estimate under COMPACT_AT, and no LLM call was made in between).
    for mode in ("mask", "memgpt"):

        class Client(ScriptClient):
            @staticmethod
            def script(messages):
                if not any(m["role"] == "tool" for m in messages):
                    return _reply(calls=[("search", {"query": f"q{i}"}) for i in range(3)])
                return _reply("\\boxed{Paris}")

        ep = asyncio.run(asyncio.wait_for(asyncio.to_thread(_run, monkeypatch, Client, mode, 12000), timeout=20))
        assert ep.artifacts["termination"] == "answered" and ep.artifacts["turns"] == 2, (mode, ep.artifacts)
        assert sum(len(m["content"] or "") for m in Client.prompts[-1]) // 3 <= 8000, mode


def test_memgpt_warning_flush_and_memory_tools(monkeypatch):
    class Client(ScriptClient):
        @staticmethod
        def script(messages):
            last = messages[-1]
            if last["role"] == "user" and last["content"] == mem.SUMMARY_PROMPT:
                return (
                    _reply("Looking for the capital; Paris is likely.")
                    if Client.tools is None
                    else _reply(calls=[("search", {"query": "x"})])
                )
            names = [tc["function"]["name"] for m in messages for tc in m.get("tool_calls") or []]
            if "[Summary of the" not in messages[1]["content"]:
                if not names:
                    return _reply(calls=[("working_context_append", {"content": "Q: capital; candidate Paris"})])
                if "Memory pressure" in (last.get("content") or ""):
                    return _reply(calls=[("archival_memory_insert", {"content": "Paris is the capital (doc 7)"})])
                return _reply(calls=[("search", {"query": f"q{len(names)}"})])
            if "conversation_search" not in names:
                return _reply(calls=[("conversation_search", {"query": "PARIS IS THE"})])
            if "archival_memory_search" not in names:
                return _reply(calls=[("archival_memory_search", {"query": "capital doc"})])
            return _reply("\\boxed{Paris}")

    ep = _run(monkeypatch, Client, "memgpt")
    a = ep.artifacts
    assert a["termination"] == "answered" and a["answer"] == "Paris", a
    assert a["compactions"] == 1 and a["mem_warnings"] == 1 and a["wc_edits"] == 1 and a["archival_inserts"] == 1
    assert a["conv_search_calls"] == 1 and a["conv_search_empty"] == 0 and a["archival_search_calls"] == 1
    assert (
        a["mem_turns"] == 4 and a["turns"] == a["search_calls"] + 1 and a["llm_calls"] == len(ep.trajectories[0].steps)
    )
    assert max(sum(len(m["content"] or "") for m in p) // 3 for p in Client.prompts) <= 8000
    assert "<working_context>\n\n</working_context>" in Client.prompts[0][0]["content"]
    final = Client.prompts[-1]
    # the flush re-rendered the working context and put the recursive summary after the task
    assert "<working_context>\nQ: capital; candidate Paris\n</working_context>" in final[0]["content"]
    assert (
        final[1]["content"].startswith("capital?\n\n---\n[Summary of the") and "Paris is likely" in final[1]["content"]
    )
    outs = [m["content"] for m in final if m["role"] == "tool"]
    assert any(o.startswith(f"{a['search_calls']} matching message(s)") for o in outs)  # finds evicted search results
    assert any("[passage 1]\nParis is the capital (doc 7)" in o for o in outs)
