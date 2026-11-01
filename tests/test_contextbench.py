"""ContextBench driver tests with a scripted fake model (no vLLM).

  python3 -m pytest tests/test_contextbench.py -q      (needs rllm, openai and contextbench installed)

The fake recalls a KV value only from its live context, writes summaries that keep nothing, and greps the history
when grep_history is offered. So on a stream larger than the window: react hits the budget wall, summary forgets
most keys, and summary_grep recovers every key through grep -- which checks streaming, compaction, the grep tool,
the budget wall and the official grader end to end.
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace as NS

import pytest
from rllm.types import AgentConfig, Step, Task

from agents import contextbench as cbmod
from agents import memory as mem
from agents.common import f1_evaluator
from agents.contextbench import STUCK_TURNS, SUMMARY_PROMPT, CBAgent, run_instance
from cb_eval import build_instances, summarize
from data.prepare_data import cb_rows


class FakeKVAgent(CBAgent):
    def __init__(self, mode):
        super().__init__(mode, "http://unused/v1", "fake")

    async def _call(self, msgs, use_tools=True):
        self.stats["llm_calls"] += 1
        last = msgs[-1]
        tool_calls = None
        if last["role"] == "user" and last["content"] == SUMMARY_PROMPT:
            content = "I have been storing SET batches."
        elif last["role"] == "tool":
            key = re.search(r"SET (K\d+) =", last["content"])
            val = re.search(r"SET K\d+ = (.+)", last["content"])
            content = (
                f"<<<ANSWER key={key.group(1)}>>>\n{val.group(1).strip()}\n<<<ANSWER END>>>\nREADY_FOR_NEXT_OP"
                if val
                else "READY_FOR_NEXT_OP"
            )
        else:
            get = re.search(r"^GET (K\d+)", last["content"], re.M) if last["role"] == "user" else None
            if get:
                key = get.group(1)
                ctx = "\n".join(m["content"] for m in msgs if m["role"] == "user")
                hit = re.search(rf"^SET {key} = (.+)$", ctx, re.M)
                if hit:
                    content = f"<<<ANSWER key={key}>>>\n{hit.group(1).strip()}\n<<<ANSWER END>>>\nREADY_FOR_NEXT_OP"
                elif self.grep_on:
                    content = ""
                    tool_calls = [
                        NS(
                            id="c0",
                            type="function",
                            function=NS(
                                name="grep_history",
                                arguments=json.dumps({"pattern": f"SET {key} =", "context_lines": 0}),
                            ),
                        )
                    ]
                else:
                    content = f"<<<ANSWER key={key}>>>\nunknown\n<<<ANSWER END>>>\nREADY_FOR_NEXT_OP"
            else:
                content = "Noted.\nREADY_FOR_NEXT_OP"
        prompt = sum(len(m.get("content") or "") for m in msgs) // 4
        msg = NS(role="assistant", content=content, reasoning_content=None, tool_calls=tool_calls)
        return NS(message=msg, finish_reason="stop"), prompt + 50, prompt


@pytest.fixture(scope="module")
def kv_instance(tmp_path_factory):
    return build_instances("kv_store:R1000", [42], tmp_path_factory.mktemp("cb"))[0]


@pytest.mark.parametrize("mode,lo,hi", [("react", 0.0, 0.0), ("summary", 0.0, 0.75), ("summary_grep", 1.0, 1.0)])
def test_kv_harnesses(kv_instance, tmp_path, mode, lo, hi):
    agent = FakeKVAgent(mode)
    res = asyncio.run(run_instance(agent, kv_instance["dir"], tmp_path / mode))
    assert lo <= res["reward"] <= hi, res
    if mode == "react":
        assert res["terminated_by"] == "budget_wall"
    else:
        assert res["terminated_by"] == "stream_exhausted" and res["compactions"] >= 1
    assert (res["grep_calls"] > 0) == (mode == "summary_grep")
    assert res["rewards"]["graded_from"] == "snapshots"
    # failure reasons: summary answers "unknown" for forgotten keys; summary_grep greps every key it answers
    assert res["kv_wrong"] + res["kv_hash_dropped"] + res["kv_unanswered"] == round(24 * (1 - res["reward"]))
    if mode == "summary_grep":
        assert res["kv_correct_grepped"] + res["kv_correct_ungrepped"] == 24 and res["kv_grepped"] > 0


class SilentAgent(FakeKVAgent):
    async def _call(self, msgs, use_tools=True):
        choice, total, prompt = await super()._call(msgs)
        choice.message.content = "Thinking about it."  # never emits READY_FOR_NEXT_OP
        return choice, total, prompt


def test_stuck_run_ends_early(kv_instance, tmp_path):
    agent = SilentAgent("summary")
    res = asyncio.run(run_instance(agent, kv_instance["dir"], tmp_path / "stuck"))
    assert res["terminated_by"] == "stuck" and res["n_turns"] == STUCK_TURNS


def test_cb_training_flow(monkeypatch, tmp_path):
    class StepFake(FakeKVAgent):
        def __init__(self, mode, base_url, model, max_retries=3):
            super().__init__(mode)

        async def _call(self, msgs, use_tools=True):
            r = await super()._call(msgs)
            self.steps.append(Step(chat_completions=list(msgs), model_response=r[0].message.content))
            return r

    monkeypatch.setattr(cbmod, "CBAgent", StepFake)
    monkeypatch.setattr(cbmod, "CB_TRAIN_DIR", str(tmp_path))
    row = cb_rows(1, 1000)[0]
    assert row["kind"] == "cb" and json.loads(row["extra"])["knobs"]["seed"] == 1000
    ep = asyncio.run(
        cbmod.run_cb_episode(
            Task(id=row["id"], instruction="", metadata=row),
            AgentConfig(base_url="x", model="m", session_uid="s"),
            "summary_grep",
        )
    )
    assert ep.artifacts["cb_reward"] == 1.0 and len(ep.trajectories[0].steps) == ep.artifacts["llm_calls"] > 0
    assert ep.metrics["cb_grep_calls"] > 0 and f1_evaluator.evaluate(row, ep).reward == 1.0
    assert not list(tmp_path.glob("run-*"))  # per-episode snapshots are removed


def test_summarize_macro_average():
    rows = [
        {
            "family": "kv_store",
            "level": "R1000",
            "pressure": 1.5,
            "reward": 1.0,
            "terminated_by": "stream_exhausted",
            "compactions": 2,
            "grep_calls": 3,
        },
        {
            "family": "log_triage",
            "level": "L640",
            "pressure": 0.86,
            "reward": 0.5,
            "terminated_by": "stream_exhausted",
            "compactions": 0,
            "grep_calls": 0,
        },
    ]
    s = summarize(rows)
    assert s["reward"] == 0.75 and s["kv_store/R1000/reward"] == 1.0 and "log_triage/reward_over_budget" not in s


class MemGPTKVAgent(FakeKVAgent):
    """FakeKVAgent that looks up a key it no longer has in context with conversation_search (memgpt only)."""

    async def _call(self, msgs, use_tools=True):
        last = msgs[-1]
        if last["role"] == "user" and last["content"] == mem.SUMMARY_PROMPT:
            assert not use_tools  # MemGPT's summarizer has no tools
            self.stats["llm_calls"] += 1
            return (
                NS(
                    message=NS(
                        role="assistant",
                        content="I have been storing SET batches.",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                ),
                100,
                50,
            )
        if last["role"] == "tool":
            self.stats["llm_calls"] += 1
            val = re.search(rf"SET {self.pending} = (.+)", last["content"])
            content = f"<<<ANSWER key={self.pending}>>>\n{val.group(1).strip() if val else 'unknown'}\n<<<ANSWER END>>>\nREADY_FOR_NEXT_OP"
            return (
                NS(
                    message=NS(role="assistant", content=content, reasoning_content=None, tool_calls=None),
                    finish_reason="stop",
                ),
                100,
                50,
            )
        choice, total, prompt = await super()._call(msgs)
        if "\nunknown\n" in choice.message.content:
            self.pending = re.search(r"key=(K\d+)", choice.message.content).group(1)
            args = json.dumps({"query": f"SET {self.pending} ="})
            choice.message.content = ""
            choice.message.tool_calls = [
                NS(id="c0", type="function", function=NS(name="conversation_search", arguments=args))
            ]
        return choice, total, prompt


@pytest.mark.parametrize("mode", ["mask", "memgpt"])
def test_kv_baselines(kv_instance, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(cbmod, "COMPACT_AT", 16000)
    agent = (MemGPTKVAgent if mode == "memgpt" else FakeKVAgent)(mode)
    res = asyncio.run(run_instance(agent, kv_instance["dir"], tmp_path / mode))
    assert res["terminated_by"] == "stream_exhausted" and res["grep_calls"] == 0 and res["tool_errors"] == 0, res
    if mode == "mask":  # replies keep nothing, so masked SETs are forgotten; no extra LLM calls
        assert res["masked"] > 0 and res["mask_events"] > 0 and res["compactions"] == 0 and 0 < res["reward"] < 1
        assert res["llm_calls"] == res["n_turns"]
        assert all(m["content"].startswith("[Old message removed") for m in agent.messages[2:6] if m["role"] == "user")
    else:  # flushes forget the SETs, conversation_search finds every one again
        assert (
            res["reward"] == 1.0
            and res["compactions"] >= 1
            and res["mem_warnings"] >= 1
            and res["conv_search_calls"] > 0
        )
        assert agent.messages[1]["content"].startswith(agent.instruction + "\n\n---\n[Summary of the")
        fork = agent.fork()
        fork.memory.call("working_context_append", {"content": "fork note"}, [])
        assert agent.memory.working == ""  # forks (AMA-Bench questions) do not share memory


def test_memgpt_tool_chain_within_one_op_stays_in_the_window(monkeypatch):
    """A chain of memory-tool calls answering one op (reasoning kept within the op) must not grow the prompt past
    COMPACT_AT: the flush cannot evict the op's own chain, so the chain itself is trimmed; the op stays."""
    monkeypatch.setattr(cbmod, "COMPACT_AT", 16000)
    prompts = []

    class ChainAgent(CBAgent):
        async def _call(self, msgs, use_tools=True):
            self.stats["llm_calls"] += 1
            prompt = int(mem.ctx_chars(msgs) / 3)
            prompts.append(prompt)
            last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
            if msgs[-1]["content"] == mem.SUMMARY_PROMPT:
                msg = NS(role="assistant", content="Summary.", reasoning_content=None, tool_calls=None)
            elif sum(m["role"] == "assistant" for m in msgs[last_user:]) < cbmod.CALLS_PER_TURN - 1:
                call = NS(
                    id=f"c{len(prompts)}",
                    type="function",
                    function=NS(name="conversation_search", arguments=json.dumps({"query": "K1"})),
                )
                msg = NS(role="assistant", content="", reasoning_content="r" * 9000, tool_calls=[call])
            else:
                msg = NS(role="assistant", content="READY_FOR_NEXT_OP", reasoning_content=None, tool_calls=None)
            return NS(message=msg, finish_reason="stop"), prompt + 3000, prompt

    agent = ChainAgent("memgpt", "http://unused/v1", "fake")
    agent.reset("You store keys.", "Answer GET ops. " + cbmod.NO_TOOLS_SENTENCE)
    for i in range(3):
        asyncio.run(agent.on_user_turn(f"SET K{i} = {'v' * 3000}\nGET K{i}"))
    assert max(prompts) <= 16000
    assert any(
        m["role"] == "assistant" and m.get("tool_calls") and "reasoning_content" not in m for m in agent.messages
    )
    assert agent.messages[-1]["content"] == "READY_FOR_NEXT_OP"
    assert [m for m in agent.messages if m["role"] == "user"][-1]["content"].startswith("SET K2 = ")
