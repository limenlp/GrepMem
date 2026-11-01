"""AMA-Bench driver tests with a scripted fake model (no vLLM, no judge).

python3 -m pytest tests/test_ama.py -q      (needs rllm, openai and contextbench installed)
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace as NS

import ama_eval
from agents import contextbench, flows
from ama_eval import AMAAgent, chunk_trajectory, file_env, run_episode, run_file_episode, summarize

EPISODE = {
    "episode_id": "e1",
    "domain": "Game",
    "task_type": "t",
    "task": "reach the goal",
    "total_tokens": "9000",
    "trajectory": [
        {"turn_idx": i, "action": f"move-{i}", "observation": f"obs-{i} " + "pad " * 700} for i in range(10)
    ],
    "qa_pairs": [
        {
            "question": f"What was observed at step {i}?",
            "answer": f"obs-{i}",
            "type": "AB"[i % 2],
            "question_uuid": f"u{i}",
        }
        for i in (1, 5, 9)
    ],
}


class FakeAMA(AMAAgent):
    """Answers a question from its live context only."""

    def __init__(self, mode):
        super().__init__(mode, "http://unused/v1", "fake")

    async def _call(self, msgs, use_tools=True):
        self.stats["llm_calls"] += 1
        last = msgs[-1]["content"]
        if last == self.summary_prompt:
            content = "Read some steps."
        elif last.startswith("The trajectory is complete"):
            step = re.search(r"step (\d+)\?", last).group(1)
            ctx = "\n".join(m["content"] for m in msgs[:-1] if m["role"] == "user")
            hit = re.search(rf"^Step {step}:\nAction: \S+\nObservation: (\S+)", ctx, re.M)
            content = hit.group(1) if hit else "unknown"
        else:
            content = "Noted."
        prompt = sum(len(m.get("content") or "") for m in msgs) // 4
        return (
            NS(
                message=NS(role="assistant", content=content, reasoning_content=None, tool_calls=None),
                finish_reason="stop",
            ),
            prompt + 20,
            prompt,
        )


def test_chunking_keeps_every_step_in_order():
    chunks = chunk_trajectory(EPISODE["trajectory"], 6000)
    text = "".join(chunks)
    assert [int(x) for x in re.findall(r"^Step (\d+):", text, re.M)] == list(range(10))
    assert len(chunks) == 5 and all(len(c) <= 6000 + 100 for c in chunks)
    parts = chunk_trajectory(EPISODE["trajectory"][:1], 1000)
    assert len(parts) == 3 and parts[0].startswith("Step 0 (part 1 of 3):")


def run(mode, monkeypatch, **patch):
    for k, v in patch.items():
        monkeypatch.setattr(contextbench, k, v)
    monkeypatch.setattr(ama_eval, "judge", lambda q, gold, pred, ep: float(gold == pred))
    agent = FakeAMA(mode)
    rec = asyncio.run(run_episode(EPISODE, agent, 6000, asyncio.Semaphore(8)))
    return agent, rec


def test_questions_are_asked_on_forks(monkeypatch):
    agent, rec = run("summary", monkeypatch)
    assert rec["acc"] == 1.0 and rec["terminated_by"] == "done" and rec["n_chunks"] == 5 and len(rec["qa"]) == 3
    assert len(agent.messages) == 2 + 2 * 5  # the forks' question turns did not touch the base conversation
    assert rec["llm_calls"] == 5 + 3


def test_compaction_forgets_and_react_hits_the_wall(monkeypatch):
    _, rec = run("summary", monkeypatch, COMPACT_AT=4000)
    assert rec["compactions"] > 0 and rec["acc"] < 1.0
    _, rec = run("react", monkeypatch, CONTEXT_BUDGET=9000)
    assert rec["terminated_by"] == "budget_wall" and rec["acc"] == 0.0
    s = summarize([rec])
    assert (
        s["n_qa"] == 3
        and s["domain/Game/acc"] == 0.0
        and s["len/le32k/acc"] == 0.0
        and s["termination"] == {"budget_wall": 1}
    )


def test_grep_trajectory_labels_steps():
    env, prompt = file_env(EPISODE)
    out = asyncio.run(env.tools["grep_trajectory"][0]({"pattern": "obs-5 ", "context_lines": 0}))
    assert out.startswith("1 matching line(s) in 1 history entry (1 in trajectory steps).") and "== [step 5]" in out
    assert "Question 3: What was observed at step 9?" in prompt and "3. \\boxed{answer to question 3}" in prompt
    assert env.n_questions == 3 and env.max_turns == ama_eval.FILE_TURNS


class FileClient:
    """Greps the trajectory for each asked step, then answers all questions."""

    def __init__(self, *a, **k):
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, model, messages, tools, **kw):
        assert [t["function"]["name"] for t in tools] == ["grep_trajectory"]
        seen = re.findall(
            r"Observation: (obs-\d+)", "\n".join(m["content"] or "" for m in messages if m["role"] == "tool")
        )
        if len(seen) < 3:
            step = [1, 5, 9][len(seen)]
            calls = [
                NS(
                    id="c",
                    type="function",
                    function=NS(
                        name="grep_trajectory", arguments=json.dumps({"pattern": f"^Observation: obs-{step} "})
                    ),
                )
            ]
            content = ""
        else:
            calls, content = None, "\n".join(f"{i + 1}. \\boxed{{{o}}}" for i, o in enumerate(seen))
        msg = NS(role="assistant", content=content, reasoning_content=None, tool_calls=calls)
        return NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=1000, completion_tokens=20))


def test_file_protocol(monkeypatch):
    monkeypatch.setattr(flows, "AsyncOpenAI", FileClient)
    monkeypatch.setattr(ama_eval, "judge", lambda q, gold, pred, ep: float(gold == pred))
    rec = asyncio.run(run_file_episode(EPISODE, "summary", "http://unused/v1", "fake"))
    assert rec["acc"] == 1.0 and rec["answered"] == 1.0 and rec["terminated_by"] == "answered"
    assert rec["trajectory_grep_calls"] == 3 and rec["grep_calls"] == 0 and rec["llm_calls"] == 4
