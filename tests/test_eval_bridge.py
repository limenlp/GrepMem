"""eval_bridge: offline eval summaries are merged into the trainer's own wandb log calls.

python3 -m pytest tests/test_eval_bridge.py -q      (needs rllm installed)
"""

from __future__ import annotations

import json

from rllm.utils import tracking

from agents import eval_bridge


class FakeWandb:
    def define_metric(self, *a, **k):
        pass


def test_bridge_logs_each_kind_once_by_eval_step(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(tracking.Tracking, "log", lambda self, data, step, *a, **k: calls.append((step, data)))
    monkeypatch.setenv("BCRL_EVAL_DIR", str(tmp_path))
    (tmp_path / "eval_bcp_logged.json").write_text("[0]")  # written by the older, BCP-only bridge
    for kind, step, key in [("bcp", 0, "acc"), ("cb", 0, "reward"), ("bcp", 10, "acc"), ("cb", 10, "kv_store/reward")]:
        (tmp_path / f"eval_{kind}_step{step}.summary.json").write_text(json.dumps({"n": 4, key: 0.5, "agent": "x"}))
    eval_bridge.install()
    t = type("T", (), {"logger": {"wandb": FakeWandb()}})()
    for step in (5, 5, 6, 7):
        tracking.Tracking.log(t, {"loss": 1.0}, step)

    logged = [(s, {k: v for k, v in d.items() if k.startswith("eval/")}) for s, d in calls]
    assert logged[0] == (5, {"eval/cb/n": 4, "eval/cb/reward": 0.5, "eval/step": 0})  # bcp:0 was already logged
    assert logged[1] == (5, {})  # one eval step per training step
    assert logged[2] == (
        6,
        {"eval/bcp/n": 4, "eval/bcp/acc": 0.5, "eval/cb/n": 4, "eval/cb/kv_store/reward": 0.5, "eval/step": 10},
    )
    assert logged[3] == (7, {})
    assert sorted(json.loads((tmp_path / "eval_logged.json").read_text())) == ["bcp:0", "bcp:10", "cb:0", "cb:10"]
    assert json.loads((tmp_path / "eval_bcp_logged.json").read_text()) == [0, 10]
