"""Exercise release launchers without downloading models or starting GPU work."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_runtime(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls.jsonl"
    python = bindir / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:]}) + '\\n')\n"
        "marker = Path(os.environ['CALLS'] + '.retry')\n"
        "if os.environ.get('RETRY_ONCE') and not marker.exists():\n"
        "    marker.touch()\n"
        "    sys.exit(2)\n"
    )
    python.chmod(0o755)
    curl = bindir / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 0\n")
    curl.chmod(0o755)
    return {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "CALLS": str(calls)}, calls


def test_evaluate_dispatch_and_retry(tmp_path, fake_runtime):
    env, calls = fake_runtime
    env.update(OUT=str(tmp_path / "results"), BENCHMARKS="bcpm cb ama amafile", RETRY_ONCE="1", STEP="80")
    subprocess.run(["bash", str(ROOT / "scripts/evaluate.sh")], cwd=tmp_path, env=env, check=True)
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert [r["args"][0] for r in records] == ["eval.py", "eval.py", "cb_eval.py", "ama_eval.py", "ama_eval.py"]
    assert all(r["cwd"] == str(ROOT) for r in records)
    assert records[0]["args"] == records[1]["args"]
    for record in records:
        args = record["args"]
        assert args[args.index("--out") + 1].endswith("_step80.jsonl")
    assert records[-2]["args"][records[-2]["args"].index("--protocol") + 1] == "stream"
    assert records[-1]["args"][records[-1]["args"].index("--protocol") + 1] == "file"


def test_train_defaults_and_private_settings(tmp_path, fake_runtime):
    env, calls = fake_runtime
    env.update(
        OUT=str(tmp_path / "train"),
        AGENT="summary_grep",
        BCRL_JUDGE_CONFIG="/private/judge.json",
        BCRL_CONTEXT_BUDGET="32768",
        BCRL_API_KEY="test-only-do-not-record",
    )
    subprocess.run(["bash", str(ROOT / "scripts/train.sh")], cwd=tmp_path, env=env, check=True)
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    args = records[-1]["args"]
    assert "train.py" in args and "rllm.trainer.total_batches=81" in args
    assert "rllm.trainer.logger=['console']" in args
    settings = (tmp_path / "train/run_env.sh").read_text()
    assert "BCRL_CONTEXT_BUDGET=32768" in settings
    assert "BCRL_JUDGE_CONFIG=/private/judge.json" in settings
    assert "test-only-do-not-record" not in settings
