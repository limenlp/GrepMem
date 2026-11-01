import asyncio
import json
from types import SimpleNamespace

import pytest

from agents import judge


def test_judge_uses_configured_api_root_and_model(tmp_path, monkeypatch):
    path = tmp_path / "judge.json"
    path.write_text(
        json.dumps({"base_url": "http://localhost:9000/v1", "api_key": "test-only", "model": "local-judge"})
    )
    monkeypatch.setenv("BCRL_JUDGE_CONFIG", str(path))
    calls = []
    monkeypatch.setattr("openai.OpenAI", lambda **kw: calls.append(kw) or object())
    judge.get_judge.cache_clear()
    try:
        client, model = judge.get_judge()
        assert client is not None and model == "local-judge"
        assert calls == [
            {"base_url": "http://localhost:9000/v1/", "api_key": "test-only", "timeout": 120, "max_retries": 5}
        ]
    finally:
        judge.get_judge.cache_clear()


@pytest.mark.parametrize(
    "content",
    [
        "{invalid",
        "[]",
        '{"api_key":"test-only"}',
        '{"base_url":"https://user:test-only@example.org/v1","api_key":"test-only","model":"judge"}',
    ],
)
def test_invalid_judge_config_does_not_expose_contents(content, tmp_path, monkeypatch):
    path = tmp_path / "judge.json"
    path.write_text(content)
    monkeypatch.setenv("BCRL_JUDGE_CONFIG", str(path))
    with pytest.raises(ValueError) as error:
        judge.load_config()
    assert "test-only" not in str(error.value)


def test_missing_judge_config_fails_before_evaluation(tmp_path, monkeypatch):
    import ama_eval
    import eval as research_eval

    monkeypatch.setenv("BCRL_JUDGE_CONFIG", str(tmp_path / "missing.json"))
    for evaluator in (research_eval, ama_eval):
        with pytest.raises(ValueError, match="Cannot read judge configuration"):
            asyncio.run(evaluator.main(SimpleNamespace(judge=True)))
