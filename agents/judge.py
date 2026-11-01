"""Provider-neutral configuration for an OpenAI-compatible judge."""

import functools
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def load_config() -> dict[str, str]:
    path = Path(os.environ.get("BCRL_JUDGE_CONFIG", "~/.config/grepmem/judge.json")).expanduser()
    try:
        with path.open() as handle:
            config = json.load(handle)
    except (OSError, ValueError):
        raise ValueError("Cannot read judge configuration; see docs/reproduction.md.") from None
    required = ("base_url", "api_key", "model")
    if not isinstance(config, dict) or any(
        not isinstance(config.get(k), str) or not config[k].strip() for k in required
    ):
        raise ValueError("Judge configuration requires nonempty base_url, api_key, and model strings.")
    url = urlsplit(config["base_url"])
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Judge base_url must be an HTTP(S) API root without credentials, query, or fragment.")
    return {k: config[k] for k in required}


@functools.lru_cache(maxsize=1)
def get_judge():
    from openai import OpenAI

    config = load_config()
    return OpenAI(
        base_url=config["base_url"].rstrip("/") + "/",
        api_key=config["api_key"],
        timeout=120,
        max_retries=5,
    ), config["model"]
