"""Log offline evals into the training run's wandb run.

scripts/evaluate.sh writes ``<run dir>/eval_<kind>_step<N>.summary.json`` for the evaluation tasks.
The trainer's own logger picks each new file
up at its next training log and adds its metrics as ``eval/<kind>/*`` with ``eval/step = N`` (the checkpoint
step, used as the x-axis of the eval charts). All kinds pending for the same N go into one log row.
A second wandb writer is avoided on purpose: wandb's shared mode renumbers ``_step`` across writers,
which would shift the training curves.

Installed by ``agents/__init__.py`` when ``BCRL_EVAL_DIR`` is set (train.sh sets it to the run dir).
Already-logged evals are recorded in ``<run dir>/eval_logged.json`` (``["bcp:0", "cb:0", ...]``; the older
``eval_bcp_logged.json``, a list of BCP steps, is still read and kept up to date) so a resumed run does not
repeat them.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re

logger = logging.getLogger(__name__)
SUMMARY_RE = re.compile(r"eval_([a-z]+)_step(\d+)\.summary\.json$")


def _flatten(kind: str, summary: dict) -> dict:
    out = {}
    for k, v in summary.items():
        if isinstance(v, dict):  # termination counts -> fractions
            n = summary.get("n") or 1
            out.update({f"eval/{kind}/{k}/{t}": c / n for t, c in v.items() if isinstance(c, (int, float))})
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"eval/{kind}/{k}"] = v
    return out


def _write(path: str, obj) -> None:
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh)
    os.replace(path + ".tmp", path)


def install() -> None:
    try:
        from rllm.utils import tracking
    except Exception:  # noqa: BLE001
        return
    if getattr(tracking.Tracking.log, "_bcrl_eval", False):
        return
    eval_dir = os.environ["BCRL_EVAL_DIR"]
    state_file = os.path.join(eval_dir, "eval_logged.json")
    bcp_state_file = os.path.join(eval_dir, "eval_bcp_logged.json")
    orig = tracking.Tracking.log
    state = {"logged": None, "defined": False, "last_step": None}

    def load_logged() -> set:
        logged = set(json.load(open(state_file))) if os.path.exists(state_file) else set()
        if os.path.exists(bcp_state_file):
            logged |= {f"bcp:{n}" for n in json.load(open(bcp_state_file))}
        return logged

    def log(self, data, step, *args, **kwargs):
        # At most one eval step per training step: wandb merges logs at the same step into one row.
        if step != state["last_step"]:
            try:
                if state["logged"] is None:
                    state["logged"] = load_logged()
                pending = sorted(
                    (int(m.group(2)), m.group(1), f)
                    for f in glob.glob(os.path.join(eval_dir, "eval_*_step*.summary.json"))
                    if (m := SUMMARY_RE.search(f)) and f"{m.group(1)}:{m.group(2)}" not in state["logged"]
                )
                if pending:
                    n = pending[0][0]
                    metrics, kinds = {}, []
                    for n2, kind, f in pending:
                        if n2 == n:
                            metrics.update(_flatten(kind, json.load(open(f))))
                            kinds.append(kind)
                    wb = getattr(self, "logger", {}).get("wandb")
                    if wb is not None and not state["defined"]:
                        wb.define_metric("eval/step")
                        wb.define_metric("eval/*", step_metric="eval/step")
                        state["defined"] = True
                    data = {**data, **metrics, "eval/step": n}
                    state["logged"] |= {f"{k}:{n}" for k in kinds}
                    state["last_step"] = step
                    _write(state_file, sorted(state["logged"]))
                    _write(
                        bcp_state_file, sorted(int(x.split(":")[1]) for x in state["logged"] if x.startswith("bcp:"))
                    )
                    logger.info("logged eval step %d (%s) at training step %s", n, ",".join(kinds), step)
            except Exception as e:  # noqa: BLE001
                logger.warning("eval bridge: %s", e)
        return orig(self, data, step, *args, **kwargs)

    log._bcrl_eval = True
    tracking.Tracking.log = log
