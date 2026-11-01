"""Training-mix builders in data/prepare_data.py.

python3 -m pytest tests/test_data.py -q      (needs rllm and pandas installed)
"""

from __future__ import annotations

import json

from data.prepare_data import CB_TRAIN_LEVELS, bundle, cb_rows


def test_bundle_groups_disjoint_questions():
    rows = [
        {"id": f"q{i}", "question": f"Q{i}?", "answer": f"A{i}", "corpus": "bcp", "kind": "single", "extra": "{}"}
        for i in range(7)
    ]
    out = bundle(rows, 3, seed=1)
    assert len(out) == 2 and all(r["kind"] == "multi" and r["corpus"] == "bcp" for r in out)
    ids = [i for r in out for i in json.loads(r["extra"])["ids"]]
    assert len(set(ids)) == 6
    x = json.loads(out[0]["extra"])
    assert [f"A{q[1:]}" for q in x["ids"]] == x["answers"] and all(q in out[0]["question"] for q in x["questions"])
    assert out == bundle(rows, 3, seed=1)  # deterministic


def test_cb_rows_fresh_seeds_and_both_families():
    rows = cb_rows(8, 1000)
    extras = [json.loads(r["extra"]) for r in rows]
    assert {x["family"] for x in extras} == set(CB_TRAIN_LEVELS)
    seeds = [x["knobs"]["seed"] for x in extras]
    assert len(set(seeds)) == 8 and min(seeds) >= 1000 and len({r["id"] for r in rows}) == 8
    assert set(rows[0]) == {"id", "question", "answer", "corpus", "kind", "extra"}
