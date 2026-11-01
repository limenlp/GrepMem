"""Fix verl's HF export of Qwen3.5 so vLLM (and transformers) load the vision tower.

verl saves the vision tower as ``model.language_model.visual.*``; the base model and vLLM use
``model.visual.*``. vLLM then refuses the checkpoint ("Following weights were not initialized from
checkpoint: {'visual.blocks...'}"). Renames those tensors in place; idempotent.

  python fix_hf_ckpt.py <checkpoint>/actor/huggingface
"""

import glob
import json
import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

BAD, GOOD = "model.language_model.visual.", "model.visual."


def rename(k: str) -> str:
    return GOOD + k[len(BAD) :] if k.startswith(BAD) else k


def main(d: str) -> None:
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            keys = list(h.keys())
            n = sum(k.startswith(BAD) for k in keys)
            if not n:
                continue
            tensors = {rename(k): h.get_tensor(k) for k in keys}
            meta = h.metadata() or {"format": "pt"}
        save_file(tensors, f + ".tmp", metadata=meta)
        os.replace(f + ".tmp", f)
        print(f"{f}: renamed {n} vision tensors to {GOOD}*")
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        j = json.load(open(idx))
        if any(k.startswith(BAD) for k in j.get("weight_map", {})):
            j["weight_map"] = {rename(k): v for k, v in j["weight_map"].items()}
            json.dump(j, open(idx, "w"), indent=2)


if __name__ == "__main__":
    main(sys.argv[1])
