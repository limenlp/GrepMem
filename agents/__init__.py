import os

if os.environ.get("BCRL_EVAL_DIR"):  # set by train.sh: log offline evals into the training wandb run
    from .eval_bridge import install

    install()
