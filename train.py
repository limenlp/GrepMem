"""rLLM unified-trainer entry point.

Select a strategy with AGENT, a reward with BCRL_REWARD=judge|f1, and data with TRAIN_DATASET.
Use scripts/train.sh for the reference configuration.
"""

import os

import hydra
from omegaconf import DictConfig
from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer

from agents.common import EVALUATORS
from agents.flows import FLOWS

FORWARD_ENV = ("PYTHONPATH", "AGENT", "RETRIEVAL_URL", "TENSORBOARD_DIR")
FORWARD_PREFIXES = ("BCRL_", "WANDB_")


def init_ray(config: DictConfig) -> None:
    """Start/attach Ray with our env vars forwarded to every worker (needed when the Ray
    cluster spans several nodes; rLLM only forwards VLLM_/HF_/CUDA_/... prefixes)."""
    import ray
    from rllm.trainer.ray_init_utils import get_ray_init_settings
    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    runtime_env = get_ppo_ray_runtime_env()
    runtime_env["env_vars"].update(
        {k: v for k, v in os.environ.items() if k in FORWARD_ENV or k.startswith(FORWARD_PREFIXES)}
    )
    ray.init(runtime_env=runtime_env, **get_ray_init_settings(config))


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    agent = os.environ.get("AGENT", "react")
    init_ray(config)
    # bcrl_mix: OpenResearcher single + 3-question episodes + ContextBench streams (data/prepare_data.py).
    train = DatasetRegistry.load_dataset(
        os.environ.get("TRAIN_DATASET", "bcrl_mix"), os.environ.get("TRAIN_SPLIT", "train")
    )
    # In-training validation is off by default; evaluate saved checkpoints separately.
    val = DatasetRegistry.load_dataset(
        os.environ.get("VAL_DATASET", "browsecomp_plus_multi"), os.environ.get("VAL_SPLIT", "test")
    )
    if train is None or val is None:
        raise RuntimeError("Datasets missing. Run: python data/prepare_data.py")
    AgentTrainer(
        backend="verl",
        agent_flow=FLOWS[agent],
        evaluator=EVALUATORS[os.environ.get("BCRL_REWARD", "f1")],
        config=config,
        train_dataset=train,
        val_dataset=val,
    ).train()


if __name__ == "__main__":
    main()
