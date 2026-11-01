#!/usr/bin/env bash
# GRPO training. See docs/reproduction.md for the paper settings and multi-GPU layout.
# AGENT=summary_grep bash scripts/train.sh [extra Hydra overrides]
set -euo pipefail
cd "$(dirname "$0")/.."

AGENT=${AGENT:-summary_grep}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
RUN=${RUN:-qwen3.5-9b-${AGENT}}
OUT=${OUT:-$PWD/outputs/$RUN}
N_ROLLOUT=${N_ROLLOUT:-8}
PROMPTS_PER_STEP=${PROMPTS_PER_STEP:-8}
STEPS=${STEPS:-80}
if [[ "${SMOKE:-0}" == 1 ]]; then
  RUN=smoke-${AGENT}; OUT=$PWD/outputs/$RUN; N_ROLLOUT=2; PROMPTS_PER_STEP=2; STEPS=2; SAVE_FREQ=2
  export BCRL_MAX_TURNS=${BCRL_MAX_TURNS:-6}
fi

export AGENT PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RETRIEVAL_URL=${RETRIEVAL_URL:-http://localhost:8100}
export TENSORBOARD_DIR=$OUT/tensorboard
export WANDB_DIR=${WANDB_DIR:-$OUT/wandb} WANDB_NAME=$RUN
# The trainer logs offline evaluation summaries from this directory into the same run.
export WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN//./_}} WANDB_RESUME=allow BCRL_EVAL_DIR=$OUT
# Workers read credentials from a private file, not from logged trainer settings.
export BCRL_REWARD=${REWARD:-${BCRL_REWARD:-judge}}
if [[ "$BCRL_REWARD" == judge ]]; then
  python3 -c 'from agents.judge import load_config; load_config()'
fi
export VLLM_USE_V1=1 RAY_DEDUP_LOGS=0
# A rollout worker that stalls longer than this (default 300 s) kills its vLLM engine for the rest of the run.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1200}
# fla's TileLang GDN backward: pip nvcc 13.4 mismatches pip CCCL 13.3 headers; use system CUDA 13.0.
# (fla's Triton fallback is refused on Hopper with triton<3.7.1.)
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
# Reduce fragmentation OOMs in the actor backward (verl's NCCL weight sync is compatible).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
mkdir -p "$OUT" "$WANDB_DIR"
# Save harness settings for scripts/evaluate.sh, excluding any credential-like variables.
{ echo "export AGENT=$AGENT RUN=$RUN WANDB_RUN_ID=$WANDB_RUN_ID"
  env | grep '^BCRL_' | grep -Ev '^BCRL_EVAL_DIR=|^BCRL_.*(KEY|TOKEN|SECRET|PASSWORD)=' | sort | while IFS='=' read -r k v; do printf 'export %s=%q\n' "$k" "$v"; done
} > "$OUT/run_env.sh"

curl -sf "$RETRIEVAL_URL/health" > /dev/null || { echo "retrieval server not reachable (see scripts/retrieval.sh)"; exit 1; }

# RESUME_FROM=<dir>/global_step_N starts from a checkpoint outside $OUT (e.g. after moving a run);
# once $OUT has its own checkpoint, resume_mode=auto takes over.
RESUME=(trainer.resume_mode=auto)
# FlashInfer's SM90 GDN prefill kernel (vLLM's default on H100, flashinfer < 0.6.15.post1) deadlocks under
# prefix caching + chunked prefill + high concurrency (flashinfer-ai/flashinfer#3986): a worker hangs in
# sample_tokens and its engine dies every few hours. Use the Triton/FLA kernel instead.
VLLM_EXTRA=(+actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend=${VLLM_GDN_BACKEND:-triton})
# verl drops `false` booleans, so async scheduling is disabled with no_async_scheduling=true.
[[ "${VLLM_ASYNC_SCHED:-1}" == 1 ]] || VLLM_EXTRA+=(+actor_rollout_ref.rollout.engine_kwargs.vllm.no_async_scheduling=true)
if [[ -n "${RESUME_FROM:-}" && ! -f "$OUT/checkpoints/latest_checkpointed_iteration.txt" ]]; then
  RESUME=(trainer.resume_mode=resume_path "trainer.resume_from_path=$RESUME_FROM")
fi

python3 -u train.py \
  rllm/backend=verl \
  data.train_batch_size=1 \
  data.max_prompt_length=${MAX_PROMPT_LEN:-32768} \
  data.max_response_length=45056 \
  actor_rollout_ref.model.path=$MODEL \
  +model.name=$MODEL \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=True \
  actor_rollout_ref.model.enable_activation_offload=${ACT_OFFLOAD:-True} \
  actor_rollout_ref.model.fused_kernel_options.impl_backend=triton \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKENS_PER_GPU:-40960} \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP:-1} \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model']" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
  actor_rollout_ref.rollout.max_model_len=65536 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=qwen3_coder \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=qwen3 \
  rllm.gateway.cumulative_token_mode=true \
  rllm.gateway.renderer_family=auto \
  rllm.rollout.n=$N_ROLLOUT \
  rllm.rollout.n_val=1 \
  rllm.rollout.train.temperature=0.7 \
  rllm.rollout.train.top_p=0.95 \
  rllm.rollout.train.max_tokens=4096 \
  rllm.rollout.val.temperature=0.7 \
  rllm.rollout.val.top_p=0.95 \
  rllm.rollout.val.max_tokens=4096 \
  rllm.algorithm.adv_estimator=grpo \
  rllm.algorithm.norm_adv_by_std_in_grpo=true \
  rllm.algorithm.kl_beta=0.01 \
  rllm.algorithm.rollout_correction.bypass_mode=false \
  rllm.algorithm.rollout_correction.tis_mode=token \
  rllm.algorithm.rollout_correction.tis_cap=2.0 \
  rllm.stepwise_advantage.enable=true \
  rllm.stepwise_advantage.mode=broadcast \
  rllm.rejection_sample.enable=true \
  rllm.rejection_sample.filter_uniform_groups=true \
  rllm.compact_filtering.enable=false \
  rllm.async_training.enable=true \
  rllm.async_training.mini_batch_size=$PROMPTS_PER_STEP \
  rllm.async_training.staleness_threshold=${STALENESS:-0.5} \
  rllm.async_training.trigger_parameter_sync_step=1 \
  rllm.async_training.partial_rollout=true \
  rllm.workflow.n_parallel_tasks=${N_PARALLEL:-128} \
  rllm.workflow.retry_limit=1 \
  rllm.workflow.raise_on_error=false \
  rllm.trainer.total_batches=$((STEPS + 1)) \
  rllm.trainer.total_epochs=${EPOCHS:-10} \
  rllm.trainer.logger="${LOGGER:-['console']}" \
  rllm.trainer.project_name=${WANDB_PROJECT:-grepmem} \
  rllm.trainer.experiment_name=$RUN \
  rllm.trainer.val_before_train=${VAL_BEFORE_TRAIN:-false} \
  rllm.trainer.test_freq=${TEST_FREQ:--1} \
  rllm.trainer.save_freq=${SAVE_FREQ:-10} \
  rllm.episode_logging.log_episodes=true \
  rllm.episode_logging.episode_log_dir=$OUT/episodes \
  trainer.critic_warmup=0 \
  trainer.nnodes=${TRAIN_NNODES:-1} \
  trainer.n_gpus_per_node=${TRAIN_GPUS:-4} \
  rollout.nnodes=${ROLLOUT_NNODES:-1} \
  rollout.n_gpus_per_node=${ROLLOUT_GPUS:-4} \
  trainer.default_local_dir=$OUT/checkpoints \
  "${RESUME[@]}" \
  "${VLLM_EXTRA[@]}" \
  trainer.max_actor_ckpt_to_keep=${KEEP_CKPTS:-2} \
  hydra.run.dir=$OUT/hydra \
  "$@" 2>&1 | tee -a "$OUT/train.log"
