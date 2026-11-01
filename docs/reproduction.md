# Reproducing GrepMem

The commands below separate the evaluation client, model server, corpus server, and trainer.
They may run on different machines. Install the repository and the relevant dependencies on each.
Use the same code revision and settings for all five context management strategies.

## Environment

Start with the CPU client installation in the [README](../README.md#quick-start).
The lightweight installation is sufficient for dataset preparation and evaluation against a running
OpenAI-compatible model server; it is not a training environment.

| Component | Reference |
|---|---|
| Policy | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |
| rLLM | `4e828c81680e5a85b638ceedeaf09d1e90109ca8` |
| ContextBench | `fb533c18c78a82d3f1b2047a9c1700735cf829ae` |
| Serving | vLLM 0.22.1; Qwen reasoning/tool parsers; Triton GDN prefill |
| Training backend | verl 0.8.0, as specified by the pinned rLLM revision |
| Retrieval encoder | [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |

These upstream source revisions are pinned. The Python dependency ranges are not a complete environment
lock. GPU training additionally depends on compatible CUDA, PyTorch, and attention kernels.
The original experiments used H100 GPUs. CPU tests do not establish GPU-stack compatibility.

## Data

```bash
bash scripts/download_data.sh
python data/prepare_data.py
```

Downloads use the standard Hugging Face cache (`HF_HOME` or `HF_HUB_CACHE` can override it).
Data preparation registers datasets under `~/.rllm`; it prints only aggregate counts.
Do not publish the registry, per-question outputs, or decrypted benchmark data.
Authenticate with `hf auth login` if a source requires access.

The training mixture has 4,000 episodes: 2,800 single
[OpenResearcher](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) questions,
600 three-question sessions, and 600 synthetic ContextBench streams.
Questions requiring attachments, non-English CJK text, and answers longer than ten words are filtered out.
The source is shuffled with seed 0, with 500 questions held out before constructing training rows.
Training triples use seed 2; a question may appear both alone and in a triple.
Synthetic streams use seeds starting at 1000 and 12 queries per stream.

BCP-Multi shuffles all 830 BrowseComp-Plus questions with seed 1, forms 276 non-overlapping triples,
and discards the last two questions. All three questions are given upfront and share one context
and a 60-turn budget; the agent returns three numbered answers.
The script also registers the original single-question dataset for compatibility, but it is not
part of the paper's five-task results.

ContextBench evaluation inputs are generated on demand: 11 family/length combinations, each with
seeds 42, 43, 44, and 45, and 24 queries per stream.
AMA-Bench uses the original `test/open_end_qa_set.jsonl`; no copy is distributed here.

## Judge configuration

Research and AMA-Bench evaluations require a judge exposing the standard OpenAI-compatible chat
completions API. Use an API root ending in `/v1` when required by your server; it is passed unchanged
apart from its trailing slash. The model must support `max_completion_tokens`.

Create a **private file outside the repository**, for example `~/.config/grepmem/judge.json`:

```json
{
  "base_url": "https://api.openai.com/v1",
  "api_key": "REPLACE_WITH_YOUR_KEY",
  "model": "REPLACE_WITH_YOUR_JUDGE_MODEL"
}
```

```bash
chmod 600 ~/.config/grepmem/judge.json
export BCRL_JUDGE_CONFIG="$HOME/.config/grepmem/judge.json"
```

For a local server that requires no authentication, use a nonempty placeholder key.
The file is parsed as JSON, never sourced as shell code. On a multi-machine trainer, securely place it
at the same path on every worker. Only the path is forwarded to workers; credentials are not written
to run settings or experiment tracking.

The paper used `gpt-5.6-sol` as the judge. Choose that model only if your provider exposes it.
Using a different judge is supported, but is **not an exact reproduction** of the reported scores.
There is deliberately no default judge model.

Missing or malformed judge configuration stops evaluation before model calls. Runtime judge failures
retain the original scoring behavior: research uses an F1-threshold fallback, and AMA assigns zero
to failed judgments. Both record `judge_failed`; do not interpret a run with such failures as a clean
paper reproduction. `eval.py --no-judge` explicitly reports F1/EM only.
ContextBench always uses its own programmatic grader and needs no judge.

## Document retrieval

Only research tasks require this server. It searches the external document corpus, not the
agent's interaction history.

```bash
python -m pip install -r requirements-retrieval.txt
RETRIEVAL_DATA=1 bash scripts/download_data.sh
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/retrieval.sh
```

Both indexes use the precomputed Qwen3-Embedding-8B representations from the dataset publishers.
The OpenResearcher matrix alone occupies roughly 122 GB in bf16 and is sharded over two GPUs by default;
the encoder and BrowseComp-Plus index use the first GPU. Loading 14.9 million document texts also needs
substantial host RAM and disk space. Reserve these GPUs separately from the policy/trainer.
The source pickle indexes must come from trusted publishers.

For **evaluation only**, download the BCP corpus/index and encoder and omit the OpenResearcher index:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/retrieval.sh --corpora bcp
```

Check `curl --fail http://localhost:8100/health`.
Set `RETRIEVAL_URL=http://<retrieval-host>:8100` on each client/trainer if it runs elsewhere.
The server binds to localhost by default; use `--host` to bind to a trusted network interface.
It has no authentication layer and should not be exposed directly to the internet.

## Training

Create a separate Python 3.12 GPU environment and follow the
[rLLM installation instructions](https://github.com/rllm-org/rllm/tree/4e828c81680e5a85b638ceedeaf09d1e90109ca8).
For this code revision, install the full verl backend, rather than the lightweight client alone:

```bash
# From the GrepMem repository, in the GPU environment:
git clone https://github.com/rllm-org/rllm.git .cache/rllm
git -C .cache/rllm checkout 4e828c81680e5a85b638ceedeaf09d1e90109ca8
git -C .cache/rllm apply "$PWD/patches/rllm-gateway-renderers.patch"
python -m pip install -e .cache/rllm/rllm-model-gateway
python -m pip install -e '.cache/rllm[verl]'
python -m pip install -r requirements.txt
python -m pip install "fla-core==0.5.2" "cupy-cuda13x==13.6.0" fastrlock
```

The kernel packages above target CUDA 13. Use a matching system CUDA toolkit (`CUDA_HOME`) and consult
the upstream GPU installation instructions if your platform differs. Native attention packages may
need to be built against the installed PyTorch. Do not install GPU dependencies into the CPU-only CI environment.
The gateway patch handles the renderer API used by this revision.

Prepare datasets, start retrieval, and configure the judge before training. A one-machine
smoke run uses four training GPUs and four rollout GPUs:

```bash
SMOKE=1 AGENT=summary_grep bash scripts/train.sh
```

The paper's GrepMem/SUPO setup uses eight training GPUs and sixteen rollout GPUs on a three-machine
Ray cluster. Start a Ray head and two workers on a trusted network:

```bash
# Head:
ray start --head --port=6379 --num-gpus=8
# On each worker, replace <head-ip>:
ray start --address=<head-ip>:6379 --num-gpus=8
```

Then on the head, with the same checkout path, Python environment, dataset registry, judge file path,
and shared output directory available to all workers:

```bash
AGENT=summary_grep RUN=qwen3.5-9b-summary_grep-mix \
  TRAIN_GPUS=8 ROLLOUT_NNODES=2 ROLLOUT_GPUS=8 N_PARALLEL=512 \
  STEPS=80 EPOCHS=10 bash scripts/train.sh
```

Repeat with `AGENT=summary`, `mask`, `memgpt`, or `react` and a distinct `RUN`.
ReAct used eight rollout GPUs instead of sixteen (`ROLLOUT_NNODES=1`).
The default single-machine layout is a convenience setting, not the paper's multi-machine layout.

| Setting | Paper value |
|---|---|
| Prompt groups per update / rollouts per group | 8 / 8 |
| Optimizer learning rate / weight decay | 1e-6 / 0.1 |
| GRPO clipping interval | 0.2 / 0.28 |
| KL coefficient / token importance-ratio cap | 0.01 / 2 |
| Async staleness threshold | 0.5 |
| Sampling temperature / top-p | 0.7 / 0.95 |
| Context window / compaction threshold | 32,768 / 28,672 tokens |
| Generation limit | 4,096 tokens per call |
| Checkpoint interval | 10 updates |

Uniform-reward groups are filtered out. Each LLM call is one rLLM step; compaction calls remain in
the training trajectory. The async trainer receives `STEPS + 1` batches to produce `STEPS` updates.

Outputs default to `outputs/$RUN`: checkpoints, training logs, episode traces, and `run_env.sh`.
Override `OUT` with an absolute path when needed. Automatic resume uses the latest local checkpoint;
`RESUME_FROM` can name an external `global_step_N` directory.
Only two checkpoints are kept by default; raise `KEEP_CKPTS` if you need the full curve.
Logging is console-only by default. To enable W&B, authenticate separately and set
`LOGGER="['console','wandb']"`, `WANDB_ENTITY`, and optionally `WANDB_PROJECT`.
Episode traces contain dataset content: keep them private.

## Evaluate the full task suite

Serve the base model using [scripts/serve.sh](../scripts/serve.sh).
To serve a saved checkpoint, first copy its Hugging Face export and fix the exported vision-tower names
in that copy; never rewrite the original training checkpoint:

```bash
mkdir -p outputs/eval-model
cp -a outputs/qwen3.5-9b-summary_grep-mix/checkpoints/global_step_80/actor/huggingface/. outputs/eval-model/
python fix_hf_ckpt.py outputs/eval-model
GPUS=0,1,2,3,4,5,6,7 bash scripts/serve.sh outputs/eval-model
```

The served model name stays `Qwen/Qwen3.5-9B` unless `SERVED_MODEL` is overridden.
Once the server is healthy, run evaluations in another terminal:

```bash
AGENT=summary_grep STEP=80 \
  RUN_ENV="$PWD/outputs/qwen3.5-9b-summary_grep-mix/run_env.sh" \
  OUT="$PWD/outputs/qwen3.5-9b-summary_grep-mix" \
  bash scripts/evaluate.sh
```

Only source a `RUN_ENV` file that you trust. Omit it for the base model and use `STEP=0`.
`BENCHMARKS="bcpm cb ama amafile"` is the default. The script evaluates sequentially against the existing
server and retries incomplete outputs up to three times; it does not start or stop GPU services.
Each evaluator also supports direct use and `--help`.

AMA **file** uses all 208 trajectories and answers 12 questions together with 60 task turns.
The trajectory stays outside the answering context and is searched with `grep_trajectory`.
AMA **stream** uses the 189 trajectories with at most 131,072 tokens, feeds the trajectory first,
then answers each question on a separate copy of the final memory state.
BCP-Multi has 276 sessions; ContextBench has 44 streams.

The paper averages three after-RL samples per strategy (two for ReAct); one invocation is one sample.
Use a **new output directory for each independent sample**; reusing a path resumes existing results.
All five after-RL models use the selected checkpoint after 80 updates. Selection inspected evaluation
curves; the CPU tests and quick start neither reproduce that selection nor supply trained weights.

For cost comparisons, use the same GPU count, concurrency, and server load. The original comparable
cost runs used eight policy GPUs, concurrency 64 for BCP-Multi, and 256 for the other tasks
(`CONCURRENCY` overrides the wrapper's convenience defaults).
Outputs record prompt, completion, cached tokens, and elapsed time. Do not equate token cost with
runtime, and inspect `judge_failed` and item counts before comparing scores.
