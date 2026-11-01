# Method and configuration

## Two kinds of memory

GrepMem keeps a bounded working context and an append-only interaction history.
At the compaction threshold, the policy writes a summary and continues with that summary
in place of the older context. The original messages and tool observations remain in the log.
After the first compaction, the policy can call `grep_history` to recover specific details.
The RL objective trains the policy's task actions, summaries, and recall decisions together.

The memory log needs no learned index. This is separate from the dense retrieval used to research
the external OpenResearcher/BrowseComp-Plus document corpora.

`grep_history(pattern, max_matches, offset, context_lines, count)` searches case-insensitively,
returns matching lines grouped by their source turn/tool, and supports paging.
The count option counts matching **lines**, not occurrences within a line.
Earlier `grep_history` outputs are excluded from subsequent searches to avoid duplicate recalled records.
Long lines and result payloads are clipped, and regular-expression matching has a timeout.
The stored history is lossless; the retrieved view is bounded.

In AMA-Bench's file protocol, `grep_trajectory` uses the same search implementation but a different
input: the supplied trajectory's action/observation steps. Every strategy receives that tool.
`grep_history` searches the answering agent's own growing conversation and is specific to GrepMem.

## Baselines

All five strategies share prompts, sampling settings, task interfaces, and the same RL recipe:

- **ReAct:** keep the conversation until its context budget is exhausted.
- **SUPO:** the same policy produces summaries and receives RL credit for those calls.
- **Masking:** replace old tool observations (stream tasks: old streamed messages) with placeholders;
  retain the agent's replies when possible, then trim further if needed to fit the budget.
- **MemGPT:** working context, a BM25 archival store, and substring search over original messages.
  Warn at 70% of the compaction threshold; flush to half the threshold and recursively summarize
  evicted messages. The original messages remain searchable.
- **GrepMem:** SUPO plus the append-only log and `grep_history`.

See the [README's source links](../README.md#reproduce-the-experiments) for the baseline papers.
Our MemGPT implementation is not an embedding-retrieval experiment.

## Budgets

Research episodes allow 30 task turns, or 60 shared turns for a three-question session.
Up to ten grep-only replies (MemGPT: memory-tool-only replies) are exempt from the task-turn cap.
All model calls, including summaries and the forced final answer, contribute to token accounting.
If the agent runs out of task turns without an answer, it receives one final call without tools.
ReAct can terminate earlier at its context limit.

Stream tasks instead use their streaming protocol: at most eight model calls per input operation,
with a five-reply no-progress limit for ContextBench. They do not use the research turn budget.
The AMA file protocol has 60 task turns for all 12 questions.

## Main settings

The `BCRL_` prefix is retained for compatibility with training and evaluation settings.
Set variables before starting Python; most settings are read at import time.

| Variable | Default | Purpose |
|---|---|---|
| `RETRIEVAL_URL` | `http://localhost:8100` | External document corpus server |
| `BCRL_CONTEXT_BUDGET` | `32768` | Context budget |
| `BCRL_COMPACT_AT` | `28672` | Compaction threshold |
| `BCRL_TURN_TOKENS` | `4096` | Maximum generation per call |
| `BCRL_MAX_TURNS` | `30` | Single-question research task turns |
| `BCRL_MULTI_TURNS_PER_Q` | `20` | Shared research budget per bundled question |
| `BCRL_GREP_FREE_TURNS` | `10` | Exempt grep-/memory-only replies |
| `BCRL_GREP_CHARS` | `6000` | Grep result payload budget |
| `BCRL_GREP_MAX_MATCHES` | `30` | Maximum matches in one page |
| `BCRL_GREP_LINE_CHARS` | `300` | Per-line display limit |
| `BCRL_GREP_TIMEOUT` | `10` | Regex time limit in seconds |
| `BCRL_DOC_TOKENS` | `8192` | Retrieved document limit |
| `BCRL_LLM_TIMEOUT` | `3600` | Model-call timeout; wrapper uses `900` for evaluation |
| `BCRL_JUDGE_CONFIG` | `~/.config/grepmem/judge.json` | Private judge configuration path |
| `BCRL_CB_TRAIN_DIR` | `.cache/grepmem/cb_train` | Generated training-stream cache |
| `AMA_FILE_TURNS` | `60` | AMA file task turns |
| `AMA_CHUNK_CHARS` | `12000` | Target AMA stream chunk size |
| `CB_SEEDS` | `42,43,44,45` | ContextBench evaluation seeds |

Serving and training controls are described in the [reproduction guide](reproduction.md).
Generated outputs, models, caches, and private configuration files are excluded from version control.
