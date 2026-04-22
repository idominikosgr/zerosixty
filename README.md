# zerosixty

`zerosixty` is a local analysis package for X/Twitter list exports.

The current goal is not to auto-label accounts as bots. The current goal is to:

- normalize raw list-member and tweet exports into one schema
- extract deterministic coordination signals
- run an unsupervised ML baseline on top of the deterministic feature matrix
- produce reviewable reports and flat files
- keep the feature matrix available for later supervised work

## Input files

The package expects raw exports in one directory:

- `twitter-ListMembers-*.csv`
- `twitter-ListMembers-*.json`
- `twitter-web-exporter-*.json`

The current sample in this folder contains:

- 604 suspected accounts in the list export
- 1,095 tweet rows in the exporter bundle

## Pipeline

The pipeline is split into four stages.

### 1. Discovery and normalization

- find the latest member export and exporter JSON in the input directory
- load member metadata
- flatten the exporter `tweets` table
- normalize tweet text, retweet targets, hashtags, mentions, timestamps, and source client
- optionally merge a manual handle supplement into the member set

### 2. Deterministic analysis

- account activity summaries
- retweet cascade summaries
- retweeter -> source-author edges
- hashtag and mention frequency tables
- account-pair overlap on shared retweeted originals
- simple coordination indicators such as:
  - retweet ratio
  - zero-original behavior
  - repeated first-retweeter behavior
  - concentration on a small set of source accounts
  - retweets that point back into the suspected-member list

### 3. Analyst outputs

The CLI writes:

- `normalized_tweets.csv`
- `account_summary.csv`
- `retweet_cascades.csv`
- `retweet_edges.csv`
- `hashtag_summary.csv`
- `mention_summary.csv`
- `account_overlap.csv`
- `network_nodes.csv`
- `network_components.csv`
- `ml_feature_matrix.csv`
- `ml_accounts.csv`
- `ml_clusters.csv`
- `summary.json`
- `report.md`

### 4. Parallel ML baseline

The ML lane runs in parallel with the deterministic one. It consumes the same
feature matrix and produces review-oriented outputs without replacing the
deterministic summaries.

Current ML behavior:

- standardize numeric account features
- cluster accounts with `KMeans`
- score outliers with `IsolationForest`
- project accounts into two dimensions with `PCA`
- write cluster- and account-level ML summaries for review

The current ML lane is unsupervised. It does not assign truth labels such as
`bot`, `organic`, or `paid`.

Recommended next steps after that:

1. Analyst-review a subset of accounts, cascades, and ML outliers.
2. Add labels such as `amplifier`, `organic`, `needs_review`, `source_hub`.
3. Compare those labels against cluster membership and anomaly rankings.
4. Move to supervised classification only after review labels are stable.

## CLI

Install and run from this directory:

```bash
uv sync
uv run zerosixty analyze --input-dir . --output-dir ./outputs/latest
```

Use a different source directory:

```bash
uv run zerosixty analyze --input-dir /absolute/path/to/export-folder
```

Use an explicit member export file and an extra-handle supplement:

```bash
uv run zerosixty analyze \
  --members-file /absolute/path/to/twitter-ListMembers-123.json \
  --exporter-json /absolute/path/to/twitter-web-exporter-456.json \
  --extra-members-file /absolute/path/to/manual_member_handles.txt
```

Build versioned clean batches from `datasets-raw`:

```bash
uv run zerosixty build-clean
```

Use explicit directories:

```bash
uv run zerosixty build-clean \
  --raw-dir /absolute/path/to/datasets-raw \
  --clean-dir /absolute/path/to/datasets-clean
```

Force a rebuild of existing clean batches:

```bash
uv run zerosixty build-clean --force
```

Run analysis directly from clean batches (default: auto-refresh clean batches and analyze latest):

```bash
uv run zerosixty analyze-clean
```

Analyze a specific batch id:

```bash
uv run zerosixty analyze-clean --batch-id batch_exporter-1776380968798__members-1776380931158
```

Disable auto-refresh and use existing clean batches only:

```bash
uv run zerosixty analyze-clean --no-auto-build
```

Write outputs into one fixed directory instead of per-batch subdirectories:

```bash
uv run zerosixty analyze-clean --no-batch-subdir --output-dir ./outputs/clean/latest
```

The clean flow writes:

- `datasets-clean/batch_*` directories with:
  - `members.csv` (normalized members)
  - `tweets.csv` (normalized tweets)
  - `manifest.json` (source fingerprints and counts)
- `datasets-clean/index.json` with all batch summaries and the latest batch id

Batch behavior:

- raw files stay unchanged in `datasets-raw`
- each exporter snapshot becomes one versioned clean batch
- a batch is skipped on rerun when source fingerprints are unchanged
- when a new raw exporter snapshot is added, only the new clean batch is built
- `analyze-clean` can auto-run `build-clean`, resolve the latest batch, and run analysis in one command

## Limits of the current approach

- This sample is one short capture window, not a long-running panel.
- Retweet timing here is only timing inside the captured sample, not the full platform.
- A high coordination score is a review signal, not proof of automation or payment.
- The current ML lane is unsupervised and review-oriented. It does not assign truth labels.
- The current code does not ingest follower graphs, liked posts, or external funding signals.
