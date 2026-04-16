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
- `twitter-web-exporter-*.json`

The current sample in this folder contains:

- 604 suspected accounts in the list export
- 1,095 tweet rows in the exporter bundle

## Pipeline

The pipeline is split into four stages.

### 1. Discovery and normalization

- find the latest member CSV and exporter JSON in the input directory
- load member metadata
- flatten the exporter `tweets` table
- normalize tweet text, retweet targets, hashtags, mentions, timestamps, and source client

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

## Limits of the current approach

- This sample is one short capture window, not a long-running panel.
- Retweet timing here is only timing inside the captured sample, not the full platform.
- A high coordination score is a review signal, not proof of automation or payment.
- The current ML lane is unsupervised and review-oriented. It does not assign truth labels.
- The current code does not ingest follower graphs, liked posts, or external funding signals.
