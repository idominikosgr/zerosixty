# Next patterns and visualizations

This is the next round of deterministic analysis worth adding as the dataset grows.

## 1. Network views

### Shared-retweet similarity graph

- nodes: accounts in the captured set
- edges: two accounts retweeted the same original
- edge weights:
  - shared retweeted originals
  - shared retweets within 15 minutes
  - shared retweets within 60 minutes
- useful outputs:
  - connected components
  - weighted degree
  - k-core or minimum-degree subgraphs
  - bridge accounts between components

This is the first graph to inspect when looking for amplification clusters.

### Source-amplifier bipartite graph

- left side: retweeters
- right side: original authors being amplified
- directed edges: retweeter -> source author
- weights:
  - retweet count
  - proportion of retweeter activity aimed at that source

This separates source hubs from pure amplifiers.

### Hashtag/account bipartite graph

- account -> hashtag usage
- useful for spotting narrative synchronization even when accounts do not retweet the exact same post

### Mention-target graph

- account -> mention target
- useful for repeated targeting of journalists, politicians, or media accounts

## 2. Time patterns

### Lead-lag matrix

- for each account pair, compute who tends to retweet first
- store counts like:
  - `A before B`
  - `B before A`
  - median lag

This is better than a flat first-retweeter count when you have multiple days of data.

### Burst windows

- split time into fixed windows such as 5, 15, and 60 minutes
- measure how many accounts activate in the same window on:
  - the same original tweet
  - the same hashtag
  - the same mention target

### Recurrent cohorts

- identify the same group of accounts appearing together across different cascades and days
- score each cohort by:
  - repeat frequency
  - time tightness
  - source concentration

## 3. Content coordination

### Canonical text reuse

- normalize original text harder than v1
- group near-duplicates, not just exact duplicates
- useful for:
  - repeated talking points
  - lightly edited quote posts
  - copy-paste campaign language

### URL and domain coordination

- extract full URLs and normalized domains
- measure:
  - repeated linking to the same domain
  - repeated linking to the same URL within short windows
  - domain concentration by account or component

### Media reuse

- if media identifiers are available, track repeated media use
- same image or video reused across multiple accounts is a stronger signal than hashtag overlap

### Quote-retweet framing

- compare a retweeter's added text against the original
- capture:
  - sentiment shift
  - repeated framing language
  - repeated insults or praise terms

## 4. Account behavior patterns

### Role classification

The deterministic layer should eventually assign review-oriented roles such as:

- `source_hub`
- `amplifier`
- `bridge`
- `burst_participant`
- `isolated_repeater`
- `mixed_behavior`

These are not accusations. They are navigation labels for analysts.

### In-list vs out-of-list dynamics

- what share of activity amplifies accounts already in the suspected list
- what out-of-list sources repeatedly trigger in-list cascades
- what missing accounts should be pulled into the next collection round

### Metadata anomalies

- account age vs activity intensity
- follower/friend ratio
- default profile or default banner use
- blue verification status
- long periods of inactivity followed by synchronized bursts

## 5. What to visualize

### Force-directed graph

- input: `account_overlap.csv` or `network_nodes.csv` + `network_components.csv`
- tool choices:
  - Gephi
  - Cytoscape
  - Observable / D3

### Cascade timeline cards

For each high-repeat original:

- original author
- first retweeter
- retweet order
- lag distribution
- accounts involved

### Heatmaps

- accounts vs hours
- hashtags vs accounts
- source authors vs amplifiers

### Component profiles

For each overlap-network component:

- node count
- edge count
- top accounts
- top sources amplified
- top hashtags
- top mention targets

## 6. What to add when more exports arrive

- daily and weekly rollups
- rolling-window overlap graphs
- component stability over time
- account role changes over time
- narrative shifts by component
- lead-lag persistence across multiple days

## 7. ML direction after that

Start with clustering and anomaly detection before supervised labels.

Recommended first feature groups:

- account-level activity ratios
- overlap-network metrics
- lead-lag metrics
- source concentration metrics
- hashtag and mention concentration metrics
- burst-window participation metrics

Only move to supervised classification after manual review labels are consistent.
