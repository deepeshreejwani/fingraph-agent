# ADR 003: Maintain Sparse, Dense, and Hybrid Retrieval

## Status

Accepted

## Context

SEC filings contain both highly specific terminology and semantically varied narrative language.

A retrieval system over annual filings must handle questions involving:

```text
exact accounting terms
regulatory language
product names
business concepts
risk themes
paraphrased user questions
historical filing context
```

Lexical and semantic retrieval therefore have complementary strengths.

The retrieval architecture should preserve both capabilities without assuming that one method is universally superior.

## Decision

Maintain three retrieval methods:

- sparse;
- dense;
- hybrid.

Evaluate each method independently.

Do not assume hybrid retrieval is automatically better than sparse or dense retrieval.

Do not add additional retrieval stages, including reranking, unless controlled evaluation demonstrates measurable benefit.

## Sparse retrieval

Sparse retrieval uses BM25-style sparse vectors in Qdrant.

Sparse retrieval is useful when exact terminology matters.

Typical strengths include:

- accounting terminology;
- SEC language;
- regulation names;
- product names;
- named business programs;
- repeated filing terminology;
- exact phrase matching.

## Dense retrieval

Dense retrieval uses:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

Dense retrieval is useful when the user's wording differs from the filing text.

Typical strengths include:

- semantic paraphrases;
- conceptual similarity;
- wording variation;
- natural-language reformulation.

## Hybrid retrieval

Hybrid retrieval combines sparse and dense rankings using reciprocal-rank-style fusion.

Conceptually:

```text
query
  |
  +----------+
  |          |
  v          v
sparse      dense
  |          |
  +----+-----+
       |
       v
 rank fusion
       |
       v
 final results
```

Hybrid retrieval is retained because lexical and semantic signals can complement each other.

However, the architecture does not assume that the fused result must outperform both component retrievers on every dataset.

## Metadata filtering

Retrieval results can be constrained using metadata such as:

```text
ticker
fiscal year
section
filing identity
```

This is important because many financial questions contain explicit company or temporal constraints.

Filtering reduces irrelevant cross-company or cross-period evidence before synthesis.

## Company diversification

Multi-company questions can otherwise be dominated by chunks from one company.

The retrieval layer therefore supports company-aware diversification where appropriate.

This helps preserve evidence coverage across requested entities.

Diversification is a retrieval control rather than a synthesis-time attempt to compensate for missing evidence.

## Dependent retrieval

The retrieval architecture also supports graph-informed retrieval.

Some HYBRID questions can follow:

```text
question
   |
   v
structured graph lookup
   |
   v
company / period / metric context
   |
   v
vector retrieval
```

This allows structured results to constrain the narrative search space.

The architecture therefore supports more than simple independent sparse/dense querying.

## Evaluation methodology

Retrieval is evaluated separately from answer generation.

The current bootstrap benchmark uses:

```text
questions: 12
k:         5
```

Each method is evaluated on the same cases.

Current reported quality metrics are:

```text
Precision@5
MRR
```

Latency metrics include:

```text
mean
P95
```

NDCG@5 and Recall@5 are reported as `N/A` because the current bootstrap dataset does not contain explicit chunk-level qrels.

## Benchmark evidence

The original development corpus contained:

```text
3,005 chunks
```

The v1.1 retrieval index was expanded to:

```text
12,666 chunks
```

using the same logical retrieval architecture and the same 12-question bootstrap benchmark at `k=5`.

The historical `retrieval_bootstrap_report.json` remains pre-fix baseline evidence.

The corrected current v1.1 release report is:

```text
evals/reports/retrieval_scale_v1_1_report.json
```

Current results are:

| Method | P@5 | MRR | NDCG@5 | Recall@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 0.633 | 0.792 | N/A | N/A | 9.782 ms | 35.278 ms |
| Dense | 0.583 | 0.729 | N/A | N/A | 18.894 ms | 65.213 ms |
| Hybrid | 0.633 | 0.686 | N/A | N/A | 17.204 ms | 34.962 ms |

NDCG@5 and Recall@5 are `N/A` because explicit chunk-level qrels are absent from the current benchmark dataset.

## Interpretation

The architectural decision is based on the distinct capabilities of the three retrieval paths together with the corrected current benchmark.

All three retrieval methods remain because they provide distinct capabilities:

```text
Sparse
  lexical and exact-term matching

Dense
  semantic paraphrase and wording-variation matching

Hybrid
  fusion of lexical and semantic signals
```

Each path remains independently measurable.

## Why keep dense retrieval

Dense retrieval provides a complementary capability for semantic paraphrases and wording mismatch.

Removing it based only on one small historical benchmark would unnecessarily reduce retrieval coverage.

## Why keep hybrid retrieval

Hybrid retrieval is retained because:

- sparse and dense methods capture different signals;
- some queries benefit from both;
- the agent supports HYBRID workflows;
- future datasets may have different retrieval characteristics;
- retrieval architecture should remain evaluable rather than prematurely collapsing into one method.

Hybrid retrieval is not presented as automatically superior.

## Reranking decision

A cross-encoder or other reranker is not currently part of the retrieval pipeline.

This is intentional.

Adding a reranker would increase:

```text
latency
compute
dependencies
operational complexity
evaluation complexity
```

A reranker should only be introduced if a stronger controlled benchmark demonstrates measurable improvement that justifies that cost.

## Incremental indexing

The retrieval architecture supports incremental ticker-level replacement as an **offline-maintenance operation**.

Concurrent serving must be stopped before replacement because the operation is rollback-safe but is not read-atomic for live readers.

Conceptually:

```text
stop concurrent serving
      |
      v
selected ticker
      |
      v
delete ticker partition
      |
      v
load current ticker chunks
      |
      v
bounded embedding batches
      |
      v
Qdrant upsert
```

The replacement path fails closed unless maintenance mode is explicitly enabled.

Deterministic point identifiers prevent logical duplication.

Incremental updates should preserve total point consistency rather than append duplicate records.

## Rebuild indexing

Full rebuild mode indexes the complete selected corpus.

The v1.1 full rebuild processed:

```text
companies: 25
chunks:    12,666
points:    12,666
batches:   211
batch size: 64
```

Measured execution:

```text
artifact load:          0.807 seconds
embedding + upsert:  1563.251 seconds
total elapsed:       1566.280 seconds
throughput:             8.102 chunks / second
```

The measured bottleneck is embedding and upsert rather than artifact loading.

## Consequences

Maintaining three retrieval strategies adds some implementation and evaluation complexity.

The application must support:

```text
multiple query paths
multiple score semantics
fusion logic
method-specific latency
method-specific evaluation
```

The benefit is that retrieval behavior remains explicit and measurable rather than being hidden behind one opaque pipeline.

## Rejected alternatives

### Sparse only

Rejected because lexical matching does not fully address semantic paraphrases.

### Dense only

Rejected because exact SEC and accounting terminology is important and currently performs strongly with sparse retrieval.

### Hybrid only

Rejected because hybrid retrieval currently does not dominate sparse retrieval and has higher latency.

Keeping only hybrid would remove useful visibility into the behavior of its component retrievers.

### Mandatory reranking

Rejected because current evidence does not justify the additional runtime complexity.

## Reconsider when

This decision should be revisited when:

- the retrieval benchmark becomes substantially larger;
- explicit relevance judgments are expanded;
- explicit chunk-level qrels are added so NDCG@K and Recall@K can be measured reliably;
- the company universe grows materially beyond the current scale;
- ranking error analysis identifies clear failures;
- a reranker demonstrates measurable improvement;
- production latency requirements require simplifying the retrieval stack;
- retrieval behavior changes significantly after future filing types are added.

Until then, sparse, dense, and hybrid retrieval remain independently available and independently evaluated.