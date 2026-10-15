# ADR 008: Add a Reranker Only When Evaluation Justifies It

## Status

Accepted

## Context

Modern RAG architectures often include a cross-encoder or another reranker after initial retrieval.

Adding a reranker can improve ranking quality in some workloads, but it also increases:

- latency;
- model dependencies;
- compute usage;
- operational complexity;
- deployment complexity;
- evaluation complexity;
- failure modes.

The presence of reranking in reference architectures is not sufficient justification for using it.

FinGraph Agent therefore treats reranking as an optional optimization that must demonstrate measurable value.

## Decision

Do not include a reranker in the current retrieval pipeline.

Maintain:

```text
sparse retrieval
dense retrieval
hybrid retrieval
```

as the current retrieval stack.

Add a reranker only if controlled evaluation demonstrates a meaningful quality improvement that justifies its runtime and operational cost.

## Current retrieval architecture

The current retrieval stack supports:

```text
BM25-style sparse retrieval
dense embedding retrieval
reciprocal-rank-style hybrid fusion
metadata filtering
company-aware diversification
graph-informed dependent retrieval
```

Current dense representation:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

The architecture intentionally evaluates the existing retrievers before introducing another model into the serving path.

## Evaluation criteria

A reranker should not be added based on intuition alone.

It should be evaluated against measurable criteria such as:

```text
Precision@K
MRR
NDCG@K when explicit chunk-level qrels exist
Recall@K when explicit chunk-level qrels exist
latency
throughput
resource usage
```

A candidate reranker should improve retrieval quality enough to justify:

```text
additional inference latency
additional model memory
additional dependencies
additional deployment complexity
additional failure modes
```

## Benchmark evidence

The original development corpus contained:

```text
3,005 chunks
```

The bootstrap retrieval benchmark used:

```text
questions: 12
k:         5
```

The v1.1 retrieval corpus expanded to:

```text
12,666 chunks
```

using the same logical retrieval architecture and benchmark shape.

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

## Why sparse still matters

SEC filings contain substantial exact and domain-specific language, including:

```text
accounting terminology
risk terminology
regulatory phrases
business segment names
product names
financial terminology
```

These characteristics make lexical retrieval an important first-stage capability.

## Why dense is still retained

Dense retrieval remains useful for:

```text
semantic paraphrases
wording mismatch
conceptual similarity
```

Different query types can benefit from different representations, so dense retrieval remains independently available and measurable.

## Why hybrid is still retained

Hybrid retrieval combines lexical and semantic signals.

The architecture preserves it because:

- some questions benefit from complementary signals;
- future datasets may behave differently;
- hybrid routing is part of the broader agent workflow;
- the cost and behavior are measurable.

Hybrid retrieval is not presented as universally superior.

## Current evidence threshold for reranking

The corrected current benchmark does not provide evidence that a second-stage reranker is required.

A mandatory reranker would add:

```text
additional inference
latency
model memory
dependencies
deployment complexity
failure modes
```

without demonstrated benchmark evidence that second-stage scoring provides a meaningful benefit.

A reranker should move into the production path only after a stronger controlled benchmark demonstrates a measurable quality improvement that justifies those costs.

## Latency implications

A reranker typically requires scoring a set of candidate query-document pairs after initial retrieval.

Conceptually:

```text
query
   |
   v
initial retrieval
   |
   v
N candidates
   |
   v
cross-encoder / reranker
   |
   v
top K
```

This introduces an additional inference stage.

For an API that already has route-specific latency characteristics, adding another model to every vector retrieval request should only happen when its value is measured.

## Production consequence

The current production path remains simpler:

```text
query
   |
   v
Qdrant retrieval
   |
   v
canonical evidence
```

rather than:

```text
query
   |
   v
Qdrant retrieval
   |
   v
reranker inference
   |
   v
canonical evidence
```

This reduces:

```text
serving dependencies
cold-start risk
model-loading requirements
runtime latency
deployment size
```

## Evaluation before complexity

FinGraph Agent follows the rule:

> Additional retrieval complexity must earn its place through controlled evaluation.

This means architecture decisions should respond to observed errors rather than trends in reference implementations.

The same principle applies to:

```text
rerankers
query expansion
multi-stage retrieval
larger embedding models
additional retrieval agents
```

## Retrieval error analysis

A reranker becomes relevant when error analysis indicates that:

```text
relevant chunks are retrieved
but ranked too low
```

That is a ranking problem.

A reranker is less likely to solve cases where:

```text
relevant chunks are not retrieved at all
```

which is primarily a recall or query-understanding problem.

The architecture should distinguish these failure classes before selecting a remedy.

## Explicit relevance judgments

The current bootstrap evaluation supports metadata-based relevance judgments.

Explicit chunk-level qrels have not yet been populated.

For that reason:

```text
NDCG@K
Recall@K
```

are not currently reported as numeric release metrics.

A stronger reranking experiment should ideally use richer explicit relevance judgments.

## Candidate reranker evaluation

If reranking is introduced experimentally, it should be compared against the current baseline using the same dataset and retrieval candidates.

A controlled experiment should measure:

```text
baseline sparse
baseline dense
baseline hybrid
retriever + candidate reranker
```

and compare:

```text
P@K
MRR
NDCG@K when explicit chunk-level qrels exist
Recall@K when explicit chunk-level qrels exist
mean latency
P50
P95
P99
throughput
```

## Acceptance criteria

A reranker should only move into the production retrieval path if:

1. quality improvement is consistent rather than isolated to a few cases;
2. improvement is meaningful on the chosen evaluation metrics;
3. latency remains acceptable;
4. serving complexity remains justified;
5. error analysis confirms it addresses a real retrieval weakness.

A marginal quality gain with a large latency increase is not automatically acceptable.

## Candidate scope

Potential future candidates may include:

```text
cross-encoder rerankers
late-interaction models
LLM-based scoring
```

No specific reranker is currently selected.

Choosing a model before establishing the need would reverse the intended evaluation-driven process.

## LLM-as-reranker

Using the primary LLM as a reranker is not the default plan.

LLM reranking can introduce:

- higher cost;
- higher latency;
- provider dependency;
- quota sensitivity;
- nondeterminism.

A dedicated reranking model would typically be preferable if reranking becomes necessary, but this must still be benchmarked.

## Scaling implication

The v1.1 scale extension is useful evidence because the corpus increased substantially:

```text
companies: 6 -> 25
chunks:    3,005 -> 12,666
```

while the corrected current v1.1 benchmark reports sparse/dense/hybrid P@5 of 0.633/0.583/0.633 and MRR of 0.792/0.729/0.686.

This does not provide a measured reason to add reranking purely as a response to corpus growth.

Future scale levels may produce different results and should be evaluated separately.

## Consequences

The current retrieval architecture remains:

```text
simpler
faster
easier to deploy
easier to test
easier to benchmark
```

than a pipeline containing another inference stage.

The trade-off is that the system does not currently benefit from any quality gains a well-selected reranker might provide on future harder cases.

That trade-off is intentional until measured evidence changes.

## Rejected alternatives

### Always use a cross-encoder

Rejected because there is no current benchmark evidence justifying the added cost and latency.

### Rerank only hybrid results by default

Rejected because the corrected current benchmark does not show hybrid retrieval outperforming sparse retrieval on P@5 or MRR.

Adding another stage specifically to hybrid would compound complexity without demonstrated need.

### Use the LLM to score every retrieved chunk

Rejected because it would increase latency, cost, provider dependence, and nondeterminism.

### Add reranking for portfolio complexity

Rejected because project architecture should reflect measured engineering needs rather than maximize the number of components.

## Reconsider when

Revisit this decision when:

- explicit relevance labels are substantially expanded;
- explicit chunk-level qrels are added so NDCG@K and Recall@K can be measured reliably;
- the company universe grows materially;
- additional filing types are introduced;
- retrieval error analysis identifies ranking rather than recall as a major failure mode;
- current P@K or MRR materially degrade, or future qrel-backed NDCG materially degrades;
- a candidate reranker demonstrates consistent measurable improvement;
- production users expose harder retrieval cases than the current bootstrap benchmark.

Until then, reranking remains intentionally excluded from the production retrieval path.