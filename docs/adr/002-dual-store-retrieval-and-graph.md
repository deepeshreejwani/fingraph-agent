# ADR 002: Separate Narrative Retrieval and Structured Financial Graph

## Status

Accepted

## Context

Financial questions do not all require the same representation.

Narrative questions require access to filing text.

Structured questions require normalized numerical facts and explicit temporal relationships.

Trying to represent both workloads through only a vector database would make structured financial reasoning unnecessarily approximate.

Trying to represent all filing narrative inside a graph would make semantic retrieval unnecessarily complex.

The architecture therefore needs to support both:

```text
unstructured filing evidence
```

and:

```text
structured financial facts
```

without forcing either workload into the wrong storage model.

## Decision

Use two specialized persistence systems:

- Qdrant for narrative retrieval;
- Neo4j for structured financial relationships.

Normalize results from both stores into a shared evidence model before answer synthesis.

The application layer is responsible for deciding whether a question requires:

```text
VECTOR
GRAPH
HYBRID
```

execution.

## Qdrant responsibilities

Qdrant stores narrative filing chunks.

It supports:

- sparse retrieval;
- dense retrieval;
- hybrid retrieval;
- metadata filtering;
- ticker filtering;
- fiscal-year filtering;
- filing filtering;
- section filtering.

Current dense representation:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

Current sparse representation uses BM25-style sparse vectors.

Qdrant is optimized for retrieving narrative evidence from SEC filing text.

## Neo4j responsibilities

Neo4j stores structured relationships among:

- companies;
- filings;
- sections;
- financial facts;
- fiscal periods.

Current node types:

```text
Company
Filing
Section
FinancialFact
FiscalPeriod
```

Current core relationships:

```text
(:Company)-[:FILED]->(:Filing)

(:Filing)-[:HAS_SECTION]->(:Section)

(:Company)-[:REPORTS]->(:FinancialFact)

(:Filing)-[:REPORTED_FACT]->(:FinancialFact)

(:FinancialFact)-[:FOR_PERIOD]->(:FiscalPeriod)
```

Neo4j is used for deterministic structured lookup, historical comparison, period-aware queries, and financial calculations.

## Why not vector-only

A vector-only architecture was rejected because questions such as:

```text
What was Microsoft's revenue in 2026?
```

should not depend on semantic retrieval.

The desired answer should come from normalized financial facts with explicit company and period relationships.

A vector-only design would make structured lookup unnecessarily approximate and would weaken temporal guarantees.

It would also make calculations depend on extracting numerical values from narrative or semi-structured text.

## Why not graph-only

A graph-only architecture was rejected because questions such as:

```text
What risks does NVIDIA describe around its supply chain?
```

require semantic retrieval over large amounts of filing prose.

Representing all filing prose as graph structure would create unnecessary complexity without improving the narrative-retrieval workload.

Graph traversal is not a replacement for semantic or lexical document search.

## Shared evidence model

Results from both stores are normalized into a canonical evidence representation.

Evidence can contain information such as:

```text
evidence ID
source type
ticker
company
filing
section
fiscal period
text or structured payload
source URL
score
rank
provenance
```

This allows answer synthesis to operate over one downstream contract rather than being coupled directly to Qdrant or Neo4j.

Conceptually:

```text
Qdrant results --------+
                       |
                       v
                canonical evidence
                       ^
                       |
Neo4j results ---------+
```

## Route behavior

### VECTOR

Used when the answer is primarily narrative.

Example:

```text
What risks does NVIDIA discuss in its filing?
```

### GRAPH

Used when the question requires structured financial facts or deterministic calculations.

Example:

```text
What was Microsoft's revenue in 2026?
```

### HYBRID

Used when both structured numerical evidence and narrative explanation are required.

Example:

```text
How did Microsoft revenue change from 2025 to 2026 and what factors drove the change?
```

The graph path provides structured financial evidence.

The vector path provides narrative filing context.

Both are normalized before synthesis.

## Dependent graph-to-vector execution

Hybrid execution is not limited to running two independent searches in parallel.

Some questions require graph evidence to constrain subsequent narrative retrieval.

Conceptually:

```text
question
   |
   v
graph query
   |
   v
company / period / metric context
   |
   v
vector retrieval
   |
   v
combined evidence
```

This allows structured results to narrow the narrative search space.

The architecture therefore supports both:

```text
parallel hybrid retrieval
```

and:

```text
dependent graph -> vector retrieval
```

where appropriate.

## Multi-company behavior

The graph can return structured evidence for multiple companies.

The retrieval layer supports diversification so that one company's filing chunks do not necessarily dominate a multi-company narrative result.

Evidence preserves entity-level provenance so downstream synthesis can distinguish which company and filing supports each claim.

## Consequences

The application must maintain:

- two persistence systems;
- separate build and indexing pipelines;
- cross-store provenance;
- orchestration logic;
- separate health checks;
- separate scale characteristics;
- separate failure modes.

This adds operational complexity.

However, each store is used for the workload it represents well.

The additional complexity is therefore intentional rather than accidental duplication.

## Local deployment consequence

The local production-style stack includes:

```text
fingraph-qdrant
fingraph-neo4j
fingraph-api
```

Qdrant and Neo4j are independently health-checked.

The API should not be considered fully ready when a required persistence dependency is unavailable.

## Managed deployment consequence

The managed deployment separates persistence from application compute.

Current production topology uses:

```text
Railway      -> API runtime
Qdrant Cloud -> narrative vector store
Neo4j Aura   -> structured graph store
```

This preserves the same logical dual-store architecture used locally.

The application does not require a different retrieval or graph model merely because the persistence systems are managed.

## Scaling consequence

Each store has a different scaling path.

### Qdrant

The v1.1 indexing pipeline uses:

```text
company-bounded artifact loading
bounded embedding batches
bounded sparse encoding batches
bounded Qdrant upserts
ticker-scoped incremental replacement
```

### Neo4j

The v1.1 graph loader uses:

```text
company-bounded artifact loading
bounded UNWIND transactions
deterministic graph identities
stale-entity pruning
post-load integrity audit
```

Scaling is therefore implemented independently according to each backend's workload.

## Current validated scale

The current v1.1 corpus contains:

```text
Companies:        25
10-K filings:    198
Sections:        731
Chunks:       12,666
Financial facts: 115,649
Fiscal periods: 1,475
```

Current Qdrant validation:

```text
expected chunks: 12,666
Qdrant points:   12,666
```

Current Neo4j validation:

```text
Companies:       25
Filings:         198
Sections:        731
Financial facts: 115,649
Fiscal periods: 1,475

Count mismatches: 0
Orphans:          0
```

## Retrieval quality after scaling

The narrative corpus increased from:

```text
3,005 chunks
```

to:

```text
12,666 chunks
```

The existing 12-question bootstrap retrieval benchmark is retained as a regression benchmark across sparse, dense, and hybrid retrieval.

The historical `retrieval_bootstrap_report.json` remains pre-fix baseline evidence.

The corrected current v1.1 release report is:

```text
evals/reports/retrieval_scale_v1_1_report.json
```

Current retrieval results are:

| Method | P@5 | MRR | NDCG@5 | Recall@5 | Mean latency | P95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 0.633 | 0.792 | N/A | N/A | 9.782 ms | 35.278 ms |
| Dense | 0.583 | 0.729 | N/A | N/A | 18.894 ms | 65.213 ms |
| Hybrid | 0.633 | 0.686 | N/A | N/A | 17.204 ms | 34.962 ms |

NDCG@5 and Recall@5 are `N/A` because the benchmark does not contain explicit chunk-level qrels.

The architecture decision therefore continues to preserve sparse, dense, and hybrid retrieval as independently evaluable capabilities without claiming unsupported current NDCG or Recall values.

## Trade-offs

The dual-store architecture requires:

```text
additional operational state
more deployment configuration
separate migrations
separate audits
cross-store observability
careful evidence normalization
```

It also creates the possibility of partial dependency failure.

For example:

```text
Qdrant available
Neo4j unavailable
```

or:

```text
Neo4j available
Qdrant unavailable
```

The application must surface these states explicitly rather than pretending the full system is ready.

## Security consequence

Text-to-Cypher safety is separate from vector retrieval safety.

Graph execution requires additional controls because generated Cypher can potentially reference schema, perform expensive queries, or attempt mutation.

The Neo4j path therefore applies:

```text
static validation
schema restrictions
parameter validation
required result bounds
EXPLAIN preflight
query timeout
read-routed execution
bounded repair
```

These application safeguards are not presented as equivalent to a verified database-level read-only authorization boundary.

## Rejected alternatives

### One vector store for everything

Rejected because deterministic financial lookup and period-aware comparisons should not depend on semantic search.

### One graph for everything

Rejected because large-scale narrative retrieval over filing prose is naturally a search problem rather than a graph-traversal problem.

### Duplicate all data into both stores

Rejected because it would increase synchronization cost without a clear workload benefit.

The design stores each representation where it is useful while preserving enough shared identity and provenance to reconnect evidence downstream.

## Reconsider when

This decision should be revisited if:

- one backend can demonstrably satisfy both workloads without weakening correctness;
- operational complexity materially outweighs the benefit of specialized stores;
- the graph evolves to include richer narrative entities that materially change retrieval requirements;
- a future storage engine provides both high-quality semantic retrieval and deterministic relationship querying with comparable operational behavior;
- measured production workloads show that the split creates an unacceptable availability or maintenance burden.

Until then, specialized vector and graph stores remain the preferred architecture.