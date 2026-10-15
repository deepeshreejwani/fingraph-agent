# ADR 009: Scale Through Bounded Concurrency and Bounded Batching Before Distributed Infrastructure

## Status

Accepted

## Context

FinGraph Agent needs to scale beyond the original small development corpus while preserving:

- determinism;
- reproducibility;
- idempotency;
- provenance;
- debuggability;
- manageable operational complexity.

The original development corpus contained:

```text
Companies:         6
10-K filings:     48
Sections:        192
Chunks:        3,005
Financial facts: 25,976
Fiscal periods:   355
```

The v1.1 target expanded the system to:

```text
Companies:        25
10-K filings:    198
Sections:        731
Chunks:       12,666
Financial facts: 115,649
Fiscal periods: 1,475
```

This represents more than a four-times increase across the main data dimensions.

A scale extension was therefore required.

However, the workload did not justify immediately introducing:

```text
Spark
Ray
Kafka
Kubernetes
distributed workflow orchestration
```

The design needed to improve throughput and bounded-memory behavior without creating infrastructure that the measured workload did not require.

## Decision

Scale the current architecture using bounded local concurrency and bounded backend-specific batches.

Use:

```text
SEC ingestion
  -> bounded company-level concurrency

Qdrant indexing
  -> company-bounded artifact loading
  -> bounded embedding batches
  -> bounded upsert batches

Neo4j loading
  -> company-bounded artifact loading
  -> bounded UNWIND transactions
```

Preserve the existing logical architecture and data model.

Do not introduce distributed processing infrastructure until measured workload characteristics require it.

## Design principle

The governing principle is:

> Increase concurrency and batch size only within explicit bounds, preserve deterministic partitions, and scale each subsystem according to its actual bottleneck.

This avoids two common failure modes:

```text
unbounded in-process materialization
```

and:

```text
premature distributed architecture
```

## Why not introduce distributed infrastructure immediately

Distributed frameworks can be appropriate for large workloads.

They also introduce:

- additional deployment complexity;
- more services;
- more configuration;
- harder local reproduction;
- additional failure modes;
- more complex observability;
- more difficult debugging;
- more operational cost.

For the current workload, those costs are not justified by measured requirements.

The objective is not to make the architecture appear large.

The objective is to make the current workload execute reliably and measurably.

## SEC ingestion strategy

The SEC ingestion pipeline uses bounded company-level concurrency.

The flow is:

```text
canonical company universe
          |
          v
deterministic partitioning
          |
          v
bounded worker pool
          |
          v
per-company ingestion
          |
          v
batched manifest updates
          |
          v
processed artifacts
```

Current configuration:

```text
companies:              25
company workers:         4
global SEC rate limit:   5 requests / second
```

The global request throttle is shared across workers.

Concurrency therefore does not create an independent SEC rate budget per worker.

## Deterministic partitioning

Company assignment is deterministic.

The current implementation uses sorted round-robin partitioning.

This provides:

- reproducible partition membership;
- simple reasoning;
- stable execution behavior;
- low implementation complexity.

Deterministic partitioning is preferred over opportunistic partition assignment where reproducibility matters.

## SEC ingestion validation

The corrected 25-company cached regeneration produced:

```text
Companies:        25
Filings:         198
Sections:        731
Chunks:       12,666
Financial facts: 115,649
```

Measured execution:

```text
elapsed:               99.637 seconds
filings per minute:   119.233
companies per second:   0.251
filings downloaded:         0
companyfacts refreshed:     0
```

This benchmark represents deterministic regeneration from cached source data.

It is not presented as fresh SEC network-download throughput.

## Qdrant indexing strategy

The retrieval indexing pipeline uses company-bounded loading and bounded embedding/upload batches.

The flow is:

```text
processed company artifacts
          |
          v
company-bounded artifact loading
          |
          v
bounded chunk batches
          |
          +----------------+
          |                |
          v                v
dense encoding        sparse encoding
          |                |
          +-------+--------+
                  |
                  v
             Qdrant upsert
```

This avoids one unbounded whole-corpus embedding operation.

## Qdrant point identity

Qdrant point identifiers remain deterministic.

The implementation uses deterministic UUID5 identifiers.

This supports:

- reproducible rebuilds;
- incremental replacement;
- duplicate avoidance;
- point-count auditing.

## Qdrant rebuild mode

Full rebuild mode replaces the current retrieval collection with the selected canonical corpus.

The v1.1 full rebuild processed:

```text
companies:        25
chunks:       12,666
points:       12,666
batches:         211
batch size:       64
dimensions:      384
```

Measured execution:

```text
artifact load:          0.807 seconds
embedding + upsert:  1563.251 seconds
total elapsed:       1566.280 seconds
throughput:             8.102 chunks / second
```

The point-count audit verified:

```text
chunks == Qdrant points
12,666 == 12,666
```

## Qdrant bottleneck

The measured runtime clearly shows that artifact loading is not the dominant cost.

```text
artifact loading:       0.807 seconds
embedding + upsert:  1563.251 seconds
```

The dominant cost is therefore:

```text
dense/sparse encoding
+
Qdrant upsert
```

Future optimization should target the measured bottleneck rather than introducing complexity in the fast artifact-loading stage.

## Qdrant incremental mode

The indexing architecture also supports ticker-scoped replacement as an **offline-maintenance operation**.

Concurrent serving must be stopped before replacement because the operation is rollback-safe but is not read-atomic for live readers.

Conceptually:

```text
stop concurrent serving
      |
      v
selected tickers
      |
      v
delete existing ticker partitions
      |
      v
load current ticker artifacts
      |
      v
bounded embedding batches
      |
      v
bounded Qdrant upserts
```

The replacement path fails closed unless maintenance mode is explicitly enabled.

This allows controlled incremental updates without rebuilding the entire index.

A validated incremental AAPL replacement preserved the overall point count rather than inflating it.

## Why company-bounded loading

The current processed data is naturally partitioned by company.

Company-level loading provides:

- predictable memory bounds;
- deterministic iteration;
- natural incremental scope;
- easy failure isolation;
- straightforward throughput measurement.

The architecture does not claim streaming byte-level JSON parsing.

The specific implemented strategy is:

> company-bounded artifact loading with bounded downstream processing batches.

## Neo4j loading strategy

Neo4j loading follows the same bounded-execution principle.

The flow is:

```text
processed company artifacts
          |
          v
company-bounded loading
          |
          v
canonical graph rows
          |
          v
bounded UNWIND transactions
          |
          v
Neo4j
```

The scale layer reuses the existing canonical graph model and Cypher queries.

It does not introduce a second graph schema.

## Neo4j transaction batching

The Neo4j storage layer consumes rows as an iterable and divides them into bounded transactions.

Current validated transaction size:

```text
1,000 rows
```

This avoids sending an arbitrarily large list in one transaction.

## Neo4j scale validation

The v1.1 graph load wrote:

```text
loaded rows:             118,078
transaction batch size:    1,000
```

Measured execution:

```text
schema:            1.315 seconds
companies:         0.138 seconds
filings/sections:  6.935 seconds
facts/periods:    52.927 seconds
load:             60.000 seconds
prune:             4.627 seconds
audit:             2.435 seconds
total elapsed:    68.968 seconds
throughput:     1,967.965 rows / second
```

## Graph integrity after scaling

The post-load audit matched the canonical processed corpus exactly:

```text
Companies:       25
Filings:         198
Sections:        731
Financial facts: 115,649
Fiscal periods:  1,475
```

Audit results:

```text
count mismatches: 0
orphans:          0
passed:           true
```

The scale extension therefore preserved graph integrity.

## Stale-entity pruning

The graph build process supports stale-entity pruning.

For the 6-company to 25-company expansion, prune counts were:

```text
0
```

because the original 6-company graph was a subset of the expanded canonical corpus.

Pruning remains important for future incremental corpus changes where entities may disappear from the canonical processed data.

## Retrieval quality after scaling

Scaling storage is not sufficient if retrieval quality degrades.

The retrieval benchmark was rerun on the expanded Qdrant collection after the corpus increased from:

```text
3,005 chunks
```

to:

```text
12,666 chunks
```

using the same 12-question benchmark at:

```text
k = 5
```

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

NDCG@5 and Recall@5 are `N/A` because the benchmark does not contain explicit chunk-level qrels.

The corrected report is the current release source for retrieval-quality and v1.1 latency claims.

## Corpus growth

The scale extension produced:

| Entity | Baseline | v1.1 | Growth |
|---|---:|---:|---:|
| Companies | 6 | 25 | 4.167x |
| Filings | 48 | 198 | 4.125x |
| Sections | 192 | 731 | 3.807x |
| Chunks | 3,005 | 12,666 | 4.215x |
| Financial facts | 25,976 | 115,649 | 4.452x |
| Fiscal periods | 355 | 1,475 | 4.155x |

The objective of v1.1 was therefore not simply to add more tickers.

It validated that the same logical architecture could operate over a materially larger corpus using bounded execution patterns.

## Backend-specific scaling

The design intentionally avoids a single generic scaling mechanism.

Different subsystems have different workloads.

### SEC ingestion

Primary concern:

```text
network request rate
company-level independence
source caching
```

Chosen strategy:

```text
bounded worker concurrency
global request throttle
```

### Qdrant

Primary concern:

```text
embedding cost
batch upload cost
```

Chosen strategy:

```text
bounded embedding batches
bounded upserts
incremental ticker replacement
```

### Neo4j

Primary concern:

```text
transaction size
large write sets
```

Chosen strategy:

```text
bounded UNWIND transactions
```

This backend-specific approach keeps each optimization aligned with the actual bottleneck.

## Memory philosophy

The scale architecture should avoid unnecessary whole-corpus materialization where a natural partition exists.

The preferred processing shape is:

```text
partition
   |
   v
bounded batch
   |
   v
backend operation
   |
   v
next batch
```

rather than:

```text
load everything
   |
   v
build one enormous in-memory payload
   |
   v
single backend operation
```

## Incremental-update philosophy

The architecture favors deterministic partitions that can be replaced independently.

Current examples include:

```text
company SEC artifacts
ticker Qdrant partitions
deterministic Neo4j entities
persistent LLM request identities
```

This creates a path toward future incremental updates without requiring a streaming architecture.

## Why not Spark

Spark was not selected because the current workload does not require distributed dataframe or shuffle processing.

The main measured bottlenecks are:

```text
embedding + Qdrant upload
Neo4j graph writes
```

rather than large-scale analytical joins requiring a distributed compute engine.

## Why not Ray

Ray was not selected because local bounded worker concurrency currently addresses the company-level parallelism requirement.

Introducing a distributed task runtime would add operational complexity without addressing a measured bottleneck.

## Why not Kafka

Kafka was not selected because the current ingestion model is batch/incremental rather than a continuous high-volume event stream.

SEC filing updates do not currently require an always-on streaming backbone.

## Why not Kubernetes

Kubernetes was not selected because the current service topology and workload do not require cluster orchestration.

Managed Railway, Qdrant Cloud, and Neo4j Aura already provide sufficient operational separation for the current project.

Adding Kubernetes would increase deployment complexity without improving the demonstrated workload.

## Why not maximize worker count

More workers are not automatically better.

SEC ingestion is constrained by responsible request-rate behavior.

Qdrant indexing is constrained largely by encoding and backend interaction.

Neo4j writes are constrained by transaction behavior.

Concurrency should therefore remain bounded according to backend characteristics.

## Observability consequence

Scale execution should expose measurable runtime statistics.

Useful metrics include:

```text
items processed
batch count
elapsed time
throughput
artifact load time
backend operation time
audit result
```

This allows future optimization to target measured bottlenecks.

## Failure-isolation consequence

Natural company partitions also improve failure isolation.

A processing error can be associated with:

```text
ticker
company artifact
batch
backend stage
```

rather than being hidden inside one monolithic whole-corpus operation.

## Reproducibility consequence

Scaling should not sacrifice determinism.

The architecture therefore preserves:

```text
stable company ordering
deterministic partition assignment
deterministic IDs
deterministic graph identities
canonical processed artifacts
repeatable audit rules
```

Concurrency changes execution order, but it should not change the logical corpus.

## Operational consequence

This decision keeps the system deployable on modest infrastructure while preserving a clear future scaling path.

The current approach requires less operational machinery than a distributed platform while still demonstrating:

```text
bounded parallel execution
batch processing
incremental updates
throughput measurement
large-corpus integrity auditing
```

## Rejected alternatives

### Keep all original whole-corpus loaders unchanged

Rejected because v1.1 explicitly requires scale-oriented execution behavior.

### Introduce distributed infrastructure immediately

Rejected because the measured workload does not currently require it.

### One universal batch size across all systems

Rejected because SEC, Qdrant, and Neo4j have different operational characteristics.

### Unbounded concurrency

Rejected because it creates unpredictable:

```text
memory usage
network pressure
provider pressure
database load
```

### Claim streaming ingestion without implementing it

Rejected because the current implementation uses company-bounded artifact loading, not byte-stream source parsing.

Architecture documentation should describe the system that actually exists.

## Consequences

This decision introduces:

- deterministic company partitioning;
- bounded SEC worker concurrency;
- shared request throttling;
- batched manifest persistence;
- company-bounded retrieval artifact loading;
- bounded embedding batches;
- bounded Qdrant upserts;
- incremental ticker replacement;
- company-bounded graph loading;
- bounded Neo4j transactions;
- scale-specific metrics;
- post-load integrity audits.

The benefit is a system that scales materially beyond the development baseline without unnecessary distributed infrastructure.

## Current validated outcome

The v1.1 scale extension has validated:

```text
25 companies
198 filings
731 sections
12,666 chunks
115,649 facts
1,475 fiscal periods
```

with:

```text
Qdrant:
12,666 chunks == 12,666 points

Neo4j:
0 count mismatches
0 orphans

Retrieval:
corrected current report available;
P@5 = 0.633 / 0.583 / 0.633;
MRR = 0.792 / 0.729 / 0.686;
NDCG@5 and Recall@5 = N/A
```

## Reconsider when

This decision should be revisited when measured workloads exceed the practical limits of bounded single-node execution.

Signals could include:

- company count increasing by orders of magnitude;
- ingestion runtime becoming operationally unacceptable;
- embedding throughput becoming a release bottleneck;
- local memory limits preventing bounded processing;
- Neo4j write throughput becoming inadequate;
- multiple concurrent corpus builds being required;
- continuous filing ingestion requiring event-driven processing;
- horizontal worker coordination becoming necessary.

At that point, candidate technologies such as:

```text
distributed task queues
Ray
Spark
Kafka
Kubernetes
managed batch platforms
```

should be evaluated against measured requirements.

They should be introduced because the workload requires them, not because they make the architecture diagram larger.