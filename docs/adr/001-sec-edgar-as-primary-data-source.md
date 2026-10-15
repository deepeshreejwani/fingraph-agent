# ADR 001: SEC EDGAR as the Primary Data Source

## Status

Accepted

## Context

FinGraph Agent requires an authoritative financial-data source that supports both narrative retrieval and structured financial analysis across companies and fiscal periods.

The source must provide:

- stable company identifiers;
- stable filing identifiers;
- historical filings;
- source provenance;
- structured financial facts;
- temporal information;
- enough real-world complexity to exercise ingestion, retrieval, graph modelling, evaluation, scaling, and agentic workflows.

## Options considered

### SEC EDGAR

SEC EDGAR provides:

- regulatory filings;
- stable CIK identifiers;
- accession numbers;
- historical 10-K documents;
- filing metadata;
- structured XBRL facts;
- direct source provenance.

### Commercial financial-data APIs

Commercial APIs can provide normalized financial data and simpler interfaces.

However, they may:

- hide source-level complexity;
- introduce licensing constraints;
- reduce control over normalization;
- weaken provenance;
- remove much of the ingestion-engineering challenge the project intends to solve.

### Curated static datasets

Static datasets are simple to develop against but are insufficient for demonstrating:

- reproducible ingestion;
- source provenance;
- historical document acquisition;
- temporal normalization;
- continuously extensible financial intelligence;
- incremental data processing;
- realistic scale behavior.

## Decision

Use SEC EDGAR as the primary source for:

- filing documents;
- filing metadata;
- company identifiers;
- XBRL financial facts.

The current ingestion scope is restricted to Form 10-K and associated XBRL data.

10-Q support is deferred until annual-report ingestion, temporal semantics, evaluation, and production behavior are fully validated.

## Reasons

SEC EDGAR provides:

- an authoritative regulatory source;
- stable company and filing identifiers;
- narrative and structured data from the same source ecosystem;
- historical and temporal analysis;
- source-level provenance;
- a realistic ingestion workload;
- a natural path from local development to larger-company-universe benchmarks.

It also forces the system to solve meaningful engineering problems including:

- irregular HTML;
- historical filing discovery;
- inconsistent filing layouts;
- taxonomy differences;
- duplicate XBRL observations;
- period normalization;
- cache integrity;
- deterministic persistence;
- bounded request behavior;
- reproducible incremental processing.

These are useful properties for the goals of FinGraph Agent rather than inconveniences to be abstracted away.

## Trade-offs

SEC filings contain:

- inconsistent HTML structure;
- taxonomy variation;
- amendments;
- duplicated or overlapping XBRL facts;
- changing filing conventions;
- temporal normalization challenges;
- section-heading inconsistencies;
- filing-specific structural edge cases.

The system therefore requires more ingestion and normalization engineering than a pre-cleaned financial API.

SEC access also requires responsible request behavior and an identifiable User-Agent.

The pipeline must therefore balance:

```text
correctness
reproducibility
request-rate compliance
parser robustness
throughput
```

rather than maximizing raw request concurrency.

## Implementation consequences

The ingestion layer must support:

- bounded SEC request rates;
- an identifiable SEC User-Agent;
- retries;
- local caching;
- deterministic accession identities;
- content hashing;
- cache-integrity validation;
- raw-data preservation;
- normalized processed artifacts;
- XBRL deduplication;
- provenance retention;
- deterministic section identifiers;
- deterministic chunk identifiers;
- deterministic fact identifiers;
- bounded concurrency;
- deterministic company partitioning;
- resumable and repeatable execution.

## Request control

SEC ingestion uses a shared request throttle.

The current scale configuration uses:

```text
maximum SEC request rate: 5 requests / second
company workers:          4
```

The throttle is shared across concurrent workers.

This avoids treating each worker as an independent request-rate budget.

Concurrency therefore improves pipeline utilization without intentionally exceeding the configured global SEC request rate.

## Deterministic company partitioning

The v1.1 ingestion path partitions the company universe deterministically.

The current strategy uses sorted round-robin assignment.

Conceptually:

```text
sorted companies
      |
      v
deterministic partition assignment
      |
      v
bounded worker pool
      |
      v
company-level ingestion
```

This supports reproducible partition membership while remaining simple enough for the current workload.

## Caching strategy

Downloaded SEC source content is cached locally.

Where valid cached artifacts exist, the ingestion pipeline can reuse them rather than repeatedly downloading unchanged filing content.

Cache-integrity checks are retained so cached content is not blindly trusted.

This separation allows the system to benchmark:

```text
source acquisition
```

and:

```text
deterministic corpus regeneration
```

as distinct concerns.

## Parsing consequences

Using raw SEC filings means the system must tolerate non-uniform document structures.

The parser therefore uses deterministic extraction logic including:

- heading normalization;
- canonical Item-heading detection;
- section-boundary rules;
- minimum section-length validation;
- section-specific maximum-length validation;
- deterministic semantic-heading fallback where required.

The parser does not use an LLM to decide section boundaries.

This keeps ingestion reproducible and testable.

## Structured fact consequences

SEC XBRL facts contain overlapping temporal and filing-context metadata.

The ingestion layer must preserve enough information to distinguish:

```text
filing context
reporting period
economic period
instant facts
duration facts
```

The filing's fiscal-year metadata is therefore not treated as a universal substitute for the economic year of every contained financial fact.

Normalized fiscal-period semantics are derived downstream.

## Industry metadata

SEC filings provide SIC-related metadata, but the current project does not invent or infer arbitrary industry labels merely to populate an optional field.

Industry enrichment is deferred until a reliable source-backed mapping is selected and validated.

This preserves the project's provenance requirement.

## Current validated corpus

The current v1.1 corpus contains:

```text
Companies:        25
10-K filings:    198
Sections:        731
Chunks:       12,666
Financial facts: 115,649
Fiscal periods: 1,475
```

Current company universe:

```text
AAPL
MSFT
GOOGL
AMZN
META
NVDA
AMD
INTC
AVGO
QCOM
CSCO
IBM
ORCL
CRM
ADBE
NOW
PANW
NFLX
TSLA
TXN
MU
AMAT
LRCX
KLAC
PLTR
```

## Scale validation

The original development corpus contained:

```text
Companies:         6
10-K filings:     48
Sections:        192
Chunks:        3,005
Financial facts: 25,976
Fiscal periods:   355
```

The v1.1 corpus increased this to:

| Entity | Baseline | v1.1 | Growth |
|---|---:|---:|---:|
| Companies | 6 | 25 | 4.167x |
| 10-K filings | 48 | 198 | 4.125x |
| Sections | 192 | 731 | 3.807x |
| Chunks | 3,005 | 12,666 | 4.215x |
| Financial facts | 25,976 | 115,649 | 4.452x |
| Fiscal periods | 355 | 1,475 | 4.155x |

## Measured cached regeneration

The corrected 25-company cached regeneration produced:

```text
Companies:             25
Filings:              198
Sections:             731
Chunks:            12,666
Financial facts:  115,649
```

Measured execution:

```text
elapsed:               99.637 seconds
filings per minute:   119.233
companies per second:   0.251
filings downloaded:         0
companyfacts refreshed:     0
```

This measurement represents deterministic regeneration from already cached SEC source material.

It should not be interpreted as fresh SEC network-download throughput.

## Consequences

Using SEC EDGAR increases implementation complexity compared with consuming a normalized commercial API.

The project must own:

```text
acquisition
caching
HTML normalization
section extraction
XBRL normalization
temporal interpretation
deduplication
provenance
request-rate control
scale behavior
```

The benefit is that answers remain traceable to authoritative regulatory source material and the project exercises the full data-to-serving lifecycle.

## Reconsider when

This decision should be revisited if:

- the project requires data that SEC filings do not provide;
- licensing requirements justify an additional commercial source;
- real-time market data becomes part of the project scope;
- filing types beyond the SEC ecosystem become necessary;
- source redundancy is required for production availability.

Additional sources may supplement SEC EDGAR in the future.

They should not silently replace SEC provenance for claims derived from regulatory filings.