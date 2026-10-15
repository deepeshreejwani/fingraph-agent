# FinGraph Agent

FinGraph Agent is a production-oriented financial intelligence system that answers questions over SEC filings by combining:

- unstructured narrative evidence from SEC 10-K filings;
- structured financial facts from SEC XBRL;
- sparse, dense, and hybrid retrieval;
- a deterministic financial knowledge graph;
- controlled LangGraph orchestration;
- safe Text-to-Cypher with bounded self-healing;
- deterministic financial calculations;
- grounded answer synthesis with provenance;
- resumable LLM execution;
- layered evaluation;
- observability;
- production API serving;
- scale-oriented ingestion, indexing, and graph loading.

The project is designed as an **Applied AI / ML Engineering system**, not as a standalone RAG demo.

> FinGraph Agent is an engineering and research project for analyzing public SEC filings. It does not provide investment, trading, legal, or accounting advice.

It focuses on the engineering problems that appear when:

```text
financial documents
+
structured financial data
+
temporal semantics
+
retrieval
+
LLM reasoning
+
knowledge graphs
+
agentic orchestration
+
reliability
+
evaluation
+
production serving
```

must work together.

---

# Status

## Phase 1

Phase 1 implements the planned system through the v1.1 scale extension.

Core implementation is complete across:

```text
SEC ingestion
baseline retrieval
production retrieval
knowledge graph
Graph QA
Text-to-Cypher
LangGraph orchestration
bounded self-healing
deterministic calculations
grounded synthesis
resumable LLM execution
FastAPI
SSE
observability
formal load testing
managed deployment
25-company scale extension
```

Phase 1 implementation and pre-commit verification are complete. The repository is ready for its first Phase-1 commit.

Remaining post-commit release work is:

```text
first/final Phase 1 commit
GitHub push
hosted GitHub Actions validation
GitHub-to-Railway deployment connection
GitHub-triggered deployment validation
final public smoke validation
```

The repository should only be considered release-complete after the remaining post-commit gates are executed successfully.

---

# What FinGraph Agent Can Do

FinGraph Agent supports several classes of financial questions.

## Narrative questions

Example:

```text
What supply chain risks does NVIDIA discuss in its filings?
```

These questions use SEC filing text from the retrieval system.

---

## Structured financial questions

Example:

```text
What was Microsoft's revenue in 2026?
```

These questions are answered from normalized structured financial facts in Neo4j.

---

## Deterministic analytical questions

Example:

```text
By what percentage did Microsoft revenue change from 2025 to 2026?
```

The system retrieves the required structured values and performs the arithmetic deterministically in Python.

The LLM does not serve as the authoritative arithmetic engine.

---

## Hybrid questions

Example:

```text
How did Microsoft's revenue change from 2025 to 2026 and what factors drove the change?
```

These questions combine:

```text
structured graph evidence
+
deterministic calculations
+
narrative filing evidence
```

before grounded answer synthesis.

---

## Multi-company questions

The system can retrieve structured and narrative evidence across multiple companies while preserving entity-level provenance.

---

# High-Level Architecture

```text
                                SEC EDGAR
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
               10-K filings                XBRL company facts
                    |                             |
                    v                             v
            HTML normalization             Fact normalization
                    |                             |
                    v                             v
              Filing sections              Financial facts
                    |
                    v
            Deterministic chunks
                    |
          +---------+----------+
          |                    |
          v                    v
       Qdrant                Neo4j
 Narrative evidence    Structured financial graph
          |                    |
          +---------+----------+
                    |
                    v
              LangGraph Router
             /       |        \
            /        |         \
       VECTOR      GRAPH      HYBRID
          |          |           |
          |          v           |
          |    Text-to-Cypher    |
          |          |           |
          |       Validator      |
          |          |           |
          |       EXPLAIN        |
          |          |           |
          |    bounded repair    |
          |          |           |
          +----------+-----------+
                    |
                    v
          Canonical Evidence Model
                    |
                    v
          Deterministic Calculations
                    |
                    v
           Grounded LLM Synthesis
                    |
                    v
       Answer + citations + provenance
                    |
                    v
                  FastAPI
             /health /ready
               /ask /ask/stream
```

Detailed architecture:

```text
docs/architecture.md
```

---

# Current Corpus

The current v1.1 corpus contains:

| Entity | Count |
|---|---:|
| Companies | 25 |
| 10-K filings | 198 |
| Filing sections | 731 |
| Retrieval chunks | 12,666 |
| Financial facts | 115,649 |
| Fiscal periods | 1,475 |

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

---

# Baseline to v1.1 Scale Growth

The original development corpus contained:

```text
6 companies
48 filings
192 sections
3,005 chunks
25,976 financial facts
355 fiscal periods
```

The v1.1 scale extension increased the corpus to:

| Entity | Baseline | v1.1 | Growth |
|---|---:|---:|---:|
| Companies | 6 | 25 | 4.167x |
| Filings | 48 | 198 | 4.125x |
| Sections | 192 | 731 | 3.807x |
| Chunks | 3,005 | 12,666 | 4.215x |
| Financial facts | 25,976 | 115,649 | 4.452x |
| Fiscal periods | 355 | 1,475 | 4.155x |

The scale extension preserves the same logical system rather than creating a separate large-corpus architecture.

---

# Data Source

FinGraph Agent uses the U.S. SEC EDGAR ecosystem as its primary source.

Current source scope:

```text
Form 10-K
SEC submissions metadata
SEC XBRL company facts
```

The system retains provenance such as:

```text
CIK
ticker
accession number
filing date
report date
source URL
content hash
fiscal metadata
```

10-Q ingestion is intentionally outside the current Phase 1 scope.

Architecture decision:

```text
docs/adr/001-sec-edgar-as-primary-data-source.md
```

---

# SEC Ingestion

The ingestion pipeline is designed to be deterministic and idempotent.

Important properties include:

- SEC-compliant User-Agent behavior;
- global request throttling;
- retries;
- local caching;
- deterministic filing selection;
- accession-based identity;
- content hashing;
- cache-integrity validation;
- atomic persistence;
- deterministic section IDs;
- deterministic chunk IDs;
- deterministic financial-fact IDs;
- bounded company concurrency;
- deterministic partitioning;
- reproducible processed artifacts.

Current scale configuration includes:

```text
companies:            25
company workers:       4
SEC request limit:     5 requests / second
chunk size:          500
chunk overlap:        75
```

---

# SEC Parser

SEC filings are structurally inconsistent.

The parser therefore uses deterministic extraction rules rather than relying on an LLM.

Current narrative sections focus on:

```text
Item 1  - Business
Item 1A - Risk Factors
Item 7  - Management's Discussion and Analysis
Item 7A - Quantitative and Qualitative Disclosures About Market Risk
```

The parser supports:

```text
canonical Item headings
heading normalization
section-boundary rules
minimum section lengths
section-specific maximum lengths
deterministic semantic-heading fallback
```

Oversized candidates are rejected rather than silently truncated.

---

# Retrieval

FinGraph Agent maintains three retrieval methods.

## Sparse

BM25-style sparse retrieval in Qdrant.

Useful for:

```text
exact accounting terminology
regulatory language
product names
filing-specific phrases
```

## Dense

Current dense representation:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

Useful for:

```text
semantic paraphrases
conceptual similarity
wording mismatch
```

## Hybrid

Sparse and dense rankings are combined using reciprocal-rank-style fusion.

Hybrid retrieval is available when lexical and semantic signals are both useful.

The system does not assume hybrid retrieval must always outperform its component retrievers.

Architecture decision:

```text
docs/adr/003-retrieval-strategy.md
```

---

# Retrieval Benchmark

The current retrieval benchmark uses:

```text
questions: 12
k:         5
```

Current release metrics:

```text
Precision@5
MRR
latency
```

NDCG@5 and Recall@5 are not reported for this dataset because it does not contain explicit chunk-level qrels.

## v1.1 corrected results

The publication-visible current release report is:

```text
evals/reports/retrieval_scale_v1_1_report.json
```

It was regenerated with the corrected retrieval-metric implementation against the 12,666-chunk v1.1 index.

| Method | P@5 | MRR | Mean latency | P95 latency |
|---|---:|---:|---:|---:|
| Sparse | 0.633 | 0.792 | 9.782 ms | 35.278 ms |
| Dense | 0.583 | 0.729 | 18.894 ms | 65.213 ms |
| Hybrid | 0.633 | 0.686 | 17.204 ms | 34.962 ms |

For all three methods:

```text
NDCG@5:   N/A
Recall@5: N/A
```

The historical `retrieval_bootstrap_report.json` is retained as pre-fix baseline evidence. It must not be used as the current source for NDCG@5 or other corrected retrieval-quality claims.

Detailed evaluation:

```text
docs/evaluation.md
```

---

# Why There Is No Reranker

A reranker is intentionally not included in the production retrieval path.

The current benchmark does not demonstrate a ranking weakness that justifies another inference stage.

A reranker would add:

```text
latency
compute
dependencies
deployment complexity
failure modes
```

without current evidence of sufficient benefit.

Architecture decision:

```text
docs/adr/008-evaluation-driven-reranking.md
```

---

# Qdrant Indexing

Qdrant stores the narrative retrieval corpus.

Current scale architecture uses:

```text
company-bounded artifact loading
bounded dense embedding batches
bounded sparse encoding batches
bounded Qdrant upserts
deterministic UUID5 point IDs
rebuild mode
incremental ticker replacement
```

The implementation does not claim byte-stream JSON parsing.

The actual strategy is:

> Company-bounded artifact loading with bounded embedding and upsert batches.

---

# Qdrant Scale Benchmark

Current full rebuild:

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

Correctness check:

```text
12,666 chunks == 12,666 Qdrant points
```

The primary measured bottleneck is embedding plus upsert rather than artifact loading.

---

# Financial Knowledge Graph

Structured financial data is stored in Neo4j.

Current node types:

```text
Company
Filing
Section
FinancialFact
FiscalPeriod
```

Core relationships:

```text
(:Company)-[:FILED]->(:Filing)

(:Filing)-[:HAS_SECTION]->(:Section)

(:Company)-[:REPORTS]->(:FinancialFact)

(:Filing)-[:REPORTED_FACT]->(:FinancialFact)

(:FinancialFact)-[:FOR_PERIOD]->(:FiscalPeriod)
```

Architecture decision:

```text
docs/adr/002-dual-store-retrieval-and-graph.md
```

---

# Graph Integrity

The current v1.1 graph contains:

```text
Companies:       25
Filings:         198
Sections:        731
Financial facts: 115,649
Fiscal periods:  1,475
```

Post-load validation:

```text
count mismatches: 0
configured orphan checks: 0
audit passed:     true
```

This audit establishes expected-versus-actual entity-count agreement plus the configured orphan checks. It does **not** prove exact node-identity equality or exact relationship-endpoint equality against the processed corpus.

---

# Neo4j Scale Loading

The scale loader uses:

```text
company-bounded artifact loading
bounded UNWIND transactions
deterministic graph identities
stale-entity pruning
post-load audit
```

Current transaction batch size:

```text
1,000 rows
```

Measured v1.1 build:

```text
loaded rows:       118,078
load time:          60.000 seconds
total elapsed:      68.968 seconds
throughput:      1,967.965 rows / second
```

---

# Temporal Semantics

Financial filings contain multiple time concepts.

FinGraph Agent distinguishes:

```text
filing date
report date
XBRL period start
XBRL period end
SEC filing-context fiscal metadata
economic fiscal period
```

Financial comparisons use normalized fiscal-period semantics rather than assuming that every filing-level `fiscal_year` field represents the economic year of every fact inside the filing.

This prevents temporal comparison errors.

---

# Text-to-Cypher

Structured questions can use LLM-generated Cypher.

Generated Cypher is not executed directly.

The graph execution path includes:

```text
LLM generation
      |
      v
static validation
      |
      v
schema restrictions
      |
      v
parameter validation
      |
      v
required result bounds
      |
      v
Neo4j EXPLAIN
      |
      v
query timeout
      |
      v
read-routed execution
```

Architecture decision:

```text
docs/adr/007-safe-text-to-cypher.md
```

---

# Database Authorization Limitation

Application safeguards and database authorization are different controls.

The project does not claim that:

```text
static validation
+
read routing
```

is equivalent to:

```text
verified database-level least-privilege authorization
```

The local Neo4j Community environment does not provide the desired project-specific database-level reader-role boundary.

The managed graph deployment is documented conservatively unless a dedicated read-only serving identity is explicitly configured and verified.

---

# Self-Healing

Text-to-Cypher can fail because of:

```text
invalid properties
incorrect relationship direction
syntax errors
schema mismatch
planning failures
```

FinGraph Agent supports bounded repair.

Conceptually:

```text
attempt 1
   |
   v
validate / EXPLAIN / execute
   |
   +----------------+
   |                |
 success          failure
   |                |
   v                v
 return       structured feedback
                    |
                    v
                 repair
                    |
                    v
                 retry
```

Repair remains subject to the same safety controls.

It is not an unrestricted retry loop.

---

# Self-Healing Benchmark

Current injected-fault benchmark:

```text
faults:                 6
baseline success:       0 / 6
self-healing success:   6 / 6
average repair attempts: 1.000
```

Measured overhead:

```text
mean added latency:      5,862.905 ms
added input tokens:         19,718
added cached-input tokens:   2,816
added output tokens:         2,351
added total tokens:         22,069
instrumented added cost:     $0.004
```

These values describe the six-case injected-fault benchmark and are not a general estimate for all Graph QA traffic.

Source artifact:

```text
evals/reports/graph_repair_report.json
```

The project therefore measures both repair value and repair cost.

---

# LangGraph Orchestration

FinGraph Agent uses controlled LangGraph routing rather than an unconstrained agent swarm.

Current routes:

```text
VECTOR
GRAPH
HYBRID
```

## VECTOR

Narrative filing questions.

## GRAPH

Structured financial facts and deterministic analytical questions.

## HYBRID

Questions requiring both structured and narrative evidence.

Architecture decision:

```text
docs/adr/005-controlled-langgraph-orchestration.md
```

---

# Dependent Graph-to-Vector Retrieval

HYBRID execution can be dependent rather than merely parallel.

Example:

```text
question
   |
   v
graph lookup
   |
   v
resolve company / period / metric
   |
   v
vector retrieval constrained by graph context
   |
   v
combined evidence
```

This allows structured graph results to guide narrative retrieval.

---

# Routing Evaluation

Routing is evaluated independently from final answer quality.

The dedicated routing benchmark is:

```text
15 / 15 passed
```

Source artifacts:

```text
evals/datasets/agent_routing_bootstrap.json
evals/reports/agent_routing_bootstrap_report.json
```

This validates the route-selection layer separately from retrieval, graph execution, and synthesis. Route checks that appear inside evidence evaluations are tracked as evidence-pipeline assertions rather than being attributed to this dedicated routing benchmark.

---

# Deterministic Financial Calculations

The LLM does not perform authoritative financial arithmetic.

Supported deterministic patterns include:

```text
absolute change
percentage change
margin
CAGR
period comparison
```

The workflow is:

```text
financial facts
      |
      v
validated operands
      |
      v
Python / Decimal calculation
      |
      v
calculation evidence
      |
      v
grounded synthesis
```

Architecture decision:

```text
docs/adr/004-deterministic-financial-calculations.md
```

---

# Calculation Examples

Validated results include:

```text
Microsoft revenue change
2025 -> 2026
17.789%
```

```text
Microsoft revenue CAGR
2024 -> 2026
16.352%
```

```text
operating margin
46.781%
```

```text
net margin
40.305%
```

User-facing calculated decimals are rounded to three decimal places.

---

# Canonical Evidence

All evidence paths normalize into a shared downstream representation.

Conceptually:

```text
vector evidence ------+
                      |
graph evidence -------+--> canonical evidence
                      |
calculation evidence -+
```

Evidence retains metadata such as:

```text
evidence ID
source type
ticker
filing
section
period
text or structured payload
source URL
rank
score
provenance
```

The synthesis layer therefore does not need backend-specific logic.

---

# Grounded Answer Synthesis

Answer synthesis receives:

```text
question
selected route
canonical evidence
deterministic calculations
```

The synthesis contract requires:

- evidence grounding;
- temporal correctness;
- source-aware citations;
- no unsupported factual claims;
- explicit insufficient-evidence behavior;
- broad evidence coverage where appropriate;
- concise responses for narrow factual questions.

The model is responsible for language generation.

It is not the system of record.

---

# Resumable LLM Execution

External LLM execution can be interrupted by:

```text
rate limits
token quotas
temporary provider errors
network failures
application restarts
```

FinGraph Agent persists LLM request state in SQLite.

Logical requests receive deterministic identities based on inputs such as:

```text
task
prompt version
model
schema
logical payload
```

Persisted request states include:

```text
pending
running
succeeded
failed
```

Recoverable provider availability conditions can surface to API/evaluation callers as a **deferred outcome**, while the durable logical request remains resumable in `pending` state rather than storing `deferred` as a persisted SQLite status.

Architecture decision:

```text
docs/adr/006-resumable-llm-execution.md
```

---

# Provider Failure Semantics

Provider availability failures are distinct from quality failures.

For example:

```text
Groq rate limit
```

or:

```text
provider quota exhaustion
```

can be represented as:

```text
deferred
```

rather than incorrectly counted as:

```text
answer quality failure
```

This distinction is important for reliable evaluation.

---

# Answer Evaluation

The golden answer suite contains:

```text
160 cases
```

The current release answer-evaluation report contains:

```text
total:               160
completed:           160
passed:              160
quality_failed:        0
llm_quality_failed:    0
deferred:              0
execution_failed:      0
accuracy:          1.000
```

The current suite therefore passes 160 / 160 cases under the current evaluator, evaluation configuration, source, corpus, dataset, and output contract.

Successful current cases can be persisted and reused only while the evaluation fingerprint, case identity, and output contract remain unchanged.

The `160 / 160` result is the pass rate for the current versioned evaluation suite under the current evaluator and configuration. It is not a claim of universal financial-question correctness.

Detailed answer-evaluation methodology, fingerprint semantics, judge behavior, and report-reproduction commands are documented in:

```text
docs/evaluation.md
```

---

# Evidence Evaluation

Evidence grounding is evaluated separately from answer semantics.

Evaluation covers areas including:

```text
source provenance
entity attribution
citation grounding
route-specific evidence
multi-company evidence
hybrid evidence
```

Curated release-evidence reports are stored under:

```text
evals/reports/
```

Only the explicitly curated report artifacts referenced by the public documentation are committed. Ad hoc or newly generated benchmark output remains ignored until it is reviewed and intentionally promoted as release evidence.

---

# Evaluation

FinGraph Agent evaluates individual layers rather than relying only on end-to-end demos.

Current evaluation areas include:

```text
retrieval
routing
graph QA
graph integrity
self-healing
evidence grounding
deterministic calculations
answer quality
API load
scale behavior
```

Detailed results:

```text
docs/evaluation.md
```

---

# Evaluation Summary

| Area | Current result |
|---|---|
| Retrieval | Corrected 12,666-chunk v1.1 report current: sparse/dense/hybrid P@5 = 0.633/0.583/0.633; MRR = 0.792/0.729/0.686; NDCG@5 and Recall@5 = N/A without explicit chunk-level qrels |
| Routing | Dedicated routing benchmark: 15 / 15 passed |
| Graph integrity | Expected entity counts match; configured orphan checks = 0 |
| Graph self-healing | 6 / 6 repaired vs 0 / 6 baseline |
| Answer evaluation | Current release report: 160 / 160 passed; quality failures = 0; LLM quality failures = 0; deferred = 0; execution failures = 0; accuracy = 1.000 |
| Deterministic calculations | Validated |
| API load | 0 errors in recorded warm resumable/cache-enabled GRAPH / VECTOR / HYBRID benchmark |
| Qdrant scale | 12,666 points |
| Neo4j scale | 118,078 write rows |
| Persistent LLM state | Validated across Railway restart |
| Langfuse tracing | Validated |

---

# API

FinGraph Agent exposes a FastAPI service.

Current endpoints:

```text
GET  /health
GET  /ready
POST /ask
POST /ask/stream
```

`POST /ask` returns structured output including information such as:

```text
request ID
route
answer
citations
calculations
evidence
latency
```

Example synchronous request:

```bash
curl -s \
  -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <configured-key>' \
  -d '{"question":"What was Microsoft revenue in 2026?"}'
```

`X-API-Key` is required when an access key is configured and is mandatory in production configuration. `X-Request-ID` is optional; the API generates one when absent.

---

# SSE Streaming

`POST /ask/stream` uses Server-Sent Events.

Supported behavior includes:

```text
accepted events
heartbeat
completion
controlled errors
disconnect handling
cancellation
```

Example streaming request:

```bash
curl -N \
  -X POST http://localhost:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <configured-key>' \
  -d '{"question":"What risks did NVIDIA describe?"}'
```

For performance reporting, TTFT refers to:

```text
validated answer availability
```

rather than merely the initial accepted SSE event.

---

# Health and Readiness

`/health` checks process liveness.

`/ready` performs bounded dependency checks.

Current readiness dependencies include:

```text
Qdrant
Neo4j
LLM configuration
```

A healthy process is not automatically considered ready.

---

# API Load Benchmark

Formal load testing is performed separately for:

```text
GRAPH
VECTOR
HYBRID
```

The formal report records the **warm resumable/cache-enabled production path**. These numbers must not be interpreted as cold LLM-provider latency.

Current recorded synchronous results:

| Route | RPS | P50 | P95 | P99 | Errors |
|---|---:|---:|---:|---:|---:|
| GRAPH | 171.517 | 21 ms | 42 ms | 58 ms | 0 |
| VECTOR | 82.818 | 44 ms | 78 ms | 120 ms | 0 |
| HYBRID | 60.468 | 63 ms | 91 ms | 120 ms | 0 |

Source artifact:

```text
evals/reports/api_load_report.json
```

Keeping route-specific measurements prevents the different workloads from being hidden behind one aggregate latency number.

---

# Deployment

FinGraph Agent separates data-build workflows from serving.

```text
Data/build plane:
SEC ingestion -> processed publication -> Qdrant indexing -> Neo4j loading

Serving plane:
FastAPI -> LangGraph -> configured Qdrant / Neo4j / Groq -> grounded response
```

SEC ingestion, processed-data publication, Qdrant indexing, and Neo4j loading are operator-run workflows. FastAPI startup connects to already-populated stores; API requests never ingest filings, rebuild indexes, or load the graph.

FinGraph Agent supports both local production-style and managed deployments.

Detailed deployment documentation:

```text
docs/deployment.md
```

---

# Local Deployment

Local Docker Compose topology:

```text
Docker Compose
 |
 +-- fingraph-api
 |
 +-- fingraph-qdrant
 |
 +-- fingraph-neo4j
```

Pinned database images:

```text
Qdrant: qdrant/qdrant:v1.19.1
Neo4j:  neo4j:2026.09.0
```

---

# Managed Deployment

Current managed components:

```text
API              Railway
Vector database  Qdrant Cloud
Graph database   Neo4j Aura
LLM provider     Groq
LLM model        openai/gpt-oss-120b
Observability    Langfuse
LLM state        persistent SQLite
```

Current public API:

```text
https://fingraph-agent-api-production.up.railway.app
```

The managed deployment has been validated for:

```text
health
readiness
GRAPH
VECTOR
HYBRID
SSE
```

## Local vs managed component placement

| Component | Local development | Railway deployment |
|---|---|---|
| FastAPI / LangGraph | Local process or Compose container | Railway container |
| Qdrant | Local or configured remote endpoint | Qdrant Cloud |
| Neo4j | Local or configured remote endpoint | Neo4j Aura |
| Groq | External API | External API |
| SQLite LLM state | Local file | Railway `/data` volume |
| Langfuse | Optional external service | Optional external service |
| SEC ingestion | Operator-run workflow | Not part of request handling |
| Evaluation | Operator-run workflow | Can target configured remote services/API |
| Load testing | Local load generator | Can target deployed API URL |

Groq remains an external API in both local and Railway modes. Qdrant and Neo4j are configuration-driven: local development can use Compose-hosted services or configured remote endpoints, while the documented Railway topology uses Qdrant Cloud and Neo4j Aura.

---

# Persistent Production State

The Railway deployment mounts persistent storage at:

```text
/data
```

Current LLM-state database:

```text
/data/fingraph_runs.db
```

State persistence has been validated across a Railway application restart.

---

# Railway Runtime Identity

The Docker image declares a non-root application user.

The current Railway deployment uses:

```text
RAILWAY_RUN_UID=0
```

because of mounted-volume ownership requirements.

The project therefore does **not** claim that the current Railway runtime executes as a non-root process.

---

# Observability

Langfuse is used for LLM tracing.

Trace context can include:

```text
request identity
task
selected route
model
provider status
latency
execution state
```

Provider-deferred execution has been validated in tracing.

Secrets should not be included in trace payloads.

---

# Scale Strategy

FinGraph Agent does not introduce distributed infrastructure merely to make the architecture appear larger.

Current scaling uses:

```text
SEC
 -> bounded company concurrency
 -> deterministic partitioning

Qdrant
 -> company-bounded loading
 -> bounded embedding batches
 -> bounded upserts

Neo4j
 -> company-bounded loading
 -> bounded UNWIND transactions
```

The current workload does not require:

```text
Spark
Ray
Kafka
Kubernetes
```

Architecture decision:

```text
docs/adr/009-bounded-scale-execution.md
```

---

# SEC Scale Benchmark

Corrected cached 25-company regeneration:

```text
companies:             25
filings:              198
sections:             731
chunks:            12,666
financial facts:  115,649
```

Measured runtime:

```text
elapsed:               99.637 seconds
filings per minute:   119.233
companies per second:   0.251
filings downloaded:         0
companyfacts refreshed:     0
```

This is a cached regeneration benchmark, not fresh network throughput.

---

# CI

The repository contains a GitHub Actions workflow that performs:

```text
checkout
setup uv
install Python
uv sync --locked --dev
Ruff lint
Ruff format check
mypy
pytest
Docker build
```

Triggers:

```text
push to main
pull requests
```

The workflow implementation exists.

Hosted GitHub execution is only considered validated after the final repository push and successful remote workflow run.

---

# Continuous Deployment

Current development deployment uses:

```text
railway up
```

The intended release flow is:

```text
GitHub push
      |
      v
GitHub Actions
      |
      v
Railway deployment
      |
      v
public smoke validation
```

GitHub-to-Railway continuous deployment is not considered release-complete until this path is connected and successfully validated.

---

# Development Environment

Primary development environment:

```text
macOS
Apple Silicon
Python 3.12
uv
VS Code
Docker
Docker Compose
```

The project intentionally uses:

```text
scripts
modules
tests
CLI workflows
```

instead of notebooks.

---

# Setup

## 1. Clone

```bash
git clone <repository-url>
cd fingraph-agent
```

## 2. Install dependencies

```bash
uv sync --locked --dev
```

## 3. Configure environment

FinGraph Agent uses explicit private environment profiles:

```text
.env.dev   local development
.env.prod  managed / production configuration
```

The public `.env.example` documents both profiles.

Create the local development profile:

```bash
cp .env.example .env.dev
```

Configure the credentials and service settings required by the workflows you plan to run.

At minimum, graph-backed local API execution requires a local Neo4j password, and LLM-backed workflows require a Groq API key.

SEC ingestion additionally requires `FINGRAPH_SEC_USER_AGENT` to contain a compliant identifying User-Agent value for SEC EDGAR access.

Before running local CLI workflows directly from the shell, load the development profile:

```bash
set -a
source .env.dev
set +a
```

For intentional local execution against managed production services, use the private production profile instead:

```bash
set -a
source .env.prod
set +a
```

`.env.prod` is not synchronized automatically to Railway. Railway production variables are configured independently in the Railway service.

Never commit `.env.dev` or `.env.prod`.

## 4. Start local infrastructure

Docker Compose uses `.env.dev` for the API service and Compose interpolation:

```bash
docker compose --env-file .env.dev up -d
```

## 5. Check services

```bash
docker compose ps
```

## 6. Check API health

```bash
curl -s http://localhost:8000/health
```

## 7. Check readiness

```bash
curl -s http://localhost:8000/ready
```

---

# Important CLI Workflows

## Local verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src scripts tests
uv run python -m compileall -q src scripts tests load_tests
uv run pytest -q
uv lock --check
```

## SEC ingestion

Development/default configuration:

```bash
uv run python scripts/ingest_sec.py
```

Full v1.1 scale configuration:

```bash
uv run python scripts/ingest_sec.py --config configs/scale.toml
```

SEC ingestion requires `FINGRAPH_SEC_USER_AGENT` and network access to SEC EDGAR.

---

## Build retrieval index

The indexing CLI supports rebuild and incremental modes.

Example full rebuild:

```bash
uv run python scripts/build_retrieval_index.py --mode rebuild --batch-size 64
```

Example incremental replacement:

```bash
uv run python scripts/build_retrieval_index.py --mode incremental --maintenance-mode --batch-size 64 --tickers AAPL
```

Incremental ticker replacement is an offline-maintenance operation. Concurrent API serving must be stopped before running it because replacement is rollback-safe but not read-atomic for live readers.

---

## Build graph

```bash
uv run python scripts/build_graph.py --batch-size 1000 --prune-stale
```

---

## Retrieval evaluation

```bash
uv run python scripts/evaluate_retrieval.py --dataset evals/datasets/retrieval_bootstrap.json --report evals/reports/retrieval_scale_v1_1_report.json --k 5
```

---

## Run API

```bash
uv run python scripts/run_api.py
```

---

# Repository Layout

```text
fingraph-agent/
|
+-- configs/
|
+-- data/
|   +-- raw/
|   +-- processed/
|
+-- docs/
|   +-- architecture.md
|   +-- evaluation.md
|   +-- deployment.md
|   +-- adr/
|
+-- evals/
|   +-- datasets/
|   +-- reports/
|
+-- scripts/
|
+-- src/
|   +-- fingraph/
|
+-- tests/
|
+-- .github/
|   +-- workflows/
|
+-- Dockerfile
+-- docker-compose.yml
+-- pyproject.toml
+-- uv.lock
+-- README.md
```

---

# Architecture Decision Records

Architecture decisions are documented under:

```text
docs/adr/
```

Current ADRs:

1. `001-sec-edgar-as-primary-data-source.md`
2. `002-dual-store-retrieval-and-graph.md`
3. `003-retrieval-strategy.md`
4. `004-deterministic-financial-calculations.md`
5. `005-controlled-langgraph-orchestration.md`
6. `006-resumable-llm-execution.md`
7. `007-safe-text-to-cypher.md`
8. `008-evaluation-driven-reranking.md`
9. `009-bounded-scale-execution.md`

---

# Repository Principles

## Evidence before generation

The LLM is not the system of record.

Answers should be grounded in retrieved or structured evidence.

---

## Determinism where possible

Use deterministic behavior for:

```text
IDs
financial calculations
persistence
partitioning
validation
state transitions
```

where probabilistic generation provides no benefit.

---

## Bounded autonomy

Agentic behavior should remain constrained by:

```text
routes
schemas
validation
timeouts
result bounds
bounded retries
bounded repair
```

---

## Evaluation before complexity

Additional architectural components should demonstrate measurable value before being added.

This principle applies to:

```text
rerankers
distributed infrastructure
extra agents
larger models
additional retrieval stages
```

---

## Preserve provenance

Answers should remain traceable to:

```text
SEC filing
company
section
period
structured fact
source URL
```

where applicable.

---

## Explicit failure states

The system should distinguish:

```text
insufficient evidence
quality failure
execution failure
provider defer
dependency failure
```

rather than fabricating a successful response.

---

## No notebooks

Ingestion, experiments, evaluation, debugging, and validation are implemented through:

```text
modules
scripts
tests
CLI tools
```

rather than Jupyter notebooks.

---

# Known Limitations

Current limitations include:

- only SEC 10-K filings are in scope;
- the retrieval benchmark contains 12 questions and is primarily a regression benchmark;
- explicit chunk-level qrels are not available for the current retrieval benchmark, so NDCG@K and Recall@K are not reported;
- industry metadata is intentionally not populated without a reliable source-backed mapping;
- no reranker is currently justified by evaluation;
- the project does not claim a verified database-level least-privilege graph serving identity in every deployment;
- the Railway runtime currently runs as root because of mounted-volume ownership requirements;
- LLM serving depends on an external provider;
- the current scale does not justify distributed data-processing infrastructure;
- Neo4j stale-entity pruning currently executes one prune transaction per entity label rather than bounded delete batches;
- graph integrity validation currently proves configured entity-count and orphan invariants, not exact identity or relationship-endpoint equivalence;
- hosted GitHub CI still requires final release-push validation;
- GitHub-to-Railway CD still requires final release integration validation.

These limitations are documented explicitly rather than hidden behind production claims.

---

# Industry Metadata

The domain model supports metadata such as:

```text
SIC
industry
```

Industry values are not fabricated.

Industry enrichment remains deferred until a reliable source-backed mapping is selected and validated.

---

# Phase 1 Release Gates

Pre-commit implementation and verification are complete. Remaining items are post-commit/public-release gates.

```text
[x] final documentation complete
[x] final Ruff check passes
[x] final Ruff format check passes
[x] final mypy passes
[x] final pytest suite passes
[x] final Docker build passes
[x] final verification-only repository audit completed
[x] release-blocking findings resolved
[x] final quality suite rerun
[x] exposed development credentials rotated
[ ] first/final Phase 1 commit created
[ ] repository pushed to GitHub
[ ] hosted GitHub Actions CI passes
[ ] GitHub-to-Railway deployment connected
[ ] GitHub-triggered deployment succeeds
[ ] final public smoke tests pass
```

Only completed gates should be represented as completed in public release documentation.

---

# Future Extensions

Potential future work includes:

- SEC 10-Q ingestion;
- larger company universes;
- richer explicit retrieval relevance judgments;
- Recall@K once relevance sets support it;
- reranking if controlled evaluation demonstrates value;
- authoritative industry enrichment;
- verified least-privilege graph serving identity;
- additional financial concepts;
- additional deterministic analytical operations;
- more extensive graph-repair benchmarks;
- incremental SEC update scheduling;
- distributed processing only when measured workload requires it.

These are future extensions, not current implementation claims.

---

# License

FinGraph Agent is licensed under the **Apache License 2.0**.

See the root `LICENSE` file for the full license text.