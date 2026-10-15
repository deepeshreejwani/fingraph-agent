# FinGraph Agent Architecture

## 1. Purpose

FinGraph Agent is a production-oriented financial intelligence system over SEC filings.

Its architecture separates concerns that are often collapsed into a single RAG pipeline:

1. source ingestion;
2. narrative retrieval;
3. structured financial reasoning;
4. deterministic calculations;
5. controlled agentic orchestration;
6. grounded answer synthesis;
7. evaluation and observability;
8. production serving.

The workflow uses deterministic control where possible and does not treat the LLM as the authoritative source for financial data, arithmetic, retrieval state, execution state, routing policy, or provenance.

The system is designed around a central rule:

> Retrieve, route, validate, and compute deterministically where possible; use the LLM for bounded query generation, repair, interpretation, and grounded synthesis where probabilistic generation adds value.

---

## 2. System overview

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
                    |
                    v
                 Railway
```

The architecture is intentionally split into two operational planes:

```text
Data/build plane:
SEC ingestion -> processed publication -> Qdrant indexing -> Neo4j loading

Serving plane:
FastAPI -> LangGraph -> configured Qdrant / Neo4j / Groq -> grounded response
```

SEC ingestion, processed-data publication, retrieval indexing, and graph loading are operator-run data-build workflows. They are not invoked by FastAPI startup or request handling. The serving process connects to already-populated stores.

---

## 3. Data acquisition

The SEC EDGAR ecosystem is the primary data source.

The current ingestion scope includes:

```text
Form 10-K
SEC submissions metadata
SEC company facts / XBRL
```

Each company is identified using SEC identifiers including CIK and ticker.

Each filing is identified using its SEC accession number.

The ingestion layer retains source-level metadata rather than reducing filings into anonymous text.

Important retained provenance includes:

```text
CIK
ticker
accession number
filing date
report date
form type
source URL
content hash
fiscal metadata
```

Only annual 10-K filings are currently in scope.

10-Q ingestion is intentionally deferred.

---

## 4. Deterministic ingestion

The ingestion layer is designed for reproducibility and idempotency.

Important deterministic identities include:

```text
company identity
filing accession identity
section ID
chunk ID
financial-fact ID
fiscal-period ID
content hash
```

Repeated ingestion should converge toward the same logical corpus rather than introduce duplicate entities.

Persistence uses atomic writes where partially written state could otherwise corrupt processed artifacts.

Cached filing content is validated before reuse.

The pipeline preserves both raw and normalized representations where appropriate.

---

## 5. SEC request control

SEC ingestion uses bounded request behavior.

The ingestion layer supports:

```text
identifiable SEC User-Agent
global request throttling
bounded request rate
retry handling
local caching
deterministic filing selection
```

The current scale configuration uses:

```text
maximum SEC request rate: 5 requests / second
company workers:          4
```

The global throttle is shared across workers so concurrent ingestion does not independently exceed the configured SEC request rate.

---

## 6. Scale-oriented ingestion

The v1.1 ingestion architecture introduces bounded company-level concurrency without adding a distributed processing framework.

The scale path uses:

```text
canonical company list
        |
        v
deterministic partitioning
        |
        v
bounded ThreadPoolExecutor
        |
        v
per-company ingestion
        |
        v
batched manifest staging
        |
        v
deterministic processed artifacts
```

Company partitioning is deterministic.

The implementation uses sorted round-robin partitioning to make partition membership reproducible.

Manifest writes are staged and flushed in bounded batches rather than requiring every worker operation to perform an independent persistence cycle.

The architecture intentionally does not introduce Spark, Ray, Kafka, or another distributed execution framework for the current corpus size.

---

## 7. Narrative processing

10-K HTML filings are normalized and divided into filing sections.

The current narrative corpus focuses primarily on:

```text
Item 1  — Business
Item 1A — Risk Factors
Item 7  — Management's Discussion and Analysis
Item 7A — Quantitative and Qualitative Disclosures About Market Risk
```

The parser handles filing-layout variation using deterministic section-detection rules.

It does not use an LLM for section extraction.

Candidate sections are validated using minimum and maximum section-length constraints.

Oversized candidates are rejected rather than silently truncated.

Sections are transformed into deterministic overlapping chunks.

Current chunk configuration:

```text
chunk size: 500
overlap:     75
```

Chunks preserve metadata including:

```text
filing
section
company
ticker
fiscal year
source URL
content hash
```

---

## 8. Narrative parser hardening

Real SEC filings do not follow one uniform HTML structure.

The parser therefore supports:

```text
canonical Item-heading extraction
heading normalization
boundary validation
minimum section length
section-specific maximum lengths
deterministic semantic-heading fallback
```

Fallback logic is only used when a valid canonical candidate is unavailable.

The parser does not use:

```text
LLM section classification
ticker-specific hardcoded extraction paths
fuzzy generative extraction
silent oversized truncation
```

This keeps the ingestion pipeline deterministic and testable.

---

## 9. Vector store

Qdrant stores narrative filing chunks.

The collection contains named representations for:

```text
dense
bm25
```

Current dense encoder:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

Payload metadata supports filtering by fields such as:

```text
ticker
fiscal year
section name
filing ID
```

Qdrant supports:

```text
sparse retrieval
dense retrieval
hybrid retrieval
metadata filtering
```

---

## 10. Scalable Qdrant indexing

The v1.1 indexing path avoids loading the entire logical corpus into one unbounded indexing operation.

The indexing flow is:

```text
processed company artifacts
          |
          v
company-bounded loading
          |
          v
bounded chunk batches
          |
          +------------------+
          |                  |
          v                  v
     dense encoding      sparse encoding
          |                  |
          +--------+---------+
                   |
                   v
             Qdrant upsert
```

Important properties:

```text
deterministic UUID5 point IDs
bounded embedding batches
bounded upload batches
company-bounded artifact loading
rebuild mode
incremental ticker replacement mode
```

Incremental mode replaces selected ticker partitions rather than appending duplicate points.

The implementation does not claim byte-stream JSON parsing.

The scale strategy is specifically:

> company-bounded artifact loading plus bounded dense/sparse embedding and Qdrant upsert batches.

---

## 11. Retrieval architecture

Retrieval is not treated as one mandatory pipeline.

Three strategies are maintained.

### 11.1 Sparse

BM25-style sparse retrieval.

Strengths include:

```text
exact terminology
filing-specific phrases
accounting vocabulary
regulatory terms
named products
named programs
```

### 11.2 Dense

Embedding-similarity retrieval.

Strengths include:

```text
semantic paraphrases
conceptual similarity
wording mismatch
```

### 11.3 Hybrid

Hybrid retrieval combines sparse and dense rankings using reciprocal-rank-style fusion.

Hybrid retrieval is useful when neither lexical nor semantic representation is reliably sufficient on its own.

The system does not assume hybrid retrieval must outperform sparse or dense retrieval on every benchmark.

---

## 12. Retrieval evaluation philosophy

Retrieval is evaluated independently from answer generation.

The current benchmark uses:

```text
questions: 12
k:         5
methods:   sparse / dense / hybrid
```

Current reported quality metrics include:

```text
Precision@K
MRR
```

Measured latency statistics include:

```text
mean
P95
```

NDCG@K and Recall@K are not currently reported as numeric release metrics because the bootstrap retrieval dataset does not contain explicit chunk-level qrels.

A reranker is intentionally absent.

The architecture follows the rule:

> Additional retrieval complexity must demonstrate measurable benefit before being added.

---

## 13. Retrieval scale validation

The retrieval corpus was expanded from:

```text
3,005 chunks
```

to:

```text
12,666 chunks
```

while preserving the same 12-question dataset and `k=5` benchmark shape.

The corrected publication-visible v1.1 report is:

```text
evals/reports/retrieval_scale_v1_1_report.json
```

Current quality results:

| Method | P@5 | MRR | NDCG@5 | Recall@5 |
|---|---:|---:|---:|---:|
| Sparse | 0.633 | 0.792 | N/A | N/A |
| Dense | 0.583 | 0.729 | N/A | N/A |
| Hybrid | 0.633 | 0.686 | N/A | N/A |

Current latency results:

| Method | Mean latency | P95 latency |
|---|---:|---:|
| Sparse | 9.782 ms | 35.278 ms |
| Dense | 18.894 ms | 65.213 ms |
| Hybrid | 17.204 ms | 34.962 ms |

The historical `retrieval_bootstrap_report.json` remains pre-fix baseline evidence only. NDCG@5 and Recall@5 are `N/A` for the current benchmark because explicit chunk-level qrels are absent.

---

## 14. Knowledge graph

Neo4j stores structured financial information.

Current node types:

```text
Company
Filing
Section
FinancialFact
FiscalPeriod
```

Current relationships:

```text
(:Company)-[:FILED]->(:Filing)

(:Filing)-[:HAS_SECTION]->(:Section)

(:Company)-[:REPORTS]->(:FinancialFact)

(:Filing)-[:REPORTED_FACT]->(:FinancialFact)

(:FinancialFact)-[:FOR_PERIOD]->(:FiscalPeriod)
```

The graph separates structured financial facts from narrative retrieval.

It preserves relationships among:

```text
company
filing
section
financial fact
economic fiscal period
```

---

## 15. Industry metadata

The Company model supports metadata fields that can include:

```text
SIC
industry
```

Industry enrichment is not currently populated unless backed by a reliable source.

FinGraph Agent does not invent or infer industry classifications merely to fill the field.

Industry enrichment remains deferred until a source-backed mapping is selected and validated.

---

## 16. Temporal model

SEC financial data contains several overlapping notions of time.

The architecture distinguishes:

```text
filing_date
report_date
XBRL start_date
XBRL end_date
SEC filing-context fiscal metadata
economic fiscal period
```

Financial comparisons use normalized economic period semantics rather than assuming that a filing-level fiscal-year field always represents the economic year of every fact contained in the filing.

This distinction is important because one filing can contain financial values for multiple periods.

---

## 17. Fiscal periods

Financial facts are connected to normalized `FiscalPeriod` nodes.

A fiscal period can retain information such as:

```text
period year
start date
end date
instant / duration semantics
```

Fiscal-period identity is deterministic.

This permits longitudinal financial queries without forcing the LLM to infer temporal relationships from filing prose.

---

## 18. Scalable Neo4j loading

The v1.1 graph loader uses company-bounded artifact loading and bounded Neo4j write transactions.

The graph loading path is:

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
bounded UNWIND batches
          |
          v
Neo4j transactions
```

The storage layer consumes an iterable and divides writes into bounded transactions.

Current validated transaction batch size:

```text
1,000 rows
```

Existing canonical Cypher, fiscal-period identity logic, graph auditing, and stale-entity pruning are reused.

The scale layer does not duplicate the graph schema or introduce a parallel graph model.

---

## 19. Graph build validation

After loading, graph integrity is checked against the processed corpus.

Validation currently includes:

```text
expected entity counts
actual entity counts
count mismatches
configured orphan checks
```

The current v1.1 audit reports:

```text
Companies:       25
Filings:         198
Sections:        731
Financial facts: 115,649
Fiscal periods:  1,475

Count mismatches: 0
Configured orphan checks: 0
```

This establishes entity-count agreement plus the configured orphan checks. It does **not** establish exact node-identity equality or exact relationship-endpoint equality against the processed corpus.

Stale-entity pruning is a separate maintenance step. The current implementation issues one prune operation per graph entity class; deletion work is therefore not transaction-batched within a label and should not be described as bounded pruning.

---

## 20. Graph QA

Structured questions can enter the graph QA path.

Example:

```text
What was Microsoft's revenue in 2026?
```

The graph path is:

```text
Question
   |
   v
LLM Text-to-Cypher
   |
   v
Static validator
   |
   v
Neo4j EXPLAIN
   |
   v
bounded execution
   |
   v
Structured rows
```

The system constrains generated queries to the supported graph schema and read-oriented query patterns.

---

## 21. Text-to-Cypher safety

LLM-generated Cypher is never sent directly to unrestricted execution.

The application applies layered safeguards:

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

The validator rejects unsupported or potentially mutating operations.

Generated queries are bounded by explicit result limits.

---

## 22. Database authorization boundary

Application-level safeguards and database-level authorization are different controls.

The local Neo4j Community environment does not provide the desired project-specific database-level read-only serving identity.

The current architecture therefore does not claim that application safeguards are equivalent to a true database-level reader role.

The managed Neo4j deployment is treated with the same conservative documentation rule unless a dedicated least-privilege serving identity is explicitly configured and verified.

Current guaranteed controls are application-level defense in depth.

---

## 23. Graph self-healing

Generated Cypher can fail because of:

```text
invalid properties
relationship-direction mistakes
syntax errors
unsupported schema references
planning failures
```

The system supports bounded repair.

Conceptually:

```text
attempt 1
   |
   +---- success -----------------> return
   |
   +---- failure
             |
             v
      structured feedback
             |
             v
         repair LLM
             |
             v
         attempt 2
             |
        +----+----+
        |         |
     success     stop
```

Repair is bounded.

The system does not allow an unconstrained autonomous retry loop.

---

## 24. Self-healing benchmark

A dedicated fault-injection benchmark validates whether repair adds measurable value.

Current benchmark:

```text
injected faults:       6
baseline success:      0 / 6
self-healing success:  6 / 6
average repair attempts: 1.000
```

Measured repair overhead is tracked separately from correctness.

This benchmark distinguishes genuine repair capability from ordinary successful Text-to-Cypher generation.

---

## 25. LangGraph orchestration

The system exposes three logical routes:

```text
VECTOR
GRAPH
HYBRID
```

The router determines which evidence sources are required.

### VECTOR

Used primarily for narrative filing questions.

Example:

```text
What risks did NVIDIA describe in its filing?
```

### GRAPH

Used for structured financial facts and deterministic analytical questions.

Example:

```text
What was Microsoft's revenue in 2026?
```

### HYBRID

Used when the question requires both structured numerical evidence and narrative context.

Example:

```text
How did Microsoft's revenue change and what factors drove the change?
```

The graph path provides structured evidence.

The vector path provides filing narrative.

Both are normalized into a shared evidence representation before synthesis.

---

## 26. Controlled agentic design

FinGraph Agent deliberately avoids an unconstrained multi-agent swarm.

The workflow favors:

```text
explicit routes
typed state
bounded tool execution
bounded repair
deterministic calculations
observable intermediate evidence
evaluation at individual layers
```

LangGraph provides orchestration and state transitions without requiring independent agents to negotiate or recursively delegate tasks.

The design prioritizes inspectability over agent count.

---

## 27. Dependent graph-to-vector flow

Some hybrid questions require graph output to inform narrative retrieval.

The architecture supports dependent execution where structured evidence can narrow or guide the subsequent vector search.

Conceptually:

```text
Question
   |
   v
Graph query
   |
   v
structured entities / periods
   |
   v
vector retrieval constrained by graph context
   |
   v
combined evidence
```

This differs from merely running two independent searches and concatenating the results.

---

## 28. Multi-company evidence

The agent supports questions involving more than one company.

Graph evidence can include multiple companies in one structured result.

Narrative retrieval supports result diversification so one company's chunks do not automatically dominate a multi-company response.

Evidence retains entity-level provenance so citations can be associated with the company and filing that produced the claim.

---

## 29. Canonical evidence model

Downstream answer synthesis should not need to understand every storage backend.

Retrieved material is normalized into canonical evidence objects containing information such as:

```text
evidence ID
source type
ticker
filing
section
text or structured payload
source URL
score
rank
provenance
```

This permits vector evidence, graph evidence, and calculation evidence to coexist in one answer contract.

---

## 30. Deterministic calculations

The LLM is not responsible for authoritative arithmetic.

For financial calculations:

```text
structured financial facts
          |
          v
Decimal-based Python calculation
          |
          v
structured calculation object
          |
          v
grounded answer synthesis
```

Supported calculation patterns include:

```text
absolute change
percentage change
margin
CAGR
period comparison
```

Calculation results retain their operands and period context.

User-facing decimal values are rounded to three decimal places.

---

## 31. Calculation validation

Representative validated calculations include:

```text
Microsoft revenue growth 2025 -> 2026: 17.789%
Microsoft revenue CAGR 2024 -> 2026:    16.352%
operating margin example:               46.781%
net margin example:                     40.305%
```

The calculations are produced deterministically rather than delegated to free-form LLM arithmetic.

---

## 32. Answer synthesis

The synthesis layer receives:

```text
question
route
canonical evidence
deterministic calculations
```

Its responsibility is language generation, not fact creation.

The synthesis contract requires:

```text
evidence grounding
temporal correctness
citation selection
no unsupported factual claims
explicit insufficient-evidence behavior
broad evidence coverage where appropriate
concise answers for narrow factual questions
```

---

## 33. Citation grounding

Citations are generated from the canonical evidence supplied to synthesis.

The answer layer does not create arbitrary citations from model memory.

Narrative evidence retains filing and section provenance.

Structured graph evidence retains the underlying company, period, filing, and fact context required for traceability.

The synthesis layer therefore operates over explicit source objects rather than anonymous text fragments.

---

## 34. Durable LLM execution

LLM calls are persisted in SQLite.

Each logical request receives a deterministic identity based on relevant request inputs such as:

```text
task
prompt version
model
schema
logical payload
```

Persisted execution state supports:

```text
resume
deduplication
quota recovery
failure analysis
reproducibility
```

Durable SQLite execution states are:

```text
pending
running
succeeded
failed
```

`deferred` is an API or evaluation outcome, not a persisted SQLite status. A recoverable provider limitation returns the durable logical request to `pending` so the same request can resume later.

Before an external provider call, a worker atomically claims the logical request with an execution-owner ID and lease expiry. The active owner can renew the lease while work is in progress. Another worker may reclaim the request only after the lease expires, which prevents simultaneous duplicate provider calls while still allowing recovery from abandoned `running` work.

The durable execution model therefore combines:

```text
deterministic request identity
+
persistent state
+
execution ownership
+
expiring leases
+
bounded retry / provider-failure handling
```

See `docs/adr/006-resumable-llm-execution.md` for the execution-state decision and resumability contract.

---

## 35. Provider failure semantics

Provider execution failures and answer-quality failures are treated differently.

Examples of provider execution failures include:

```text
rate limits
provider quota exhaustion
temporary upstream errors
network failures
```

Recoverable provider quota or rate conditions can surface outward as deferred work instead of being incorrectly reported as an answer-quality failure.

For durable execution, that recoverable condition returns the logical request to `pending`; `deferred` is not stored as a fifth SQLite state.

Evaluation logic can therefore resume without repeating already successful cases.

---

## 36. API architecture

FastAPI exposes the application.

Current endpoints:

```text
GET  /health
GET  /ready
POST /ask
POST /ask/stream
```

The response contract contains structured answer metadata rather than returning only free text.

Response information can include:

```text
request ID
selected route
answer
citations
calculations
evidence
latency metadata
```

---

## 37. SSE streaming

`POST /ask/stream` exposes Server-Sent Events.

The streaming path supports:

```text
accepted events
heartbeat
completion events
controlled errors
disconnect handling
cancellation
```

For performance reporting, TTFT is defined using validated answer availability rather than merely the initial accepted event.

This prevents a connection-acceptance event from being incorrectly presented as useful model-output latency.

---

## 38. Dependency readiness

The API distinguishes process health from dependency readiness.

```text
/health
```

indicates that the API process is functioning.

```text
/ready
```

performs bounded dependency checks.

Current readiness dependencies include:

```text
Qdrant
Neo4j
LLM configuration
```

Readiness checks are time-bounded so an unavailable dependency cannot block indefinitely.

---

## 39. Local deployment topology

The production-style local environment uses Docker Compose.

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

Persistent database state is mounted outside ephemeral containers.

The API image defines a non-root runtime user.

---

## 40. Managed deployment topology

The deployed environment separates application compute from managed persistence.

```text
                        Internet
                           |
                           v
                       Railway
                   FinGraph API
                    /    |    \
                   /     |     \
                  v      v      v
              Qdrant   Neo4j   Groq
               Cloud    Aura   LLM
                  \      |      /
                   \     |     /
                    +----+----+
                         |
                         v
                      Langfuse
```

Current managed components:

```text
API deployment:       Railway
vector database:      Qdrant Cloud
graph database:       Neo4j Aura
LLM provider:         Groq
LLM model:            openai/gpt-oss-120b
observability:        Langfuse
durable LLM state:    SQLite on persistent Railway volume
```

---

## 41. Railway runtime

Railway provides the public API runtime.

A persistent volume is mounted for durable application state.

The deployed SQLite request-state database is stored on the persistent volume so request/evaluation state survives application restarts.

The Railway runtime currently executes with elevated container privileges because of mounted-volume ownership requirements.

The project therefore does not claim that the Railway deployment currently runs as a non-root process.

This is distinct from the Docker image itself, which declares a non-root application user.

---

## 42. Observability

Langfuse is used for LLM tracing and execution visibility.

Tracing can capture information including:

```text
request identity
route
LLM interaction
provider failures
latency
execution status
```

Provider deferrals are observable rather than silently converted into generic internal errors.

Observability is treated as part of the production architecture rather than only as local debugging instrumentation.

---

## 43. Evaluation architecture

Evaluation is layered.

```text
                        End-to-end answers
                               ^
                               |
                       Answer evaluation
                               ^
                               |
              +----------------+----------------+
              |                                 |
        Evidence evaluation               Calculation tests
              ^                                 ^
              |                                 |
       Routing evaluation                 Graph QA evaluation
              ^                                 ^
              |                                 |
      Retrieval evaluation              Repair benchmark
              ^                                 ^
              |                                 |
         Store health                    Graph integrity
```

This makes failures diagnosable.

A bad answer can be traced to:

```text
retrieval
routing
graph generation
graph execution
temporal semantics
calculation
evidence grounding
synthesis
provider execution
```

rather than being treated as one opaque end-to-end failure.

---

## 44. Answer evaluation

The golden answer dataset contains:

```text
160 cases
```

The current release answer-evaluation report completed all 160 cases successfully:

```text
completed:           160
passed:              160
quality_failed:        0
llm_quality_failed:    0
deferred:              0
execution_failed:      0
accuracy:          1.000
```

The current suite therefore passes 160 / 160 cases under the current evaluator, evaluation configuration, source, corpus, dataset, and output contract.

The evaluation remains resumable: successful cases from a matching current fingerprint and case identity can be reused rather than unnecessarily repeating LLM calls.

---

## 45. API load testing

Formal load testing is performed separately for each logical route.

The formal concurrent report represents the **warm resumable/cache-enabled production path**. These measurements must not be interpreted as cold LLM-provider latency.

Recorded results:

| Route | RPS | P50 | P95 | P99 | Errors |
|---|---:|---:|---:|---:|---:|
| GRAPH | 171.517 | 21 ms | 42 ms | 58 ms | 0 |
| VECTOR | 82.818 | 44 ms | 78 ms | 120 ms | 0 |
| HYBRID | 60.468 | 63 ms | 91 ms | 120 ms | 0 |

Route-specific benchmarking avoids hiding the different cost profiles of graph, vector, and combined execution behind one aggregate number. The source artifact is `evals/reports/api_load_report.json`.

---

## 46. v1.1 corpus scale

The current validated corpus contains:

```text
Companies:        25
10-K filings:    198
Sections:        731
Chunks:       12,666
Facts:       115,649
Fiscal periods: 1,475
```

The company universe is:

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

## 47. Baseline-to-v1.1 growth

The original development corpus contained:

```text
Companies:         6
10-K filings:     48
Sections:        192
Chunks:        3,005
Facts:        25,976
Fiscal periods:   355
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

---

## 48. Scale benchmark: SEC ingestion

The corrected 25-company cached regeneration produced:

```text
companies:             25
filings:              198
sections:             731
chunks:            12,666
financial facts:  115,649
fiscal periods:      1,475
```

Measured execution:

```text
elapsed:              99.637 seconds
filings per minute:  119.233
companies per second:  0.251
filings downloaded:        0
companyfacts refreshed:    0
```

This measurement primarily represents deterministic regeneration from the locally cached source corpus rather than fresh network-download throughput.

---

## 49. Scale benchmark: Qdrant indexing

The full v1.1 Qdrant rebuild indexed:

```text
chunks:          12,666
points:          12,666
batches:            211
batch size:          64
dense dimensions:   384
```

Measured execution:

```text
artifact load:              0.807 seconds
embedding + upsert:      1563.251 seconds
total elapsed:           1566.280 seconds
throughput:                 8.102 chunks / second
```

The measured bottleneck is dense/sparse encoding plus Qdrant upsert rather than processed-artifact loading.

---

## 50. Scale benchmark: Neo4j loading

The full v1.1 graph load wrote:

```text
loaded rows:             118,078
transaction batch size:    1,000
```

Measured execution:

```text
schema:          1.315 seconds
companies:       0.138 seconds
filings/sections:6.935 seconds
facts/periods:  52.927 seconds
load:           60.000 seconds
prune:           4.627 seconds
audit:           2.435 seconds
total elapsed:  68.968 seconds
throughput:   1,967.965 rows / second
```

The post-load audit reported:

```text
count mismatches:         0
configured orphan checks: 0
audit passed:             true
```

These checks validate aggregate entity counts and the configured orphan predicates; they do not prove exact graph identity or relationship-endpoint equivalence.

---

## 51. Scale philosophy

FinGraph Agent does not introduce distributed infrastructure merely to make the system appear larger.

The current design intentionally avoids unnecessary use of:

```text
Spark
Ray
Kafka
Kubernetes
distributed orchestration frameworks
```

The observed workload is currently addressed through:

```text
bounded concurrency
deterministic partitioning
company-bounded loading
bounded embedding batches
bounded Qdrant upserts
bounded Neo4j transactions
idempotent execution
incremental replacement
```

This preserves architectural simplicity while providing measured scale behavior.

---

## 52. Incremental data-path philosophy

Scale support should not require rebuilding every downstream system for every future data change.

The architecture therefore favors deterministic partitions.

Examples include:

```text
company-bounded SEC artifacts
ticker-scoped Qdrant replacement
deterministic Neo4j entity identities
stale-entity pruning
resumable LLM execution
```

This provides a path toward incremental updates without requiring a streaming platform at the current workload scale.

---

## 53. CI architecture

The repository includes a GitHub Actions CI workflow.

The CI definition performs:

```text
checkout
Python / uv setup
locked dependency sync
Ruff lint
Ruff format check
mypy
pytest
Docker image build
```

The workflow is configured for:

```text
push to main
pull requests
```

Local CI-equivalent validation is part of the release process.

Hosted GitHub execution is only considered validated after the repository is pushed and the workflow succeeds remotely.

---

## 54. CD architecture

The current Railway deployment was created before the final repository release push.

Deployment during development can therefore be performed through the Railway CLI.

The intended release-state delivery path is:

```text
GitHub
   |
   v
GitHub CI
   |
   v
Railway deployment integration
```

GitHub-to-Railway continuous deployment is not considered complete until the final repository push is connected and validated.

---

## 55. Security boundaries

FinGraph Agent distinguishes between:

```text
application validation
database authorization
provider credentials
network transport
runtime identity
```

Application validation does not replace database authorization.

Secrets are supplied through environment configuration and must not be committed to source control.

Credentials exposed during development were rotated before the current release-candidate validation; publication-visible configuration retains blank or non-production credential values.

---

## 56. Development environment

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

The repository uses:

```text
scripts
modules
tests
CLI workflows
```

instead of notebooks.

Experiments and evaluations are designed to be reproducible from command-line entry points.

---

## 57. Known limitations

Current limitations include:

- only Form 10-K filings are ingested;
- explicit chunk-level qrels are not available for the current retrieval benchmark, so NDCG@K and Recall@K are not reported as numeric release metrics;
- no reranker is included because current evaluation does not justify it;
- industry metadata remains intentionally unpopulated where no reliable source-backed mapping has been selected;
- the application does not claim a verified database-level reader-role security boundary for every Neo4j deployment;
- LLM availability depends on an external provider;
- the Railway runtime currently uses elevated privileges because of persistent-volume ownership requirements;
- the current scale does not justify distributed data-processing infrastructure;
- GitHub-hosted CI and GitHub-to-Railway CD require final release-push validation.

These limitations are explicit rather than hidden behind production claims.

---

## 58. Architecture principles

### Evidence before generation

LLMs synthesize evidence but do not replace authoritative source and data layers.

### Determinism where possible

Identifiers, calculations, persistence, partitions, and structured transformations should be deterministic.

### Bounded autonomy

Agentic behavior is constrained through explicit routes, schemas, validation, retries, and repair limits.

### Evaluate before adding complexity

A more complicated architecture is not automatically a better architecture.

### Preserve provenance

Answers should remain traceable to SEC source material and structured source facts.

### Scale through measured bottlenecks

Scaling decisions should respond to measured workload characteristics rather than architecture fashion.

### Fail explicitly

Unsupported questions, insufficient evidence, provider quotas, dependency outages, and execution failures should result in controlled states rather than fabricated answers.

### Separate quality from availability

Provider execution failures must not be confused with retrieval, reasoning, or answer-quality failures.

### Keep the LLM out of deterministic work

Arithmetic, identity generation, persistence, schema enforcement, and other deterministic operations should remain outside free-form model generation whenever possible.