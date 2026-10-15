# FinGraph Agent Deployment

## 1. Purpose

This document describes the runtime, deployment, persistence, readiness, observability, and release architecture for FinGraph Agent.

The project supports two main runtime modes:

```text
local production-style deployment
```

and:

```text
managed public deployment
```

The logical application architecture remains the same across both environments.

---

# Local Runtime

## 2. Local topology

The local production-style environment uses Docker Compose.

```text
Docker Compose
 |
 +-- fingraph-api
 |
 +-- fingraph-qdrant
 |
 +-- fingraph-neo4j
```

The API depends on both persistence services.

The application should not be considered fully ready until required dependencies are available.

---

## 3. Local services

Current local services:

```text
fingraph-api
fingraph-qdrant
fingraph-neo4j
```

Pinned database images:

```text
Qdrant: qdrant/qdrant:v1.19.1
Neo4j:  neo4j:2026.09.0
```

Persistent state is mounted outside ephemeral container layers.

---

## 4. API container

The application is packaged as a Docker image.

The image installs the project using the locked Python environment and starts the FastAPI service.

The Docker image declares a non-root application user.

This applies to the image definition.

It should not be confused with the current Railway runtime identity, which has separate volume-ownership constraints.

---

## 5. Local startup

FinGraph Agent uses explicit private environment profiles:

```text
.env.dev   -> local development
.env.prod  -> managed / production configuration
```

The public `.env.example` documents both profiles. Private profiles must remain ignored by Git.

For the local Docker Compose stack, start from the development profile and configure the required local credentials:

```bash
cp .env.example .env.dev
```

Docker Compose uses `.env.dev` for the API service and for Compose interpolation:

```bash
docker compose --env-file .env.dev up -d
```

Service state can be inspected with:

```bash
docker compose ps
```

The API can then be checked through:

```bash
curl -s http://localhost:8000/health
```

and:

```bash
curl -s http://localhost:8000/ready
```

For a direct host run without Compose, first load the development profile into the current shell:

```bash
set -a
source .env.dev
set +a

uv run python scripts/run_api.py
```

A new terminal does not automatically inherit variables sourced in another terminal.

The API process is a serving runtime, not a data-build runtime.

```text
Data/build workflows:
SEC ingestion -> processed publication -> Qdrant indexing -> Neo4j loading

Serving runtime:
FastAPI -> LangGraph -> configured Qdrant / Neo4j / Groq -> grounded response
```

SEC ingestion, processed-data publication, Qdrant indexing, and Neo4j loading are operator-run workflows. FastAPI startup connects to already-populated stores. API requests never ingest filings, rebuild the retrieval index, or load the graph.

---

# API

## 6. FastAPI endpoints

FinGraph Agent exposes:

```text
GET  /health
GET  /ready
POST /ask
POST /ask/stream
```

### `/health`

Verifies that the API process is alive.

### `/ready`

Performs bounded checks against required dependencies.

### `/ask`

Executes a synchronous FinGraph question.

### `/ask/stream`

Executes the same logical workflow through Server-Sent Events.

Example synchronous request:

```bash
curl -s \
  -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <configured-key>' \
  -d '{"question":"What was Microsoft revenue in 2026?"}'
```

Example streaming request:

```bash
curl -N \
  -X POST http://localhost:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <configured-key>' \
  -d '{"question":"What risks did NVIDIA describe?"}'
```

`X-API-Key` is required when an access key is configured and is mandatory in production configuration. `X-Request-ID` is optional; the API generates one when absent.

---

## 7. Response contract

The API returns structured output rather than only free-form text.

Response data can include:

```text
request ID
selected route
answer
citations
calculations
evidence
latency metadata
```

This keeps machine-readable execution metadata available to clients.

---

# Health and Readiness

## 8. Health semantics

`/health` verifies process liveness.

It does not claim that all external dependencies are available.

A healthy process can still be unready.

---

## 9. Readiness semantics

`/ready` verifies that required external dependencies are reachable and sufficiently configured.

Current readiness checks cover:

```text
Qdrant
Neo4j
LLM configuration
```

Checks are time-bounded.

An unavailable dependency should not cause readiness to hang indefinitely.

---

## 10. Why health and readiness are separate

The distinction is:

```text
health
  -> is the application process alive?
```

```text
readiness
  -> can the application currently serve its intended workload?
```

This separation is important for deployment platforms and operational debugging.

---

# Managed Deployment

## 11. Production topology

The managed deployment uses:

```text
Railway       -> application runtime
Qdrant Cloud  -> narrative vector store
Neo4j Aura    -> structured graph store
Groq          -> LLM provider
Langfuse      -> observability
```

High-level topology:

```text
                         Internet
                            |
                            v
                         Railway
                     FinGraph API
                      /   |    \
                     /    |     \
                    v     v      v
              Qdrant   Neo4j    Groq
               Cloud    Aura    LLM
                    \     |      /
                     \    |     /
                      +---+----+
                          |
                          v
                       Langfuse
```

Durable application execution state is stored separately on a Railway persistent volume.

---

## 12. Public API

Current public deployment:

```text
https://fingraph-agent-api-production.up.railway.app
```

The public deployment has been validated for:

```text
health
readiness
GRAPH execution
VECTOR execution
HYBRID execution
SSE execution
```

---

# Railway

## 13. Railway project

Current Railway deployment consists of:

```text
project: fingraph-agent
service: fingraph-agent-api
```

The service hosts the FastAPI application.

Railway volume, health-check, environment-variable, runtime-identity, and related service settings are currently managed in the Railway platform rather than by a committed Railway manifest.

There is no committed `railway.toml`, `railway.json`, or `Procfile` defining those platform settings. The values documented here are deployment requirements and observed configuration, while live Railway platform state remains an external deployment dependency.

---

## 14. Railway persistent volume

A persistent Railway volume is mounted at:

```text
/data
```

The volume is used for durable application state.

Current LLM execution database path:

```text
/data/fingraph_runs.db
```

---

## 15. Persistent LLM state

LLM request state is stored in SQLite.

The deployment configuration uses:

```text
FINGRAPH_LLM_STATE_DB_PATH=/data/fingraph_runs.db
```

This ensures that resumable request state is stored on persistent storage rather than the ephemeral container filesystem.

Persistence has been validated across a Railway application restart.

---

## 16. Railway runtime identity

The Docker image declares a non-root runtime user.

However, the current Railway deployment uses:

```text
RAILWAY_RUN_UID=0
```

because of persistent-volume ownership requirements.

Therefore the project does not claim that the current Railway runtime executes as a non-root process.

The distinction is:

```text
Docker image
  -> declares non-root application user
```

```text
current Railway runtime
  -> executes as root because of mounted-volume ownership requirements
```

This limitation should remain explicit until the deployment is changed and revalidated.

---

## 17. Railway health check

Railway uses:

```text
/ready
```

as the service health check.

Current health-check timeout:

```text
300 seconds
```

The readiness endpoint verifies required dependencies before the deployment is considered ready.

---

## 18. Railway deployment during development

Before the final GitHub release push, deployment is performed using the Railway CLI.

Development deployment flow:

```text
local working tree
      |
      v
railway up
      |
      v
Railway service
```

This allows deployment validation before the repository's first final Phase 1 commit.

---

# Qdrant Cloud

## 19. Purpose

Qdrant Cloud is the managed narrative retrieval store.

It hosts:

```text
dense vectors
BM25-style sparse vectors
chunk payload metadata
```

The logical retrieval behavior is the same as the local Qdrant deployment.

---

## 20. Retrieval representations

Current dense model:

```text
BAAI/bge-small-en-v1.5
```

Dimension:

```text
384
```

The collection also contains BM25-style sparse vectors.

Supported retrieval modes:

```text
sparse
dense
hybrid
```

---

## 21. Qdrant credentials

Managed Qdrant credentials are supplied through environment configuration.

They must not be committed to source control.

Required values include configuration equivalent to:

```text
Qdrant URL
Qdrant API key
collection configuration
```

Exact secrets are intentionally not documented in the repository.

---

## 22. Local versus managed Qdrant state

The logical architecture supports both:

```text
local Qdrant
managed Qdrant Cloud
```

The deployment documentation must not assume that a local rebuild automatically updated the managed collection.

Corpus version and point counts should be verified against the actual configured target before claiming that a managed environment contains a specific dataset version.

---

# Neo4j Aura

## 23. Purpose

Neo4j Aura provides the managed structured financial graph.

The graph stores:

```text
Company
Filing
Section
FinancialFact
FiscalPeriod
```

with the canonical relationships defined by the FinGraph graph model.

---

## 24. Aura configuration

Aura credentials and connection information are provided through environment configuration.

The deployment uses the instance-specific database identity required by the Aura deployment.

Secrets must not be committed to source control.

---

## 25. Application safeguards

Managed Neo4j execution still passes through the same application controls as local execution.

These include:

```text
static Cypher validation
schema restrictions
parameter validation
required result bounds
EXPLAIN preflight
query timeout
read-routed execution
bounded repair
```

Managed hosting is not treated as a reason to bypass application validation.

---

## 26. Database-level read-only boundary

FinGraph Agent distinguishes between:

```text
application-level read restrictions
```

and:

```text
database-level authorization
```

The project does not claim a verified dedicated read-only serving identity for Aura unless such an identity has been explicitly configured and tested.

Until then, the guaranteed controls are application-level defense in depth.

This limitation is intentional and documented.

---

# LLM Provider

## 27. Groq

The primary LLM provider is:

```text
Groq
```

Current primary model:

```text
openai/gpt-oss-120b
```

The model is used for tasks including:

```text
Text-to-Cypher generation
Cypher repair
grounded answer synthesis
evaluation tasks
```

Route selection itself follows the application's controlled routing policy and is not documented as an LLM-authoritative decision.

Deterministic financial calculations are not delegated to the model.

---

## 28. Provider failure handling

Provider failures are classified explicitly.

Recoverable conditions can include:

```text
rate limits
quota exhaustion
temporary provider failures
network interruptions
```

These conditions can produce a deferred execution outcome at the API or evaluation layer. The durable resumable request record remains pending when execution can be retried later.

They are not automatically treated as:

```text
quality failures
retrieval failures
graph failures
```

---

## 29. Provider retry behavior

LLM calls use bounded retries and backoff.

The system does not retry indefinitely.

Persistent request identity allows incomplete calls to be resumed without repeating already successful logical requests.

---

# Durable Execution State

## 30. SQLite state model

LLM execution uses SQLite for durable state.

Durable persisted request states are:

```text
pending
running
succeeded
failed
```

A recoverable provider condition can surface as a deferred execution outcome while the durable request remains pending for later resumption.

Logical requests receive deterministic identities.

This enables:

```text
resume
deduplication
provider quota recovery
failure analysis
reproducibility
```

---

## 31. Deployment persistence

Local and managed deployments can use different SQLite paths.

The execution semantics remain the same.

Production persistence requires a path backed by durable storage.

For Railway:

```text
/data/fingraph_runs.db
```

is used.

---

# Observability

## 32. Langfuse

Langfuse provides LLM observability.

Tracing can capture information such as:

```text
request identity
task
route
model
provider execution
latency
execution status
deferred outcome
```

Tracing is part of the production architecture rather than only a development debugging mechanism.

---

## 33. Provider defer tracing

A provider-deferred execution outcome has been validated in Langfuse.

This confirms that recoverable provider conditions remain observable rather than being collapsed into generic internal-server errors, while resumable durable state remains represented by the persisted request record.

---

## 34. Sensitive-data handling

Secrets must not be stored in trace payloads.

Tracing should focus on:

```text
execution metadata
model behavior
route behavior
latency
error classification
```

without exposing provider API keys, database credentials, or other sensitive configuration.

---

# Environment Configuration

## 35. Local environment files

Local development uses the private `.env.dev` profile.

Typical development flow:

```bash
cp .env.example .env.dev
```

Configure the local credentials and settings required by the workflows you plan to run.

Before running host-side CLI workflows, load the profile into the current shell:

```bash
set -a
source .env.dev
set +a
```

Docker Compose uses the same profile explicitly:

```bash
docker compose --env-file .env.dev up -d
```

`.env.dev` contains secrets and must remain ignored by Git.

The current application settings implementation still retains an optional legacy `.env` dotenv fallback. The Phase 1 workflow does not depend on that fallback; explicit `.env.dev` sourcing supplies the intended process environment.

---

## 36. Remote configuration

Managed configuration can be loaded locally through the private `.env.prod` profile.

```text
.env.prod
```

is not committed.

When intentionally running local commands against managed services:

```bash
set -a
source .env.prod
set +a
```

exports the production-profile values into the current shell.

`.env.prod` on the developer machine is not automatically synchronized to Railway. Railway production variables are configured independently in the Railway service.

The current Railway production configuration includes the required FinGraph API access key and uses:

```text
FINGRAPH_LLM_STATE_DB_PATH=/data/fingraph_runs.db
```

for persistent resumable LLM state.

---

## 37. Secret categories

Secret configuration includes values such as:

```text
Groq API credentials
Qdrant Cloud credentials
Neo4j Aura credentials
Langfuse credentials
```

None of these values should appear in:

```text
README
documentation
tests
committed configuration
example logs
benchmark reports
```

---

# API Streaming

## 38. SSE endpoint

Streaming is exposed through:

```text
POST /ask/stream
```

The endpoint uses Server-Sent Events.

Supported behavior includes:

```text
accepted events
heartbeat
completion
controlled errors
disconnect handling
cancellation
```

---

## 39. TTFT semantics

TTFT in FinGraph performance reporting refers to:

```text
validated answer availability
```

not merely:

```text
HTTP acceptance
```

or:

```text
initial SSE accepted event
```

This avoids reporting connection establishment as useful answer latency.

---

# Performance Validation

## 40. Load testing

Formal API load testing is performed independently for:

```text
GRAPH
VECTOR
HYBRID
```

Current results:

| Route | RPS | P50 | P95 | P99 | Errors |
|---|---:|---:|---:|---:|---:|
| GRAPH | 171.517 | 21 ms | 42 ms | 58 ms | 0 |
| VECTOR | 82.818 | 44 ms | 78 ms | 120 ms | 0 |
| HYBRID | 60.468 | 63 ms | 91 ms | 120 ms | 0 |

The benchmark keeps route-specific behavior visible.

---

## 41. Performance reports

Primary API load report:

```text
evals/reports/api_load_report.json
```

Related tooling includes:

```text
scripts/load_test_api.py
scripts/summarize_load_tests.py
```

---

# Data Build Deployment

## 42. SEC ingestion

Current scale configuration covers:

```text
25 companies
up to 8 filings per company
4 company workers
5 global SEC requests / second
chunk size 500
chunk overlap 75
```

The canonical processed corpus currently contains:

```text
25 companies
198 filings
731 sections
12,666 chunks
115,649 financial facts
1,475 fiscal periods
```

---

## 43. Retrieval index build

The retrieval index supports:

```text
rebuild mode
incremental ticker replacement
```

Current v1.1 full rebuild:

```text
chunks:      12,666
points:      12,666
batches:        211
batch size:      64
```

Measured throughput:

```text
8.102 chunks / second
```

---

## 44. Graph build

The graph loader supports:

```text
company-bounded loading
bounded UNWIND transactions
stale-entity pruning
post-load audit
```

Current transaction batch size:

```text
1,000
```

Current v1.1 graph result:

```text
Companies:       25
Filings:         198
Sections:        731
Financial facts: 115,649
Fiscal periods:  1,475
Orphans:          0
Count mismatches: 0
```

---

# Operational Commands

## 45. Data-build and inspection workflows

### SEC ingestion

Development/default configuration:

```bash
uv run python scripts/ingest_sec.py
```

Full v1.1 scale configuration:

```bash
uv run python scripts/ingest_sec.py --config configs/scale.toml
```

SEC ingestion requires `FINGRAPH_SEC_USER_AGENT` and network access to SEC EDGAR.

### Retrieval index

Full rebuild:

```bash
uv run python scripts/build_retrieval_index.py --mode rebuild --batch-size 64
```

Incremental offline ticker replacement:

```bash
uv run python scripts/build_retrieval_index.py \
  --mode incremental \
  --maintenance-mode \
  --batch-size 64 \
  --tickers AAPL
```

Incremental replacement is an offline-maintenance operation. Concurrent API serving must be stopped because replacement is rollback-safe but not read-atomic for live readers.

### Graph build and audit

```bash
uv run python scripts/build_graph.py --batch-size 1000 --prune-stale
```

Audit without loading:

```bash
uv run python scripts/build_graph.py --audit-only
```

### Graph and metric inspection

```bash
uv run python scripts/query_graph.py
```

```bash
uv run python scripts/query_graph.py \
  --ticker MSFT \
  --concept Revenue \
  --limit 10
```

```bash
uv run python scripts/query_metric.py \
  --ticker MSFT \
  --metric revenue \
  --start-year 2024 \
  --end-year 2026
```

### Graph QA

```bash
uv run python scripts/graph_qa.py \
  "What was Microsoft's revenue in 2026?"
```

### Retrieval inspection

```bash
uv run python scripts/search_filings.py \
  "What risks did NVIDIA describe?" \
  --ticker NVDA \
  --section "Risk Factors" \
  --limit 5
```

### Langfuse smoke

```bash
uv run python scripts/smoke_langfuse.py
```

## 46. Evaluation workflows

```bash
uv run python scripts/evaluate_agent_routing.py
uv run python scripts/evaluate_agent_evidence.py
uv run python scripts/evaluate_graph_qa.py
uv run python scripts/evaluate_graph_repair.py
uv run python scripts/evaluate_new_answer_cases_evidence.py
uv run python scripts/evaluate_answers.py
```

Answer evaluator modes:

```bash
uv run python scripts/evaluate_answers.py --rerun-failed
uv run python scripts/evaluate_answers.py --fresh --cooldown-seconds 65
```

Retrieval evaluation:

```bash
uv run python scripts/evaluate_retrieval.py \
  --dataset evals/datasets/retrieval_bootstrap.json \
  --report evals/reports/retrieval_scale_v1_1_report.json \
  --k 5
```

Routing is deterministic and service-free. Other evaluations require the applicable configured Qdrant, Neo4j, Groq, processed corpus, and durable-state dependencies.

## 47. Load-testing workflow

The Locust harness supports `graph`, `vector`, and `hybrid`, each in `sync` or `stream` mode.

Example:

```bash
FINGRAPH_LOAD_ROUTE=graph \
FINGRAPH_LOAD_MODE=sync \
FINGRAPH_LOAD_HOST=http://localhost:8000 \
uv run locust -f load_tests/locustfile.py
```

Repeat the required route/mode combinations while writing CSV prefixes consumed by the summarizer, then run:

```bash
uv run python scripts/summarize_load_tests.py \
  --input-dir reports/load \
  --output evals/reports/api_load_report.json
```

The exact historical Locust user count, spawn rate, duration, and complete headless command used for the existing curated report are not recorded in the repository and must not be inferred from request counts.

---

# CI

## 48. GitHub Actions

The repository contains a GitHub Actions CI workflow.

The workflow runs on:

```text
push to main
pull requests
```

CI performs:

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

---

## 49. Locked dependencies

CI uses the repository lockfile.

Dependency installation is performed through:

```text
uv
```

with the locked dependency graph.

This reduces environment drift between local and hosted validation.

---

## 50. Docker build gate

Docker image construction is a separate CI gate.

A repository can therefore pass Python checks while still failing container construction.

Keeping Docker build validation in CI catches packaging and runtime-image regressions.

---

## 51. Hosted CI status

The CI workflow definition has been implemented and validated locally.

Hosted GitHub CI should only be claimed as complete after:

```text
final repository push
+
successful GitHub Actions run
```

Until that occurs, documentation should distinguish:

```text
CI workflow implemented
```

from:

```text
hosted CI validated
```

---

# Continuous Deployment

## 52. Development deployment

Current pre-release development deployment uses:

```text
railway up
```

This allows the public service to be tested before the repository's first final Phase 1 release commit.

---

## 53. Intended release deployment

The intended release flow is:

```text
local development
      |
      v
GitHub push
      |
      v
GitHub Actions CI
      |
      v
Railway deployment integration
      |
      v
public service
```

---

## 54. CD completion criterion

GitHub-to-Railway continuous deployment is only considered complete when:

```text
final repository exists on GitHub
CI succeeds
Railway is connected to the repository
a GitHub-triggered deployment succeeds
public smoke tests pass
```

CLI-based development deployment does not by itself satisfy this release criterion.

---

# Release Validation

## 55. Final quality gates

The Phase 1 implementation and pre-commit verification gates have been completed.

Validated local/repository gates include:

```text
Ruff lint
Ruff formatting check
mypy
Python compilation
pytest
uv lock check
Docker build
verification-only repository audit
release-blocker resolution
final quality-suite rerun
credential rotation
```

The repository is ready for its first/final Phase 1 commit.

Hosted GitHub CI, GitHub-to-Railway integration, GitHub-triggered deployment, and final public smoke validation remain post-commit/public-release gates.

---

## 56. Codex audit

The repository-level audit is intended to review:

```text
correctness
architecture consistency
security issues
dead code
configuration mistakes
deployment assumptions
tests
documentation accuracy
```

The audit is performed near the final release gate rather than after every development block.

---

## 57. Credential rotation

Credentials exposed during development were rotated before the current release-candidate validation; publication-visible configuration retains blank or non-production credential values.

This includes applicable:

```text
Groq credentials
Qdrant credentials
Neo4j credentials
Langfuse credentials
```

Credential rotation is complete for the current release candidate and remains part of the release audit boundary.

---

# Production Smoke Validation

## 58. Core smoke tests

The managed deployment should validate:

```text
GET /health
GET /ready
POST /ask with GRAPH route
POST /ask with VECTOR route
POST /ask with HYBRID route
POST /ask/stream
```

A successful health check alone does not prove the full application works.

---

## 59. Dependency smoke tests

Managed services should also be verified independently where useful.

Examples include:

```text
Qdrant connectivity
Qdrant collection access
Neo4j connectivity
graph query execution
Groq provider execution
Langfuse tracing
persistent SQLite state
```

---

## 60. Persistent-state smoke test

A deployment restart should not erase the LLM execution database.

Validation flow:

```text
create / observe persisted state
      |
      v
restart Railway application
      |
      v
verify /data/fingraph_runs.db still exists
      |
      v
verify state remains readable
```

This behavior has been validated in the current deployment.

---

# Security

## 61. Security model

Production security is layered.

```text
environment secrets
       |
       v
application validation
       |
       v
database connection controls
       |
       v
managed provider controls
```

No single layer is treated as sufficient.

---

## 62. Text-to-Cypher protection

Generated Cypher passes through:

```text
static validation
schema restrictions
parameter validation
required LIMIT
EXPLAIN
timeout
read routing
bounded repair
```

These controls reduce risk.

They are not documented as equivalent to a verified database-level authorization boundary.

---

## 63. Secret management

Secrets must be stored using environment or platform secret configuration.

Do not commit:

```text
.env.dev
.env.prod
API keys
database passwords
private service credentials
```

The repository may retain general `.env` / `.env.*` ignore rules as a safety fallback, while `.env.example` remains the public placeholder-only template.

Example environment files should contain placeholders only.

---

# Known Deployment Limitations

## 64. Current limitations

Current deployment limitations include:

- Railway currently runs the container as root because of volume ownership requirements;
- the project does not yet claim a verified dedicated database-level read-only Neo4j serving identity;
- LLM serving depends on an external provider;
- provider quota or rate limitations can temporarily defer requests;
- managed data-store corpus versions must be verified independently from local data builds;
- hosted GitHub CI requires final repository-push validation;
- GitHub-to-Railway CD requires final release integration validation;
- the current workload does not justify Kubernetes or another cluster orchestrator.

---

# Operational Principles

## 65. Keep environments logically equivalent

Local and managed environments may use different infrastructure implementations.

The logical application behavior should remain equivalent.

For example:

```text
local Qdrant
<-> Qdrant Cloud
```

and:

```text
local Neo4j
<-> Neo4j Aura
```

should use the same logical retrieval and graph models.

---

## 66. Do not infer deployment state

A successful local build does not prove that:

```text
Qdrant Cloud contains the same points
Neo4j Aura contains the same graph
Railway is using the latest code
```

Each environment must be verified directly.

---

## 67. Fail readiness explicitly

If a required dependency is unavailable, readiness should fail clearly.

The application should not report itself ready while silently operating without a required dependency.

---

## 68. Preserve durable state

Execution state that is required for resumability must live outside ephemeral container storage in production.

---

## 69. Measure before adding infrastructure

Deployment complexity should follow measured requirements.

The current system uses:

```text
Railway
Qdrant Cloud
Neo4j Aura
Langfuse
Groq
```

without introducing:

```text
Kubernetes
Kafka
distributed task orchestration
```

because the measured workload does not currently require them.

---

# Release-State Checklist

## 70. Phase 1 release checklist

Phase 1 implementation and pre-commit verification are complete. Remaining items are post-commit/public-release gates.

```text
[x] final documentation refreshed
[x] final Ruff check passes
[x] final Ruff format check passes
[x] final mypy passes
[x] final Python compilation passes
[x] final pytest suite passes
[x] uv lock check passes
[x] final Docker build passes
[x] repository-level Codex audit completed
[x] release-blocking audit findings resolved
[x] final quality suite rerun
[x] exposed development credentials rotated
[ ] first/final Phase 1 commit created
[ ] repository pushed to GitHub
[ ] hosted GitHub Actions CI passes
[ ] GitHub-to-Railway deployment configured
[ ] GitHub-triggered Railway deployment succeeds
[ ] public production smoke tests pass
```

The project should only claim completion for gates that have actually been executed successfully.
