# FinGraph Agent Evaluation

## 1. Purpose

FinGraph Agent is evaluated as a multi-layer system rather than as a single opaque RAG application.

The evaluation strategy separates:

- retrieval quality;
- routing correctness;
- graph QA correctness;
- graph self-healing;
- evidence grounding;
- deterministic calculation correctness;
- answer quality;
- API performance;
- scale behavior.

The goal is to make failures diagnosable.

A bad final answer should be traceable to the layer that caused it.

---

## 2. Evaluation architecture

```text
                       End-to-end answer quality
                                 ^
                                 |
                         answer evaluation
                                 ^
                                 |
                 +---------------+---------------+
                 |                               |
          evidence evaluation             calculation tests
                 ^                               ^
                 |                               |
          routing evaluation               graph QA evaluation
                 ^                               ^
                 |                               |
        retrieval evaluation              repair benchmark
                 ^                               ^
                 |                               |
            store health                  graph integrity
```

This layered structure avoids treating every failure as:

```text
"the RAG system failed"
```

Instead, failures can be associated with:

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

---

## 3. Evaluation principles

The evaluation design follows several principles.

### 3.1 Evaluate components independently

Retrieval should be evaluated without answer generation.

Routing should be evaluated without requiring full synthesis.

Graph repair should be evaluated on injected graph failures.

Calculations should be tested independently from the LLM.

### 3.2 Distinguish quality failures from provider failures

A provider quota or rate-limit event is not an answer-quality failure.

Evaluation states distinguish:

```text
passed
quality failure
deferred
execution failure
```

### 3.3 Preserve resumability

Completed current cases should be reused.

Evaluation should not repeat successful LLM requests unnecessarily.

### 3.4 Use deterministic metrics where possible

Metrics such as:

```text
Precision@K
MRR
latency
throughput
error rate
graph entity counts
orphan counts
```

are preferred over subjective inspection alone.

### 3.5 Do not claim unsupported metrics

NDCG@K and Recall@K are not currently reported for the bootstrap retrieval benchmark because it does not contain explicit chunk-level qrels.

---

# Retrieval Evaluation

## 4. Retrieval benchmark

Retrieval is evaluated independently across:

```text
sparse
dense
hybrid
```

The current bootstrap dataset contains:

```text
questions: 12
k:         5
```

Current reported quality metrics:

```text
Precision@5
MRR
```

NDCG@5 and Recall@5 are reported as `N/A` because the dataset does not contain explicit chunk-level qrels.

Reported latency metrics:

```text
mean
P50
P95
```

---

## 5. Historical baseline retrieval report

The original development retrieval index contained:

```text
3,005 chunks
```

The saved baseline report was generated before the corrected M04 metric implementation. Its retrieval-quality values are therefore retained as **historical evidence only** and must not be presented as current Precision@5 or NDCG@5 results.

Its latency measurements remain useful operational evidence:

| Method | Mean latency | P50 latency | P95 latency |
|---|---:|---:|---:|
| Sparse | 5.605 ms | 3.831 ms | 14.064 ms |
| Dense | 18.154 ms | 8.597 ms | 63.605 ms |
| Hybrid | 14.635 ms | 13.534 ms | 21.051 ms |

This baseline report remains historical pre-fix evidence. Current quality claims come from the corrected v1.1 rerun.

---

## 6. Current v1.1 corrected retrieval report

The retrieval index contains:

```text
12,666 chunks
```

The same 12-question dataset and `k=5` retrieval setup were reused.

The corrected current release report is:

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

NDCG@5 and Recall@5 remain `N/A` because explicit chunk-level qrels are absent from the benchmark dataset.

---

## 7. Retrieval latency context

The historical baseline report remains useful as pre-fix operational evidence, while the corrected v1.1 report supplies the current scale latency measurements.

Historical baseline mean latency:

| Method | Mean latency |
|---|---:|
| Sparse | 5.605 ms |
| Dense | 18.154 ms |
| Hybrid | 14.635 ms |

Current corrected v1.1 mean latency:

| Method | Mean latency |
|---|---:|
| Sparse | 9.782 ms |
| Dense | 18.894 ms |
| Hybrid | 17.204 ms |

The current release documentation uses the corrected v1.1 report for present-tense latency and quality claims.

---

## 8. Retrieval interpretation

The current code retains sparse, dense, and hybrid retrieval because each mode serves a distinct retrieval role.

Current quality ordering is taken from the corrected v1.1 report rather than from the historical pre-M04 baseline report.

A reranker should only be introduced if controlled evaluation demonstrates measurable value that justifies another inference stage.

---

## 9. Retrieval report files

Primary reports:

```text
evals/reports/retrieval_bootstrap_report.json
evals/reports/retrieval_scale_v1_1_report.json
```

Primary dataset:

```text
evals/datasets/retrieval_bootstrap.json
```

---

# Routing Evaluation

## 10. Routing benchmark

The LangGraph router is evaluated independently from final answer generation.

The router selects among:

```text
VECTOR
GRAPH
HYBRID
```

Current dedicated routing benchmark result:

```text
cases:   15
passed:  15
failed:   0
accuracy: 1.000
```

This result comes from `evals/reports/agent_routing_bootstrap_report.json` and validates the route-selection layer independently from downstream execution. It should not be conflated with route assertions embedded in other evaluation suites.

---

## 11. Why routing is evaluated separately

A correct final answer does not necessarily prove that routing behavior is sound.

For example, a question might accidentally succeed through an unnecessarily expensive route.

Separating routing evaluation allows the system to check whether:

```text
narrative questions -> VECTOR
structured questions -> GRAPH
mixed questions -> HYBRID
```

without relying only on final answer text.

---

## 12. Routing reports

Relevant files include:

```text
evals/datasets/agent_routing_bootstrap.json
evals/reports/agent_routing_bootstrap_report.json
```

---

# Graph QA Evaluation

## 13. Graph QA

Graph QA is evaluated independently from end-to-end synthesis.

The graph path validates:

```text
natural-language question
        |
        v
Text-to-Cypher
        |
        v
validation
        |
        v
EXPLAIN
        |
        v
execution
        |
        v
structured graph rows
```

The evaluation checks whether supported financial questions produce valid structured results.

---

## 14. Graph integrity audit

Graph correctness is also audited against the canonical processed corpus using entity-count and configured orphan checks.

The current v1.1 graph contains:

```text
Companies:       25
Filings:         198
Sections:        731
Financial facts: 115,649
Fiscal periods:  1,475
```

Post-load audit:

```text
count mismatches: 0
orphans:          0
passed:           true
```

This validates graph storage independently from LLM-generated Cypher.

---

## 15. Graph QA reports

Relevant files include:

```text
evals/datasets/graph_qa_bootstrap.json
evals/reports/graph_qa_bootstrap_report.json
```

---

# Graph Self-Healing Evaluation

## 16. Purpose

Self-healing is evaluated using deliberately injected graph-query failures.

This prevents the project from claiming repair capability merely because ordinary Text-to-Cypher queries succeed.

The benchmark compares:

```text
baseline execution
```

against:

```text
bounded repair execution
```

on the same fault cases.

---

## 17. Fault benchmark

Current injected-fault set:

```text
6 cases
```

Results:

```text
baseline success:      0 / 6
self-healing success:  6 / 6
average repair attempts: 1.000
```

The repair path recovered all tested injected faults.

---

## 18. Repair overhead

Measured repair overhead from the current curated report:

```text
mean added latency:        5,862.905 ms
added input tokens:           19,718
added cached-input tokens:     2,816
added output tokens:           2,351
added total tokens:           22,069
instrumented added cost:       $0.004
```

These values describe the six-case injected-fault benchmark and are not a general estimate for all Graph QA traffic.

This makes the reliability benefit and its runtime cost explicit.

Self-healing is therefore not treated as free.

---

## 19. Repair benchmark files

Dataset:

```text
evals/datasets/graph_repair_cases.json
```

Report:

```text
evals/reports/graph_repair_report.json
```

---

# Evidence Evaluation

## 20. Evidence grounding

FinGraph Agent evaluates evidence behavior separately from answer semantics.

The objective is to verify that answers are grounded in retrieved or structured evidence rather than unsupported model knowledge.

The evidence layer validates areas such as:

```text
citation coverage
source provenance
entity attribution
route-specific evidence
hybrid evidence
```

---

## 21. Multi-company evidence

Multi-company questions require evidence to remain associated with the correct company.

The system supports:

```text
multi-company graph rows
company-diversified vector evidence
entity-level citation grounding
```

This prevents evidence from one company from being silently attributed to another.

---

## 22. Evidence reports

Relevant files include:

```text
evals/datasets/agent_evidence_bootstrap.json
evals/reports/agent_evidence_bootstrap_report.json
evals/reports/answer_new_cases_evidence_report.json
```

---

# Deterministic Calculation Evaluation

## 23. Calculation strategy

Financial arithmetic is evaluated independently from LLM reasoning.

Supported operations include:

```text
absolute change
percentage change
margin
CAGR
period comparison
```

The calculations are performed deterministically in Python.

---

## 24. Validated calculation examples

Representative validated results:

```text
Microsoft revenue percentage change
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

The LLM may explain these values, but the arithmetic itself is not model-generated.

---

## 25. Why calculation evaluation is independent

A final answer can hide several possible errors:

```text
wrong retrieved value
wrong period
wrong formula
wrong arithmetic
wrong synthesis wording
```

Testing deterministic calculations independently isolates formula correctness from the rest of the system.

---

# Answer Evaluation

## 26. Golden answer evaluation

The golden answer dataset contains:

```text
160 cases
```

The suite covers combinations of:

- structured financial values;
- historical values;
- financial changes;
- deterministic calculations;
- narrative filing questions;
- risk analysis;
- business descriptions;
- temporal questions;
- multi-company questions;
- hybrid questions;
- unsupported or insufficient-evidence cases.

---

## 27. Current answer-evaluation result

The current release answer-evaluation report was completed under the current evaluator, evaluation configuration, source, corpus, dataset, and output contract.

Current release result:

```text
total:               160
completed:           160
passed:              160
quality_failed:        0
llm_quality_failed:    0
deferred:              0
execution_failed:      0
accuracy:          1.000
failure_counts:        {}
```

The current suite therefore passes 160 / 160 cases. This report is the publication-visible release evidence for end-to-end answer evaluation.

The `160 / 160` result is the pass rate for the current versioned evaluation suite under the current evaluator and configuration. It is not a claim of universal financial-question correctness.

### 27.1 Deterministic answer-contract checks

Answer evaluation does not rely on an LLM judge alone.

The evaluator first applies deterministic contract checks covering the case requirements, including applicable expectations for:

```text
selected route
insufficient-evidence behavior
citation presence and validity
evidence years
company attribution
deterministic financial calculations
structured output contract
```

These checks are used to catch failures that should not be delegated to subjective semantic judging.

### 27.2 Structured Groq answer judge

Where semantic answer correctness must be assessed, the evaluator uses `GroqAnswerJudge`.

The judge evaluates the generated answer against the versioned case contract after deterministic execution and contract validation. Judge execution uses the same durable/resumable LLM execution infrastructure as other provider-backed workflows.

This separation is important:

```text
deterministic assertions
        +
structured semantic judge
        =
answer-evaluation result
```

A provider availability failure is not silently converted into an answer-quality failure.

### 27.3 Fingerprints and reusable results

Reusable answer-evaluation results are identity-bound.

The evaluation tracks:

```text
case fingerprint
dataset fingerprint
evaluation fingerprint
evaluator version
```

The evaluation fingerprint binds the result to non-secret evaluation configuration and the implementation/data surfaces that affect serving behavior, including:

```text
model / retrieval / graph evaluator configuration
src/fingraph Python source-tree hash
processed-corpus hash
```

A prior successful case may be reused only when its case identity and the current evaluation identity still match.

Changes to the relevant case, dataset, evaluator contract/version, serving source tree, non-secret evaluation configuration, or processed corpus invalidate reuse and require regeneration of the affected answer evidence.

Documentation-only edits do not change the serving-source fingerprint.

---

## 28. Resumable answer evaluation

Answer evaluation uses persistent LLM request state.

Successful current cases are reused on reruns.

Conceptually:

```text
evaluation case
      |
      v
derive deterministic request identity
      |
      v
existing successful result?
      |
   +--+--+
   |     |
  yes    no
   |     |
   v     v
reuse   execute
```

This prevents completed evaluations from being unnecessarily repeated.

---

## 29. Provider failures and evaluation state

Provider failures are not automatically counted as answer-quality failures.

The evaluation distinguishes:

```text
passed
quality failure
deferred
execution failure
```

A Groq rate-limit or quota condition can therefore surface to evaluation as a deferred outcome without contaminating answer-quality metrics. The durable request record remains resumable using the persisted execution state rather than requiring a persisted `deferred` status.

---

## 30. Answer evaluation report

Primary report:

```text
evals/reports/answer_golden_report.json
```

Primary dataset:

```text
evals/datasets/answer_golden.json
```

---

## 30.1 Curated release reports

Exactly nine publication-visible curated reports are retained under `evals/reports/`:

```text
agent_evidence_bootstrap_report.json
agent_routing_bootstrap_report.json
answer_golden_report.json
answer_new_cases_evidence_report.json
api_load_report.json
graph_qa_bootstrap_report.json
graph_repair_report.json
retrieval_bootstrap_report.json
retrieval_scale_v1_1_report.json
```

`retrieval_bootstrap_report.json` is intentionally retained as historical pre-fix retrieval evidence. Current retrieval-quality claims use `retrieval_scale_v1_1_report.json`.

Ad hoc or newly generated benchmark output should not be treated as release evidence until it has been reviewed and intentionally promoted.

## 30.2 Reproducing curated reports

The following commands map the supported evaluation entry points to their primary release artifacts.

| Evaluation | Command | Primary dataset/input | Services required | Curated report |
|---|---|---|---|---|
| Routing | `uv run python scripts/evaluate_agent_routing.py` | `evals/datasets/agent_routing_bootstrap.json` | None; deterministic/service-free | `evals/reports/agent_routing_bootstrap_report.json` |
| Agent evidence | `uv run python scripts/evaluate_agent_evidence.py` | `evals/datasets/agent_evidence_bootstrap.json` | Configured Qdrant, Neo4j, Groq | `evals/reports/agent_evidence_bootstrap_report.json` |
| Graph QA | `uv run python scripts/evaluate_graph_qa.py` | `evals/datasets/graph_qa_bootstrap.json` | Configured Neo4j, Groq | `evals/reports/graph_qa_bootstrap_report.json` |
| Graph repair | `uv run python scripts/evaluate_graph_repair.py` | `evals/datasets/graph_repair_cases.json` | Configured Neo4j, Groq, durable SQLite state | `evals/reports/graph_repair_report.json` |
| New answer-case evidence | `uv run python scripts/evaluate_new_answer_cases_evidence.py` | Current answer suite, focused on cases 81-160 | Configured Qdrant, Neo4j, Groq | `evals/reports/answer_new_cases_evidence_report.json` |
| Answer evaluation | `uv run python scripts/evaluate_answers.py` | `evals/datasets/answer_golden.json` | Configured Qdrant, Neo4j, Groq, durable SQLite state, processed corpus | `evals/reports/answer_golden_report.json` |
| Retrieval, historical/bootstrap | `uv run python scripts/evaluate_retrieval.py --dataset evals/datasets/retrieval_bootstrap.json --report evals/reports/retrieval_bootstrap_report.json --k 5` | `evals/datasets/retrieval_bootstrap.json` | Configured Qdrant containing the intended index | `evals/reports/retrieval_bootstrap_report.json` |
| Retrieval, current v1.1 | `uv run python scripts/evaluate_retrieval.py --dataset evals/datasets/retrieval_bootstrap.json --report evals/reports/retrieval_scale_v1_1_report.json --k 5` | Same 12-question dataset against the v1.1 index | Configured Qdrant containing the v1.1 index | `evals/reports/retrieval_scale_v1_1_report.json` |
| API load | Locust route/mode runs followed by `scripts/summarize_load_tests.py` | Route-specific Locust CSV files | Running API at the configured load host | `evals/reports/api_load_report.json` |

Answer-evaluator reuse modes:

```bash
uv run python scripts/evaluate_answers.py --rerun-failed
```

reuses passing completed cases while rerunning eligible failures.

A deliberately fresh evaluation can be requested with:

```bash
uv run python scripts/evaluate_answers.py --fresh --cooldown-seconds 65
```

The cooldown value is an operator choice for provider-rate management; it is not part of the benchmark definition.

Graph-repair evaluation also supports explicit dataset, report, state-database, and cooldown options when an isolated run is required.

---

# API Performance Evaluation

The formal concurrent load report represents the **warm resumable/cache-enabled production path**. It must not be interpreted as cold LLM-provider latency.

## 31. Load-testing strategy

API performance is measured separately for:

```text
GRAPH
VECTOR
HYBRID
```

This is important because the routes have different workloads.

Combining all routes into one aggregate latency number would hide meaningful behavior.

---

## 32. Formal load results

The current curated report records synchronous and SSE measurements for the warm resumable/cache-enabled serving path.

### 32.1 Synchronous `/ask`

| Route | Requests | RPS | Mean | P50 | P95 | P99 | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRAPH | 4,977 | 171.517 | 23.249 ms | 21 ms | 42 ms | 58 ms | 0 |
| VECTOR | 2,405 | 82.818 | 48.135 ms | 44 ms | 78 ms | 120 ms | 0 |
| HYBRID | 1,757 | 60.468 | 65.918 ms | 63 ms | 91 ms | 120 ms | 0 |

### 32.2 SSE completed response and validated TTFT

| Route | Completed requests | RPS | Completed mean | Completed P50 | Completed P95 | Completed P99 | TTFT mean | TTFT P50 | TTFT P95 | TTFT P99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRAPH | 4,522 | 155.842 | 25.626 ms | 23 ms | 45 ms | 63 ms | 25.592 ms | 23 ms | 45 ms | 63 ms |
| VECTOR | 2,520 | 86.844 | 45.961 ms | 44 ms | 65 ms | 81 ms | 45.911 ms | 44 ms | 65 ms | 81 ms |
| HYBRID | 1,627 | 56.023 | 71.304 ms | 67 ms | 100 ms | 140 ms | 71.254 ms | 67 ms | 100 ms | 140 ms |

### 32.3 Stream-open diagnostic

| Route | Mean | P50 | P95 | P99 |
|---|---:|---:|---:|---:|
| GRAPH | 4.764 ms | 4 ms | 7 ms | 24 ms |
| VECTOR | 6.815 ms | 6 ms | 11 ms | 26 ms |
| HYBRID | 7.371 ms | 7 ms | 12 ms | 27 ms |

No request errors were observed in the recorded benchmark.

The stream-open diagnostic measures connection establishment. It is not the project TTFT metric.

---

## 33. Load-test interpretation

The results reflect the relative complexity of each route.

### GRAPH

Fastest route in the benchmark.

Primary work:

```text
structured graph lookup
```

### VECTOR

Higher latency than GRAPH because it includes retrieval and embedding-related work.

### HYBRID

Highest latency because it combines multiple evidence paths.

The route-specific measurements support keeping explicit routing rather than forcing every question through one universal pipeline.

---

## 34. TTFT definition

For SSE performance reporting, TTFT is defined using:

```text
validated answer availability
```

rather than:

```text
initial accepted event
```

An early connection-acceptance event is not treated as useful generated-output latency.

This avoids reporting an artificially optimistic streaming number.

---

## 35. Load-test report

Primary report:

```text
evals/reports/api_load_report.json
```

The formal Locust harness supports `graph`, `vector`, and `hybrid`, each in `sync` or `stream` mode.

Example local invocation:

```bash
FINGRAPH_LOAD_ROUTE=graph \
FINGRAPH_LOAD_MODE=sync \
FINGRAPH_LOAD_HOST=http://localhost:8000 \
uv run locust -f load_tests/locustfile.py
```

The current Locust harness does not send `X-API-Key`, so authenticated production load testing is not implemented in the Phase 1 harness. Use the harness against local or otherwise unprotected test endpoints. Adding authenticated remote-load support should be paired with a future load-evidence refresh rather than inferred from the current report.

To create the curated report, route/mode runs must emit CSV prefixes expected by the summarizer, for example:

```text
reports/load/graph_sync_stats.csv
```

After all required route/mode CSVs exist:

```bash
uv run python scripts/summarize_load_tests.py \
  --input-dir reports/load \
  --output evals/reports/api_load_report.json
```

The exact historical Locust user count, spawn rate, duration, and complete headless launch commands used for the already-published report are not captured in the current repository documentation and must not be inferred from request counts.

Future benchmark runs should record those workload parameters explicitly alongside the generated evidence.

---

# Scale Evaluation

## 36. Scale objective

The v1.1 scale extension validates that the same logical architecture can operate over a materially larger corpus without introducing unnecessary distributed infrastructure.

Baseline:

```text
6 companies
48 filings
192 sections
3,005 chunks
25,976 financial facts
355 fiscal periods
```

v1.1:

```text
25 companies
198 filings
731 sections
12,666 chunks
115,649 financial facts
1,475 fiscal periods
```

---

## 37. Corpus growth

| Entity | Baseline | v1.1 | Growth |
|---|---:|---:|---:|
| Companies | 6 | 25 | 4.167x |
| Filings | 48 | 198 | 4.125x |
| Sections | 192 | 731 | 3.807x |
| Chunks | 3,005 | 12,666 | 4.215x |
| Financial facts | 25,976 | 115,649 | 4.452x |
| Fiscal periods | 355 | 1,475 | 4.155x |

---

## 38. SEC ingestion scale benchmark

Corrected cached regeneration:

```text
companies:             25
filings:              198
sections:             731
chunks:            12,666
financial facts:  115,649
```

Measured execution:

```text
elapsed:               99.637 seconds
filings per minute:   119.233
companies per second:   0.251
filings downloaded:         0
companyfacts refreshed:     0
```

This measurement represents regeneration from cached source data.

It is not fresh network-download throughput.

---

## 39. Qdrant scale benchmark

Full v1.1 rebuild:

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

The dominant cost is embedding plus upsert, not artifact loading.

---

## 40. Neo4j scale benchmark

Full v1.1 graph build:

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

Final integrity result:

```text
count mismatches: 0
orphans:          0
passed:           true
```

---

## 41. Scale strategy validation

The scale extension validates:

```text
bounded SEC company concurrency
deterministic partitioning
company-bounded artifact loading
bounded Qdrant embedding/upsert batches
ticker-scoped incremental indexing
bounded Neo4j UNWIND transactions
post-load graph auditing
```

The current workload does not require:

```text
Spark
Ray
Kafka
Kubernetes
```

for correctness or measured scale.

---

# Production Validation

## 42. Public API validation

The deployed API has been validated across:

```text
health
readiness
GRAPH execution
VECTOR execution
HYBRID execution
SSE execution
```

The production API is deployed on Railway.

Managed persistence uses:

```text
Qdrant Cloud
Neo4j Aura
```

LLM execution uses:

```text
Groq
openai/gpt-oss-120b
```

Tracing uses:

```text
Langfuse
```

---

## 43. Persistent execution-state validation

LLM execution state is stored in SQLite.

In the Railway deployment, the database is stored on a persistent volume:

```text
/data/fingraph_runs.db
```

Persistence has been validated across application restart.

This confirms that resumable LLM state is not limited to an ephemeral container filesystem.

---

## 44. Observability validation

Langfuse tracing has been validated for production execution.

Provider-deferred execution has been observed in traces.

This confirms that provider failures can be inspected separately from generic application errors.

---

# Current Evaluation Summary

## 45. Summary table

| Area | Result |
|---|---|
| Retrieval | Corrected v1.1 current: sparse/dense/hybrid P@5 = 0.633/0.583/0.633; MRR = 0.792/0.729/0.686; NDCG@5 and Recall@5 = N/A |
| Routing | Dedicated routing benchmark: 15 / 15 passed |
| Graph integrity | Expected entity counts match; configured orphan checks = 0 |
| Graph self-healing | 6 / 6 repaired vs 0 / 6 baseline |
| Answer evaluation | Current release report: 160 / 160 passed; quality failures = 0; LLM quality failures = 0; deferred = 0; execution failures = 0; accuracy = 1.000 |
| Deterministic calculations | Validated |
| API load | 0 errors in recorded warm resumable/cache-enabled GRAPH / VECTOR / HYBRID benchmark |
| Qdrant scale | 12,666 points, 8.102 chunks/s |
| Neo4j scale | 118,078 rows, 1,967.965 rows/s |
| SEC scale | 198 cached filings regenerated in 99.637 s |
| Persistent LLM state | Validated across Railway restart |
| Langfuse tracing | Validated |

---

# Evaluation Limitations

## 46. Current limitations

The current evaluation suite has several explicit limitations.

### Retrieval relevance labels

The bootstrap retrieval dataset does not contain explicit chunk-level qrels.

Therefore:

```text
NDCG@K
Recall@K
```

are not reported as current numeric metrics.

### Retrieval benchmark size

The current retrieval benchmark contains only:

```text
12 questions
```

It is useful for regression and architecture comparison, but it is not presented as a comprehensive retrieval benchmark.

### Self-healing benchmark size

The repair benchmark contains:

```text
6 injected faults
```

It validates the repair mechanism on targeted failure classes but does not represent every possible Cypher failure.

### Graph integrity scope

The current graph audit verifies expected-versus-actual entity counts and configured orphan checks. It does not prove exact node-identity equality or exact relationship-endpoint equality against the processed corpus.

Stale-entity pruning currently executes one prune operation per entity label rather than bounded deletion batches. This is acceptable at the current measured corpus scale, but larger maintenance workloads should introduce bounded pruning and stronger identity/endpoint checks before claiming exact correspondence.

### Provider dependence

LLM-backed evaluation depends on an external provider.

Resumable execution mitigates interruption but does not eliminate provider availability as an operational dependency.

### Answer-suite scope

The current answer result is:

```text
160 / 160 passed
```

This establishes correctness for the current versioned cases under the current evaluator, configuration, source tree, and processed corpus.

It does not establish universal correctness for arbitrary financial questions, companies, filing types, time periods, or future model/provider behavior.

### Scale

The current scale is materially larger than the original development corpus but does not justify claims of internet-scale or distributed-scale processing.

The project reports measured scale rather than extrapolated scale.

---

# Evaluation Philosophy

## 47. Final principles

FinGraph Agent follows these evaluation rules:

```text
measure before optimizing
```

```text
evaluate layers independently
```

```text
do not confuse provider availability with model quality
```

```text
do not add architectural complexity without measurable benefit
```

```text
preserve benchmark methodology when comparing scale changes
```

```text
report limitations explicitly
```

```text
prefer reproducible metrics over anecdotal demos
```

The objective is not to produce the largest possible benchmark table.

The objective is to make each architectural claim traceable to reproducible evidence.