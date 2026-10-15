# ADR 006: Make LLM Execution Resumable and Idempotent

## Status

Accepted

## Context

LLM-backed workflows can be interrupted by:

- provider rate limits;
- daily token quotas;
- transient provider errors;
- network failures;
- local process termination;
- API restarts;
- long evaluation runs;
- deployment restarts.

Large evaluations become wasteful if every interruption requires restarting completed work.

Repeated execution also creates several risks:

```text
duplicate provider calls
unnecessary cost
inconsistent retries
repeated completed work
hard-to-audit execution history
```

FinGraph Agent therefore requires LLM execution to be resumable, idempotent at the logical-request level, and observable.

## Decision

Persist LLM request state.

Assign each logical LLM request a deterministic identity derived from the logical inputs that define the request.

Store request state in SQLite.

Use the persisted state to:

- reuse successful work;
- resume interrupted work;
- distinguish deferred provider execution from application-quality failures;
- avoid blindly repeating completed requests;
- support reproducible evaluation.

## Logical request identity

A logical LLM request receives a deterministic identity based on relevant inputs such as:

```text
task
prompt version
model
response schema
logical payload
```

The identity should change when inputs that materially affect model behavior change.

This allows the persistence layer to distinguish:

```text
same logical request
```

from:

```text
new request requiring fresh execution
```

## Why deterministic identity matters

Without deterministic request IDs, two identical logical operations can appear unrelated.

That makes it difficult to:

- deduplicate completed calls;
- resume partially completed evaluations;
- associate failures with exact inputs;
- compare prompt versions;
- reproduce provider behavior;
- avoid repeated cost.

Deterministic request identity provides the boundary for resumability.

## Persistence model

LLM execution state is stored in SQLite.

Typical persisted information can include:

```text
request ID
task
prompt version
model
logical payload identity
execution status
attempt count
provider result
structured response
error information
timestamps
```

The persistence layer is part of application execution, not merely a temporary debugging cache.

## Execution states

The durable persistence layer stores request states such as:

```text
pending
running
succeeded
failed
```

### pending

The logical request exists but has not completed successfully or permanently failed.

A recoverable provider limitation can leave the durable request in `pending` so the same logical request can be resumed later.

### running

The request is currently being processed by the active execution owner.

### succeeded

The provider returned a valid response and the application successfully processed it.

A succeeded request can be reused when the logical request identity remains current.

### failed

The execution encountered a non-recoverable or application-level failure.

Examples can include:

```text
invalid structured response after bounded handling
schema violation after bounded handling
non-retryable provider/client rejection
```

A **deferred** result is an API/evaluation outcome, not a distinct durable SQLite request status. It indicates that execution could not currently complete because of a recoverable external condition such as a rate limit, quota, temporary provider failure, or connection problem. The durable request remains resumable rather than being misclassified as an answer-quality failure.

## Execution flow

Conceptually:

```text
logical request
      |
      v
deterministic request ID
      |
      v
check persistent state
      |
  +---+--------------------------+
  |                              |
  v                              v
succeeded?                      not complete
  |                              |
  v                              v
reuse result                 provider call
                                 |
                      +----------+----------+
                      |          |          |
                      v          v          v
                   success    deferred    failure
                      |          |          |
                      v          v          v
                  succeeded   pending     failed
```

## Idempotency behavior

Idempotency applies at the logical-request level.

If the same logical request has already succeeded, the application can reuse the persisted result rather than issuing another provider call.

This is particularly important for:

- answer evaluation;
- routing evaluation;
- evidence evaluation;
- Text-to-Cypher generation;
- repair generation;
- answer synthesis.

The architecture does not assume that the external provider itself provides request-level idempotency.

The application owns the logical deduplication boundary.

## Prompt versioning

Prompt behavior can change over time.

A persisted response from an older prompt should not automatically be reused after a meaningful prompt change.

Prompt version is therefore part of the logical execution identity where appropriate.

Examples include:

```text
answer-eval-v2
grounded-answer-v3
```

This permits controlled cache invalidation.

Changing a prompt version creates a new logical request identity and therefore requires fresh evaluation for affected cases.

## Schema versioning

Structured LLM responses can also change.

The logical request identity should account for response-schema changes where they materially alter expected output.

A persisted response produced under an incompatible schema should not be silently reused.

## Bounded retries

Provider execution uses bounded retry behavior.

Retries can use:

```text
bounded attempt count
backoff
provider-specific error classification
```

The system does not retry indefinitely.

A recoverable quota or rate condition can surface as a deferred execution outcome while the durable request remains `pending` for later resumption.

This prevents long-running workflows from becoming stuck inside uncontrolled retry loops.

## Provider rate and quota failures

Provider quota and rate failures are treated as execution availability conditions.

They should not be reported as:

```text
answer-quality failures
retrieval failures
routing failures
graph failures
```

The application therefore maps recoverable provider limitations to a dedicated deferred execution outcome while keeping the durable resumable request `pending`.

This distinction is essential for reliable evaluation.

## Evaluation semantics

An evaluation run can contain:

```text
passed
quality failure
deferred
execution failure
```

These states have different meanings.

### Passed

The case executed successfully and satisfied its quality criteria.

### Quality failure

Execution completed, but the result failed the defined quality checks.

### Deferred

The case could not execute because of a recoverable provider condition.

### Execution failure

The application or provider interaction failed in a way that is not classified as a quality result.

This prevents infrastructure availability from contaminating answer-quality metrics.

## Resume behavior

Evaluation reruns should avoid repeating successful current cases.

Conceptually:

```text
evaluation case
      |
      v
derive current request identity
      |
      v
existing succeeded result?
      |
   +--+--+
   |     |
  yes    no
   |     |
   v     v
reuse   execute
```

This allows long evaluation suites to resume efficiently.

## Fresh execution

A full fresh execution should only be used when there is a reason to invalidate existing successful results.

Examples include:

- prompt changes;
- response-schema changes;
- model changes;
- synthesis logic changes;
- calculation-contract changes;
- intentionally controlled re-benchmarking.

Routine reruns should preserve resumability.

## Current answer-evaluation state

The golden answer dataset contains:

```text
160 cases
```

The current release answer-evaluation report completed all 160 cases successfully under the current evaluator, evaluation configuration, source, corpus, dataset, and output contract.

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

The current `160 / 160` result is therefore publication-visible release evidence. Evaluation reuse remains valid only when the evaluation fingerprint, per-case identity, and output contract still match.

For a matching current evaluation fingerprint, successful cases can still be reused rather than unnecessarily repeating LLM execution.

## Runtime API behavior

The same persistence model is useful during API execution.

A deterministic request identity allows provider interactions to remain inspectable and recoverable across application restarts where appropriate.

The persistence layer is not used to fabricate an answer when execution is incomplete.

Incomplete states remain explicit.

## Railway persistence

The deployed Railway environment stores the SQLite execution database on a persistent mounted volume.

Current deployment configuration uses:

```text
/data/fingraph_runs.db
```

for durable execution state.

This means request and evaluation state can survive normal application/container restarts.

The persistence was validated across a Railway restart.

## Local persistence

Local development can use a local SQLite path.

The logical persistence semantics remain the same across local and deployed environments.

The storage location can differ without changing the request-state model.

## Concurrency considerations

Persistent request state may be accessed by multiple execution paths.

The implementation must therefore avoid treating SQLite as an uncoordinated dictionary.

State transitions should preserve a consistent lifecycle.

Important concerns include:

```text
duplicate simultaneous execution
partial state updates
stale running states
atomic result persistence
```

Before an external provider call, a worker atomically claims the logical request using an execution-owner ID and lease expiry.

The active owner can renew the lease while work remains in progress.

A different worker may reclaim the request only after the lease expires. This prevents two healthy workers from issuing the same provider call concurrently while still allowing recovery when a worker dies or abandons a `running` request.

Conceptually:

```text
pending
  |
  v
atomic claim(owner_id, lease_expiry)
  |
  v
running
  |
  +--> owner renews lease while active
  |
  +--> success ----------------------> succeeded
  |
  +--> permanent failure ------------> failed
  |
  +--> recoverable provider defer ---> pending
  |
  +--> owner disappears
          |
          v
     lease expires
          |
          v
   another worker may reclaim
```

Ownership and leases complement deterministic request identity:

```text
deterministic identity
  -> identifies the logical request

transactional claim
  -> establishes the current execution owner

lease expiry
  -> bounds ownership in time

reclaim after expiry
  -> recovers abandoned work
```

The application should prefer deterministic request identity plus transactional ownership/state updates over best-effort in-memory deduplication.

## Observability integration

LLM execution state is observable through tracing and application logs.

Langfuse tracing can include context such as:

```text
request identity
task
model
route
execution status
provider defer
latency
```

A provider defer can therefore be inspected separately from an answer-quality failure.

## Self-healing integration

Text-to-Cypher repair is also LLM-backed.

Repair generation therefore benefits from the same principles:

```text
deterministic logical request identity
bounded retries
persistent result
provider defer handling
```

Self-healing should not silently issue unlimited repeated model calls.

## Evaluation integration

The persistence design is particularly important for evaluation because evaluations can be:

```text
large
costly
long-running
provider-limited
interrupted
```

Without persistence, a provider quota failure near the end of a run could require unnecessarily repeating all previous successful cases.

With resumability, only incomplete current cases require further execution.

## Why SQLite

SQLite is used because the current workload requires:

```text
durability
simple transactional state
local portability
low operational overhead
easy inspection
```

The project does not currently require a distributed state database solely for LLM execution tracking.

SQLite is sufficient for the current scale and deployment topology.

## Why not in-memory state

In-memory state was rejected because it is lost when:

```text
the process exits
the API restarts
a deployment restarts
an evaluation is interrupted
```

That would defeat resumability.

## Why not filesystem JSON state

Standalone JSON files were rejected as the primary execution-state system because concurrent and incremental state transitions are more naturally represented through transactional storage.

SQLite provides a clearer lifecycle and update model.

## Why not rely on provider request history

Provider-side request history is not the application's logical state.

It may not encode:

```text
prompt version
application task
response schema
evaluation case
structured validation state
application retry semantics
```

The application therefore maintains its own durable execution ledger.

## Consequences

This decision adds:

- a persistence layer;
- deterministic request-ID generation;
- execution-state transitions;
- retry classification;
- resume logic;
- prompt-version awareness;
- additional tests;
- deployment persistence requirements.

The benefits are:

```text
lower repeated cost
safe resumability
clear execution semantics
reproducible evaluation
better failure diagnosis
provider-quota resilience
```

## Failure semantics

A persisted state must never be interpreted more strongly than it actually means.

For example:

```text
deferred outcome != failed quality
pending durable state != permanent failure
running != succeeded
provider response != validated structured result
```

Only validated successful execution should be marked succeeded.

## Reconsider when

This decision should be revisited if:

- request volume exceeds SQLite's practical concurrency envelope;
- multiple distributed workers require strongly coordinated execution state;
- the application needs cross-region state;
- large-scale asynchronous job execution is introduced;
- state retention requirements justify a managed transactional database.

A future persistence backend may replace SQLite.

The deterministic logical-request model and explicit execution-state semantics should remain.