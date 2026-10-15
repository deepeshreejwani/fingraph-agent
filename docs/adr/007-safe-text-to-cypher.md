# ADR 007: Layer Application Safeguards Around Text-to-Cypher

## Status

Accepted

## Context

FinGraph Agent converts natural-language financial questions into Cypher for structured graph querying.

LLM-generated Cypher cannot be trusted for direct unrestricted execution.

Potential problems include:

- invalid syntax;
- unsupported schema references;
- expensive queries;
- accidental mutation;
- administrative commands;
- unbounded result sets;
- incorrect relationship direction;
- nonexistent properties;
- unsupported procedures;
- unsafe query composition.

A production-oriented graph QA system therefore requires multiple safeguards between model generation and database execution.

## Decision

Use layered application safeguards before generated Cypher is executed.

Current protection layers include:

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

If a supported query fails during validation, planning, or execution, the system may attempt bounded self-healing.

The application does not send unrestricted model-generated Cypher directly to Neo4j.

## Static validation

Generated queries are inspected before execution.

The validator rejects unsupported patterns.

The validation layer is designed to prevent obvious unsafe or out-of-contract behavior before the query reaches Neo4j.

Examples of disallowed behavior include mutation-oriented operations and administrative commands.

The exact validation logic is intentionally deterministic and testable.

## Schema restrictions

Generated Cypher is constrained to the graph schema supported by FinGraph Agent.

Current graph entities include:

```text
Company
Filing
Section
FinancialFact
FiscalPeriod
```

Current core relationships include:

```text
(:Company)-[:FILED]->(:Filing)

(:Filing)-[:HAS_SECTION]->(:Section)

(:Company)-[:REPORTS]->(:FinancialFact)

(:Filing)-[:REPORTED_FACT]->(:FinancialFact)

(:FinancialFact)-[:FOR_PERIOD]->(:FiscalPeriod)
```

The model should not be allowed to invent arbitrary labels, relationship types, or properties and then execute them without validation.

## Parameter validation

Query parameters are validated separately from query structure.

This helps prevent generated query text from silently embedding arbitrary execution behavior where structured parameters are expected.

Structured values should remain structured values rather than being concatenated into free-form Cypher.

## Result bounds

Generated queries must be bounded.

The application enforces result limits so an LLM-generated query cannot casually return an unbounded result set.

This reduces:

- accidental full-graph scans;
- excessive response sizes;
- memory pressure;
- downstream synthesis overload.

Result bounding is part of execution safety, not merely a performance optimization.

## Neo4j EXPLAIN preflight

Before execution, supported generated queries pass through Neo4j `EXPLAIN`.

Conceptually:

```text
generated query
      |
      v
static validation
      |
      v
EXPLAIN
      |
   +--+--+
   |     |
 valid  invalid
   |     |
   v     v
execute reject / repair
```

This catches planning problems before the query is executed against the graph.

## Query timeout

Graph execution is bounded by a query timeout.

A generated query should not be allowed to run indefinitely.

Timeouts reduce the risk of:

```text
unexpectedly expensive traversal
planner mistakes
large intermediate results
resource exhaustion
```

A timeout is treated as an execution failure condition, not as permission to bypass safety checks.

## Read routing

Graph queries are executed using read-oriented driver routing.

This communicates the intended workload to Neo4j.

Read routing is one safeguard.

It is not equivalent to a database-level authorization boundary.

## Database-level authorization boundary

Application-level safeguards and database-level permissions are separate controls.

The local Neo4j Community deployment does not provide the desired project-specific database-level reader-role boundary.

FinGraph Agent therefore does not claim that the local graph execution path is protected by a true least-privilege database reader role.

The current guarantees are application-level defense in depth.

The managed Neo4j deployment should be described with the same conservative rule unless a dedicated serving identity with verified read-only privileges has been explicitly configured and tested.

The project must not imply that:

```text
read routing
+
query validation
```

is equivalent to:

```text
database authorization
```

## Why this distinction matters

Application validation reduces the likelihood that unsafe Cypher reaches the database.

Database authorization limits what the database will permit even if application validation fails.

The strongest production boundary would use both.

Current FinGraph guarantees the application layer.

A verified database-level least-privilege identity remains a separate infrastructure control.

## Bounded self-healing

Generated Cypher can fail because of issues such as:

- invalid property names;
- incorrect relationship directions;
- schema mismatch;
- syntax errors;
- planning errors.

The system permits bounded repair.

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
                repair LLM
                    |
                    v
                attempt 2
                    |
               +----+----+
               |         |
            success     stop
```

Self-healing does not bypass the original safeguards.

The repaired query must pass through the same validation and execution controls.

## Self-healing is not a security escape hatch

Repair logic receives structured feedback about the failure.

It is not allowed to respond to a validation failure by relaxing the validation policy.

For example:

```text
query rejected as mutation
```

must not become:

```text
disable mutation check and retry
```

Repair can change the query.

It cannot change the safety contract.

## Dedicated repair benchmark

The repair mechanism is evaluated separately from ordinary successful Text-to-Cypher execution.

Current injected-fault benchmark:

```text
fault cases:            6
baseline success:       0 / 6
self-healing success:   6 / 6
average repair attempts: 1.000
```

Measured repair overhead includes:

```text
mean added latency:        5,862.905 ms
added input tokens:           19,718
added cached-input tokens:     2,816
added output tokens:           2,351
added total tokens:           22,069
instrumented added cost:       $0.004
```

These values describe the six-case injected-fault benchmark and are not a general estimate for all Graph QA traffic.

This demonstrates that repair improves recoverability on the tested fault set while making the additional latency and token cost explicit.

## Why bounded repair

Unbounded repair loops were rejected because they introduce:

- unpredictable latency;
- uncontrolled token usage;
- repeated unsafe query generation;
- poor debuggability;
- difficult cost control;
- hard-to-reproduce behavior.

The system instead treats repair as a limited recovery mechanism.

## Error feedback

Repair receives structured execution feedback rather than an unrestricted dump of arbitrary internal state.

Useful feedback can include:

```text
validation error
planning error
schema error
execution error
```

The objective is to provide enough information for correction while preserving a controlled execution boundary.

## Supported query scope

Text-to-Cypher is intended for supported structured financial questions.

Examples include:

```text
company financial metric lookup
historical values
period comparisons
multi-company structured queries
```

It is not intended as a general unrestricted Neo4j console.

## Deterministic calculations

Cypher is responsible for retrieving structured facts.

Authoritative financial arithmetic is handled outside the LLM and outside arbitrary generated query arithmetic where possible.

For example:

```text
graph query
   |
   v
retrieve start and end financial values
   |
   v
deterministic Python calculation
```

This reduces the amount of generated logic that must be trusted inside Cypher.

## Temporal semantics

Graph QA must respect normalized fiscal-period semantics.

A query that returns the correct metric but from the wrong economic period is still incorrect.

The graph model therefore connects financial facts to normalized fiscal periods.

Text-to-Cypher generation is expected to query the appropriate economic-period representation instead of assuming filing-context fiscal-year metadata is always sufficient.

## Multi-company queries

The graph QA path supports structured questions involving more than one company.

Generated queries may return multiple company-specific rows.

The canonical evidence layer preserves entity identity so downstream synthesis can attribute values correctly.

Multi-company support does not change the safety contract.

## Result normalization

Raw Neo4j rows are not passed directly into free-form synthesis without structure.

Graph results are normalized into canonical evidence.

This permits:

```text
stable provenance
company attribution
period attribution
metric attribution
downstream citation grounding
```

## Observability

Graph generation and repair behavior can be traced.

Relevant execution information can include:

```text
request ID
generated query
validation outcome
EXPLAIN outcome
repair attempt
execution latency
provider status
```

Sensitive credentials are not part of trace content.

## Provider failures

Text-to-Cypher generation and repair depend on the LLM provider.

Provider rate limits or quota exhaustion are execution availability conditions.

They are not interpreted as:

```text
unsafe query
graph-quality failure
repair-quality failure
```

Recoverable provider failures can surface as a deferred execution outcome while the durable resumable request remains `pending` for later retry.

## Local and managed Neo4j

The same application safety model applies to both:

```text
local Neo4j
Neo4j Aura
```

The deployment environment may differ, but generated Cypher still passes through the same application controls.

Managed infrastructure should not be used as a reason to remove application-level validation.

## Defense in depth

The intended production security model is layered:

```text
natural-language question
      |
      v
controlled prompt / schema context
      |
      v
generated Cypher
      |
      v
static validator
      |
      v
schema restrictions
      |
      v
parameter validation
      |
      v
result limits
      |
      v
EXPLAIN
      |
      v
timeout
      |
      v
read-routed execution
      |
      v
database authorization when available
```

No individual safeguard is treated as sufficient on its own.

## Rejected alternatives

### Direct LLM-to-Neo4j execution

Rejected because model-generated queries are not trustworthy enough for unrestricted execution.

### Prompt-only safety

Rejected because instructions in the generation prompt are not a deterministic enforcement boundary.

### EXPLAIN only

Rejected because a query can be syntactically/planner-valid while still violating application policy.

### Read routing only

Rejected because driver routing expresses intent but is not a substitute for authorization and query validation.

### Unlimited repair

Rejected because it creates uncontrolled cost and behavior.

### Dynamic relaxation of safety rules

Rejected because a failed query should be repaired to satisfy the contract, not cause the contract to weaken.

## Consequences

This decision introduces:

- static validation logic;
- schema allow-listing;
- result-bound enforcement;
- parameter validation;
- EXPLAIN preflight;
- execution timeout;
- read routing;
- bounded repair;
- repair evaluation;
- additional graph QA tests.

The graph path is therefore more complex than direct LLM-generated query execution.

That complexity is intentional.

## Current guarantees

FinGraph Agent can currently claim:

```text
application-level Cypher validation
schema restrictions
bounded result size
EXPLAIN preflight
execution timeout
read-oriented routing
bounded repair
repair benchmark coverage
```

It should not claim:

```text
unrestricted Text-to-Cypher safety
formal query sandboxing
complete database-level isolation
verified least-privilege database authorization in every deployment
```

unless those controls are separately implemented and validated.

## Reconsider when

This decision should be revisited if:

- a dedicated database-level read-only serving identity is configured and verified;
- Neo4j authorization capabilities change the infrastructure boundary;
- query complexity materially increases;
- additional graph procedures are introduced;
- user-defined graph mutations become a legitimate product requirement;
- the application moves to a more general graph-agent model.

Even then, model-generated Cypher should continue to pass through deterministic application validation.