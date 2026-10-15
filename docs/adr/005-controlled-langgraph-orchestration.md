# ADR 005: Use Controlled LangGraph Routing Instead of an Agent Swarm

## Status

Accepted

## Context

FinGraph Agent must decide how to answer questions that may require:

- narrative SEC filing retrieval;
- structured financial facts;
- deterministic calculations;
- combinations of structured and narrative evidence.

A fully autonomous multi-agent architecture would introduce:

- additional LLM calls;
- higher latency;
- higher cost;
- more nondeterminism;
- harder evaluation;
- more failure states;
- more complicated tracing;
- more difficult reproducibility.

The problem does not currently require multiple independent agents negotiating or recursively delegating work to each other.

The orchestration layer should therefore remain explicit, bounded, inspectable, and testable.

## Decision

Use LangGraph as a controlled workflow with explicit routes:

```text
VECTOR
GRAPH
HYBRID
```

The router decides which evidence path is required.

The workflow uses typed state transitions and bounded tool execution.

Agentic behavior is limited to decisions that benefit from model reasoning, while deterministic work remains in ordinary application code.

## High-level flow

```text
user question
      |
      v
LangGraph router
      |
   +--+------------------+
   |                     |
   v                     v
VECTOR                 GRAPH
   |                     |
   |                     v
   |               Text-to-Cypher
   |                     |
   |                     v
   |                 validation
   |                     |
   |                     v
   |              bounded execution
   |                     |
   +----------+----------+
              |
              v
           HYBRID
        when required
              |
              v
     canonical evidence
              |
              v
 deterministic calculations
              |
              v
  grounded answer synthesis
```

## VECTOR route

The VECTOR route is primarily used for narrative filing questions.

Example:

```text
What risks does NVIDIA discuss in its filing?
```

Typical flow:

```text
question
   |
   v
retrieval planning
   |
   v
Qdrant
   |
   v
sparse / dense / hybrid retrieval
   |
   v
canonical narrative evidence
   |
   v
grounded synthesis
```

The route can use metadata such as:

```text
ticker
fiscal year
section
filing
```

to constrain retrieval.

## GRAPH route

The GRAPH route is used for structured financial questions.

Example:

```text
What was Microsoft's revenue in 2026?
```

Typical flow:

```text
question
   |
   v
Text-to-Cypher
   |
   v
static validation
   |
   v
Neo4j EXPLAIN
   |
   v
bounded execution
   |
   v
structured graph evidence
```

If the question requires arithmetic, graph evidence is passed to the deterministic calculation layer.

## HYBRID route

The HYBRID route is used when both structured and narrative evidence are required.

Example:

```text
How did Microsoft revenue change from 2025 to 2026,
and what factors drove the change?
```

Typical flow:

```text
question
   |
   +----------------------+
   |                      |
   v                      v
GRAPH                  VECTOR
   |                      |
   v                      v
financial facts      filing narrative
   |
   v
deterministic
calculation
   |                      |
   +----------+-----------+
              |
              v
       canonical evidence
              |
              v
      grounded synthesis
```

## Dependent hybrid execution

HYBRID does not always mean two independent branches.

Some questions require structured graph output to guide narrative retrieval.

Example flow:

```text
question
   |
   v
GRAPH
   |
   v
resolve:
company
metric
period
   |
   v
VECTOR
constrained by graph context
   |
   v
combined evidence
```

This is useful when the graph first determines the relevant entity or period, and the narrative search then uses that context.

## Why LangGraph

LangGraph is used because it provides explicit stateful orchestration.

Useful properties include:

- typed state;
- deterministic graph topology;
- explicit nodes;
- explicit transitions;
- observable intermediate states;
- bounded retries;
- resumability;
- controlled branching;
- easier evaluation.

The goal is not to maximize the number of agents.

The goal is to make model-driven control flow explicit.

## Bounded autonomy

The workflow permits model reasoning where it is useful, but autonomy is constrained.

Examples of bounded model behavior include:

```text
route selection
Text-to-Cypher generation
repair generation
answer synthesis
```

Examples of deterministic behavior include:

```text
financial calculations
identifier generation
persistence
retrieval execution
schema validation
Cypher safety checks
timeouts
result limits
state storage
```

## Router behavior

The router selects evidence requirements rather than directly generating the final answer.

Conceptually:

```text
narrative question
    -> VECTOR

structured financial question
    -> GRAPH

structured + narrative question
    -> HYBRID
```

The route is stored as explicit workflow state.

This makes route decisions auditable.

## Routing evaluation

Routing is evaluated independently from answer quality.

Current routing validation contains:

```text
80 route-audit cases
80 passed
0 failed
```

This separation is important because an answer can fail even when the route is correct, and a correct-looking answer should not hide a poor routing policy.

## Graph QA integration

The GRAPH route uses the dedicated graph QA subsystem.

The orchestration layer does not bypass graph safety.

Generated Cypher still passes through:

```text
static validator
schema restrictions
parameter validation
required result bounds
EXPLAIN preflight
query timeout
read-routed execution
```

If a supported failure occurs, bounded repair can be attempted.

## Self-healing integration

Self-healing is a controlled sub-flow, not a separate autonomous agent.

Conceptually:

```text
generated Cypher
      |
      v
validation / planning / execution
      |
   +--+--+
   |     |
success failure
   |     |
   |     v
   |  structured error
   |     |
   |     v
   |   repair
   |     |
   |     v
   |  retry once
   |     |
   +-----+
```

Repair attempts are bounded.

The system does not recursively retry until something happens to succeed.

## Self-healing benchmark

A dedicated injected-fault benchmark validates the repair path.

Current results:

```text
faults:                 6
baseline success:       0 / 6
repair success:         6 / 6
average repair attempts: 1.000
```

This makes self-healing measurable rather than a purely architectural claim.

## Canonical evidence boundary

All routes converge into a canonical evidence model.

This prevents answer synthesis from needing route-specific backend logic.

Conceptually:

```text
VECTOR evidence ----+
                    |
GRAPH evidence -----+--> canonical evidence
                    |
calculation evidence+
```

The synthesis layer receives a consistent downstream representation.

## Multi-company behavior

The orchestration layer supports questions involving multiple companies.

Graph execution can return multiple entities.

Narrative retrieval can diversify results across requested companies.

Evidence retains entity-level provenance.

This allows synthesis to distinguish which source supports which company-specific claim.

## Deterministic calculations

When a route requires arithmetic, LangGraph does not delegate the arithmetic to the LLM.

Instead:

```text
graph evidence
      |
      v
deterministic Python calculation
      |
      v
calculation evidence
      |
      v
synthesis
```

The model explains the result but does not authoritatively compute it.

## Durable execution

LLM-backed nodes use resumable execution state.

Logical LLM requests receive deterministic identities based on relevant inputs.

Persisted state allows:

```text
resume
deduplication
quota recovery
failure analysis
reproducibility
```

Provider quota failures can be represented as deferred execution rather than destroying the workflow state.

## Provider failure handling

Provider execution failures are distinct from quality failures.

Examples include:

```text
rate limit
daily quota exhaustion
temporary provider error
network failure
```

A recoverable provider limitation should not be reported as:

```text
retrieval failure
routing failure
graph failure
answer-quality failure
```

The orchestration layer preserves that distinction.

## Observability

LangGraph workflow execution is observable through application logs and Langfuse tracing.

Relevant trace context can include:

```text
request identity
selected route
LLM call
execution status
provider defer
latency
```

This makes the control flow inspectable during production execution.

## API integration

Both synchronous and streaming API paths invoke the same underlying orchestration logic.

Current endpoints:

```text
POST /ask
POST /ask/stream
```

The streaming endpoint provides Server-Sent Events.

Streaming behavior does not create a separate reasoning architecture.

## Why not a multi-agent swarm

A multi-agent swarm was rejected because the current problem does not require independent specialist agents negotiating or delegating work.

Adding agents such as:

```text
retrieval agent
graph agent
calculation agent
critic agent
citation agent
manager agent
```

would increase complexity without demonstrating a corresponding quality benefit.

Many of those responsibilities are better represented as deterministic modules or explicit workflow nodes.

## Why not a sequential chain

A simple sequential chain was also rejected.

Not every question should execute:

```text
retrieval
then graph
then calculation
then synthesis
```

A route-aware graph can avoid unnecessary operations.

For example, a simple structured revenue lookup should not require narrative retrieval.

## Why not let the model freely choose tools repeatedly

Unbounded tool choice was rejected because it creates:

- unpredictable execution depth;
- harder cost control;
- harder latency control;
- harder evaluation;
- difficult failure reproduction;
- increased risk of tool loops.

The system instead exposes controlled transitions.

## Consequences

This decision requires:

- route definitions;
- typed graph state;
- route evaluation;
- explicit tool interfaces;
- bounded retries;
- canonical evidence normalization;
- orchestration tests.

The benefit is a workflow that is:

```text
agentic
but bounded

model-driven
but inspectable

adaptive
but evaluable
```

## Current validation

The architecture has been validated across:

```text
VECTOR execution
GRAPH execution
HYBRID execution
dependent graph -> vector execution
multi-company evidence
deterministic calculations
self-healing
grounded synthesis
SSE serving
resumable LLM execution
```

The current release answer-evaluation report is publication-visible release evidence.

It completed 160 / 160 cases successfully under the current evaluator, evaluation configuration, source, corpus, dataset, and output contract.

```text
completed:           160
passed:              160
quality_failed:        0
llm_quality_failed:    0
deferred:              0
execution_failed:      0
accuracy:          1.000
```

This replaces the earlier historical/pending status for the answer-evaluation gate.

## Performance consequence

Different routes have different runtime characteristics.

Current formal API load results:

| Route | RPS | P50 | P95 | P99 | Errors |
|---|---:|---:|---:|---:|---:|
| GRAPH | 171.517 | 21 ms | 42 ms | 58 ms | 0 |
| VECTOR | 82.818 | 44 ms | 78 ms | 120 ms | 0 |
| HYBRID | 60.468 | 63 ms | 91 ms | 120 ms | 0 |

This is another reason to maintain explicit route semantics rather than hiding all questions behind one universal execution path.

## Reconsider when

This decision should be revisited if:

- genuinely independent agent roles become necessary;
- tasks require long-horizon planning across many tools;
- cross-agent specialization demonstrates measurable quality improvement;
- workflows require asynchronous distributed execution;
- a future use case cannot be represented cleanly as bounded state transitions.

Until then, controlled LangGraph orchestration remains preferable to an unconstrained multi-agent architecture.