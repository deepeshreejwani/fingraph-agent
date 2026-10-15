"""Safe graph question-answering execution service."""

from neo4j.exceptions import Neo4jError

from fingraph.graph.qa.derived_metrics import (
    requested_derived_financial_metric,
    validate_derived_metric_cypher,
    validate_derived_metric_rows,
)
from fingraph.graph.qa.models import (
    GraphQAResult,
)
from fingraph.graph.qa.schema import (
    GRAPH_SCHEMA,
)
from fingraph.graph.qa.validator import (
    CypherValidationError,
    validate_cypher,
    validate_result_rows,
)
from fingraph.llm.base import (
    CypherGenerator,
)
from fingraph.storage.neo4j import (
    Neo4jQueryValidationError,
    Neo4jStore,
)


class GraphQAError(RuntimeError):
    """Graph QA failed after bounded attempts."""


EMPTY_RESULT_REPAIR_MESSAGE = """
The Cypher query executed successfully but returned zero rows.

Re-check the query semantically.

In particular:
- use the canonical company ticker from the supplied schema;
- do not invent XBRL concept names;
- use the supplied canonical financial metric mappings;
- use FiscalPeriod.period_year for economic-year filtering;
- preserve the user's original intent.

If the question cannot be answered using the supplied graph schema without
inventing a company, metric, XBRL concept, label, relationship, or property,
return an empty Cypher string and an empty parameter list.

Generate a corrected query or a controlled unsupported result.
""".strip()


DERIVED_METRIC_EMPTY_PLAN_REPAIR_MESSAGE = """
The user asked for a supported derived financial metric, but the generated
Cypher was empty.

Do not calculate the derived metric in Cypher.

Retrieve every required raw canonical metric instead:
- operating margin -> operating_income + revenue
- net margin -> net_income + revenue

For multiple raw metrics in the same company and economic period:
- preserve fact.concept while selecting the latest filing observation;
- select the latest observation separately per fact.concept;
- RETURN latest.fact.concept AS metric_concept;
- return one row per required raw metric.

Generate a corrected raw-value query using only the supplied graph schema.
""".strip()


class GraphQAService:
    """Generate, validate, preflight and execute safe Cypher."""

    def __init__(
        self,
        *,
        store: Neo4jStore,
        generator: CypherGenerator,
        max_attempts: int = 2,
        max_rows: int = 25,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive.")

        self.store = store
        self.generator = generator
        self.max_attempts = max_attempts
        self.max_rows = max_rows

    def ask(
        self,
        question: str,
    ) -> GraphQAResult:
        """Answer a graph question with bounded self-healing."""

        previous_cypher: str | None = None
        previous_error: str | None = None

        for attempt in range(
            1,
            self.max_attempts + 1,
        ):
            plan = self.generator.generate_cypher(
                question=question,
                schema=GRAPH_SCHEMA,
                previous_cypher=previous_cypher,
                error=previous_error,
            )

            previous_cypher = plan.cypher

            # Empty Cypher is normally a controlled unsupported result.
            #
            # Supported derived margin intents are different: their base metrics
            # exist in the graph even though the derived percentage itself does
            # not. Give those questions one bounded semantic-repair opportunity.
            if not plan.cypher.strip():
                if (
                    requested_derived_financial_metric(question) is not None
                    and attempt < self.max_attempts
                ):
                    previous_error = DERIVED_METRIC_EMPTY_PLAN_REPAIR_MESSAGE
                    continue

                return GraphQAResult(
                    question=question,
                    cypher="",
                    parameters={},
                    rows=[],
                    attempts=attempt,
                )

            try:
                validated = validate_cypher(
                    plan,
                    max_rows=self.max_rows,
                    question=question,
                )

                validate_derived_metric_cypher(
                    question=question,
                    cypher=validated.cypher,
                    parameters=validated.parameters,
                )

                # Parse and plan the generated query
                # without executing it. This provides a
                # second safety/correctness boundary after
                # static application validation.
                self.store.explain_read(
                    validated.cypher,
                    parameters=validated.parameters,
                )

                rows = self.store.execute_read(
                    validated.cypher,
                    parameters=validated.parameters,
                )

                validate_result_rows(
                    question=question,
                    rows=rows,
                )

                validate_derived_metric_rows(
                    question=question,
                    rows=rows,
                )

                # A syntactically valid query can still
                # be semantically wrong. Give an empty
                # result one bounded repair opportunity.
                if not rows and attempt < self.max_attempts:
                    previous_error = EMPTY_RESULT_REPAIR_MESSAGE

                    continue

                return GraphQAResult(
                    question=question,
                    cypher=validated.cypher,
                    parameters=validated.parameters,
                    rows=rows,
                    attempts=attempt,
                )

            except (
                CypherValidationError,
                Neo4jQueryValidationError,
                Neo4jError,
            ) as exc:
                previous_error = str(exc)

        raise GraphQAError(
            f"Graph QA failed after {self.max_attempts} attempt(s). Last error: {previous_error}"
        )
