"""Groq-backed resumable Text-to-Cypher generation."""

import hashlib
import json
import random
import time
from typing import Any, cast
from uuid import uuid4

import groq
from groq import Groq

from fingraph.graph.qa.models import CypherPlan
from fingraph.llm.state import LLMStateStore
from fingraph.llm.usage import LLMUsage, extract_groq_usage

PROMPT_VERSION = "graph-cypher-v8"


SYSTEM_INSTRUCTIONS = """
You generate safe read-only Neo4j Cypher for a financial intelligence system.

Rules:

1. Use ONLY the supplied graph schema.

2. Generate exactly one read-only query.

3. Never use CREATE, MERGE, DELETE, DETACH DELETE, SET, REMOVE,
   DROP, ALTER, CALL, SHOW, LOAD CSV, FOREACH, UNION, USE, or
   administrative commands.

4. Use MATCH/OPTIONAL MATCH, WHERE, WITH, RETURN, ORDER BY,
   aggregation, and LIMIT only when needed.

5. Use Cypher parameters for user-provided ticker, concept, company
   names, dates, years, and literal filter values.

6. Every Cypher parameter must use exactly one of these representations.

   Scalar parameter:

   {
     "name": "ticker",
     "kind": "scalar",
     "scalar": "MSFT",
     "values": []
   }

   List parameter:

   {
     "name": "years",
     "kind": "list",
     "scalar": "",
     "values": ["2024", "2025", "2026"]
   }

7. All parameter values are represented as strings.

8. When Cypher uses IN $parameter, the parameter MUST use kind="list".

9. For scalar numeric comparisons, explicitly cast the parameter.

   WRONG:

   period.period_year = $year

   CORRECT:

   period.period_year = toInteger($year)

10. For numeric list comparisons, NEVER compare an integer property
    directly against a list of string parameters.

    WRONG:

    period.period_year IN $years

    CORRECT:

    period.period_year IN [
        value IN $years
        | toInteger(value)
    ]

11. Apply the same explicit conversion rule to any other numeric list
    parameter.

12. Use FiscalPeriod.period_year for the actual economic year.

13. FinancialFact.fiscal_year is SEC filing-context metadata and is NOT
    necessarily the economic year represented by the fact.

14. Comparative financial facts may appear in multiple filings.

15. For one observation per economic period, choose the observation
    associated with the latest Filing.filing_date for that period.

16. For canonical financial metrics, use ONLY exact XBRL concepts
    supplied in the graph schema. Never invent concept names.

17. Convert company names to the canonical ticker supplied in the
    graph schema.

18. Never invent labels, relationship types, or properties.

19. Never calculate financial ratios, growth rates, percentages, CAGR,
    or other derived metrics in Cypher. Return raw values so downstream
    deterministic Python code performs those calculations.

20. Keep returned result sets small and relevant.

21. When the user asks for an annual metric "from YEAR_A to YEAR_B",
    "between YEAR_A and YEAR_B", or over a year range, treat the range
    as inclusive.

22. For inclusive year ranges, prefer scalar start/end parameters.

    CORRECT:

    {
      "name": "start_year",
      "kind": "scalar",
      "scalar": "2024",
      "values": []
    }

    {
      "name": "end_year",
      "kind": "scalar",
      "scalar": "2026",
      "values": []
    }

    and:

    period.period_year >= toInteger($start_year)
    AND period.period_year <= toInteger($end_year)

23. Do NOT represent a continuous year range using only the two
    endpoint years in an IN list.

    WRONG:

    period.period_year IN [
        value IN $years
        | toInteger(value)
    ]

    where $years contains only ["2024", "2026"].

24. If the user explicitly asks for several non-contiguous years,
    use a list parameter instead.

25. For multi-company financial comparisons, ALWAYS return one raw observation
    for EVERY requested company before any downstream comparison is performed.
    This includes questions using words such as compare, higher, lower, highest,
    lowest, larger, smaller, more, or less.

26. Never select the winning or losing company inside Cypher. Do NOT use metric
    ordering, aggregation, filtering, or LIMIT 1 to reduce a multi-company
    comparison to one company. Downstream deterministic Python code performs
    winner/loser/ranking selection.

27. For every multi-company canonical financial result, RETURN the canonical
    company ticker and economic year explicitly using these aliases:

    company.ticker AS ticker
    period.period_year AS year

    Also return the requested raw metric value. When a Filing node is already
    used to select the latest observation, also return:

    filing.filing_date AS filing_date

    or the equivalent filing date from the selected latest observation.

28. A multi-company comparison is incomplete unless every requested company is
    represented in the returned rows. Preserve all requested companies even
    when the user asks which one is higher, lower, largest, or smallest.

29. Cypher WITH changes variable scope. Any variable referenced after a WITH
    clause MUST be explicitly carried through every intervening WITH clause.
    In particular, if RETURN uses company.ticker, preserve the company variable
    through every WITH. If RETURN uses period.period_year, preserve period.

30. When selecting the latest filing observation, prefer this scope-safe shape:

    WITH company, period, fact, filing
    ORDER BY period.period_year ASC, filing.filing_date DESC
    WITH company, period,
         collect({fact: fact, filing: filing})[0] AS latest
    RETURN company.ticker AS ticker,
           period.period_year AS year,
           latest.fact.value AS value,
           latest.fact.unit AS unit,
           latest.filing.filing_date AS filing_date

31. Never reference a variable after a WITH unless that variable, or an alias
    containing the required value, was explicitly projected by that WITH.
""".strip()


class LLMDeferredError(RuntimeError):
    """Transient provider or execution failure; the request remains resumable."""


class LLMQualityError(RuntimeError):
    """Model output exhausted bounded quality or deterministic validation."""


class LLMPermanentError(RuntimeError):
    """Provider rejected a request in a non-transient way."""


def _canonical_json(
    value: Any,
) -> str:
    """Serialize deterministically for stable hashing."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _sha256(
    value: str,
) -> str:
    """Return a deterministic SHA-256 hex digest."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _retry_after_seconds(
    error: groq.RateLimitError,
) -> float | None:
    """Extract Groq retry-after seconds when available."""

    response = getattr(
        error,
        "response",
        None,
    )

    if response is None:
        return None

    raw = response.headers.get("retry-after")

    if raw is None:
        return None

    try:
        return float(raw)

    except ValueError:
        return None


def _is_json_validation_failure(
    error: object,
) -> bool:
    """Return whether Groq rejected model output against the JSON schema."""

    return "json_validate_failed" in str(error).lower()


class GroqCypherGenerator:
    """Generate schema-constrained Cypher with durable checkpoints."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        state_store: LLMStateStore,
        reasoning_effort: str = "low",
        infrastructure_max_attempts: int = 4,
        max_retry_wait_seconds: float = 30.0,
        execution_lease_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.state_store = state_store
        self.reasoning_effort = reasoning_effort
        self.infrastructure_max_attempts = infrastructure_max_attempts
        self.max_retry_wait_seconds = max_retry_wait_seconds

        if execution_lease_seconds <= 0:
            raise ValueError("execution_lease_seconds must be positive.")

        self.execution_lease_seconds = execution_lease_seconds

        # Our orchestration owns retries so that retry state
        # remains visible and resumable in SQLite.
        self.client = Groq(
            api_key=api_key,
            max_retries=0,
            timeout=60.0,
        )

        self._last_usage: LLMUsage | None = None

    @property
    def last_usage(
        self,
    ) -> LLMUsage | None:
        """Return usage for the most recently resolved logical request."""

        return self._last_usage

    def _request_identity(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None,
        error: str | None,
    ) -> tuple[
        str,
        str,
        dict[str, Any],
    ]:
        """Create deterministic identity for one LLM request."""

        response_schema = CypherPlan.model_json_schema()

        response_schema_hash = _sha256(_canonical_json(response_schema))

        payload: dict[
            str,
            Any,
        ] = {
            "task_type": "graph_qa_cypher",
            "provider": "groq",
            "model": self.model,
            "prompt_version": PROMPT_VERSION,
            "response_schema_hash": (response_schema_hash),
            "question": question,
            "schema": schema,
            "previous_cypher": (previous_cypher),
            "error": error,
        }

        canonical = _canonical_json(payload)

        input_hash = _sha256(canonical)

        request_id = _sha256(f"fingraph:graph_qa_cypher:{input_hash}")

        return (
            request_id,
            input_hash,
            payload,
        )

    @staticmethod
    def _build_user_prompt(
        *,
        question: str,
        schema: str,
        previous_cypher: str | None,
        error: str | None,
    ) -> str:
        """Build generation or semantic-repair prompt."""

        parts = [
            "GRAPH SCHEMA:",
            schema,
            "",
            "USER QUESTION:",
            question,
        ]

        if previous_cypher is not None and error is not None:
            parts.extend(
                [
                    "",
                    "PREVIOUS CYPHER:",
                    previous_cypher,
                    "",
                    "VALIDATION OR EXECUTION ERROR:",
                    error,
                    "",
                    ("Repair the query while preserving the user's intent."),
                ]
            )

        return "\n".join(parts)

    def _backoff_seconds(
        self,
        attempt: int,
    ) -> float:
        """Calculate bounded exponential backoff with jitter."""

        base = min(
            2 ** (attempt - 1),
            self.max_retry_wait_seconds,
        )

        jitter = random.uniform(
            0.0,
            min(
                1.0,
                base * 0.25,
            ),
        )

        return float(
            min(
                base + jitter,
                self.max_retry_wait_seconds,
            )
        )

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        """Generate or resume one deterministic Text-to-Cypher request."""

        (
            request_id,
            input_hash,
            request_payload,
        ) = self._request_identity(
            question=question,
            schema=schema,
            previous_cypher=previous_cypher,
            error=error,
        )

        self.state_store.register(
            request_id=request_id,
            task_type="graph_qa_cypher",
            provider="groq",
            model=self.model,
            prompt_version=(PROMPT_VERSION),
            input_hash=input_hash,
            request_payload=(request_payload),
        )

        existing = self.state_store.get(request_id)

        if (
            existing is not None
            and existing.status == "succeeded"
            and existing.response_json is not None
        ):
            self._last_usage = existing.usage()

            return CypherPlan.model_validate_json(existing.response_json)

        if existing is not None and existing.status == "failed":
            if _is_json_validation_failure(existing.last_error):
                self.state_store.mark_pending(
                    request_id=request_id,
                    error=str(existing.last_error),
                    expected_status="failed",
                )

            else:
                raise LLMPermanentError(
                    "This deterministic request "
                    "previously failed permanently: "
                    f"{existing.last_error}"
                )

        owner_id = uuid4().hex

        claimed = self.state_store.claim_execution(
            request_id=request_id,
            owner_id=owner_id,
            lease_seconds=(self.execution_lease_seconds),
        )

        if not claimed:
            current = self.state_store.get(request_id)

            if (
                current is not None
                and current.status == "succeeded"
                and current.response_json is not None
            ):
                self._last_usage = current.usage()

                return CypherPlan.model_validate_json(current.response_json)

            if current is not None and current.status == "failed":
                raise LLMPermanentError(
                    "This deterministic request "
                    "failed while another caller "
                    "was executing it: "
                    f"{current.last_error}"
                )

            raise LLMDeferredError(
                "This deterministic Text-to-Cypher "
                "request is already being executed "
                "by another caller. Retry the same "
                "request later to reuse its durable "
                "result."
            )

        prompt = self._build_user_prompt(
            question=question,
            schema=schema,
            previous_cypher=previous_cypher,
            error=error,
        )

        for infrastructure_attempt in range(
            1,
            self.infrastructure_max_attempts + 1,
        ):
            attempt_started = self.state_store.begin_attempt(
                request_id,
                owner_id=owner_id,
                lease_seconds=(self.execution_lease_seconds),
            )

            if not attempt_started:
                current = self.state_store.get(request_id)

                if (
                    current is not None
                    and current.status == "succeeded"
                    and current.response_json is not None
                ):
                    self._last_usage = current.usage()

                    return CypherPlan.model_validate_json(current.response_json)

                raise LLMDeferredError(
                    "Execution ownership for this "
                    "deterministic Text-to-Cypher "
                    "request was lost. Retry the "
                    "same request to reuse or resume "
                    "its durable state."
                )

            try:
                completion = cast(
                    Any,
                    self.client.chat.completions.create,
                )(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (SYSTEM_INSTRUCTIONS),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    reasoning_effort=(self.reasoning_effort),
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": ("cypher_plan"),
                            "strict": True,
                            "schema": (CypherPlan.model_json_schema()),
                        },
                    },
                )

                content = completion.choices[0].message.content

                if content is None:
                    raise ValueError("Groq returned no structured content.")

                plan = CypherPlan.model_validate_json(content)

                usage = extract_groq_usage(
                    completion=completion,
                    model=self.model,
                )

                persisted = self.state_store.mark_succeeded(
                    request_id=request_id,
                    response_json=(plan.model_dump_json()),
                    usage=usage,
                    owner_id=owner_id,
                )

                if not persisted:
                    raise LLMDeferredError(
                        "Text-to-Cypher execution "
                        "completed after ownership "
                        "changed, so its result was "
                        "not persisted. Retry the "
                        "same deterministic request."
                    )

                self._last_usage = usage

                return plan

            except groq.RateLimitError as exc:
                retry_after = _retry_after_seconds(exc)

                if retry_after is None:
                    retry_after = self._backoff_seconds(infrastructure_attempt)

                error_message = f"Groq rate limit: {exc}"

                can_retry = (
                    infrastructure_attempt < self.infrastructure_max_attempts
                    and retry_after <= self.max_retry_wait_seconds
                )

                if can_retry:
                    time.sleep(retry_after)

                    continue

                self.state_store.mark_pending(
                    request_id=request_id,
                    error=error_message,
                    owner_id=owner_id,
                )

                raise LLMDeferredError(
                    "Groq quota/rate limit "
                    "interrupted this request. "
                    "Its checkpoint remains "
                    "pending; rerun the same "
                    "command later to continue."
                ) from exc

            except (
                groq.APITimeoutError,
                groq.APIConnectionError,
                groq.InternalServerError,
            ) as exc:
                error_message = f"{type(exc).__name__}: {exc}"

                if infrastructure_attempt < self.infrastructure_max_attempts:
                    time.sleep(self._backoff_seconds(infrastructure_attempt))

                    continue

                self.state_store.mark_pending(
                    request_id=request_id,
                    error=error_message,
                    owner_id=owner_id,
                )

                raise LLMDeferredError(
                    "Transient Groq failure "
                    "exhausted retries. "
                    "The request remains "
                    "pending and can be resumed."
                ) from exc

            except groq.APIStatusError as exc:
                error_message = f"Groq HTTP {exc.status_code}: {exc}"

                if exc.status_code == 400 and _is_json_validation_failure(exc):
                    if infrastructure_attempt < self.infrastructure_max_attempts:
                        time.sleep(self._backoff_seconds(infrastructure_attempt))

                        continue

                    self.state_store.mark_pending(
                        request_id=request_id,
                        error=error_message,
                        owner_id=owner_id,
                    )

                    raise LLMDeferredError(
                        "Groq structured-output "
                        "validation failed "
                        "repeatedly. The request "
                        "remains resumable."
                    ) from exc

                if exc.status_code >= 500:
                    if infrastructure_attempt < self.infrastructure_max_attempts:
                        time.sleep(self._backoff_seconds(infrastructure_attempt))

                        continue

                    self.state_store.mark_pending(
                        request_id=request_id,
                        error=error_message,
                        owner_id=owner_id,
                    )

                    raise LLMDeferredError("Groq server failure remains resumable.") from exc

                self.state_store.mark_failed(
                    request_id=request_id,
                    error=error_message,
                    owner_id=owner_id,
                )

                raise LLMPermanentError(error_message) from exc

            except LLMDeferredError:
                raise

            except (
                ValueError,
                TypeError,
            ) as exc:
                error_message = f"{type(exc).__name__}: {exc}"

                self.state_store.mark_failed(
                    request_id=request_id,
                    error=error_message,
                    owner_id=owner_id,
                )

                raise LLMPermanentError(error_message) from exc

        raise AssertionError("Unreachable retry state.")
