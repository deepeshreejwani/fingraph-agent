"""Answer-level evaluation for FinGraph."""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import groq
from groq import Groq
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from fingraph.agent import Evidence
from fingraph.answer.grounding import evaluate_company_citation_grounding
from fingraph.answer.models import (
    AnswerCitation,
    DeterministicCalculation,
    SynthesisPlan,
)
from fingraph.llm.groq_client import (
    LLMDeferredError,
    LLMPermanentError,
    LLMQualityError,
)
from fingraph.llm.state import (
    LLMStateStore,
)

ANSWER_EVALUATOR_VERSION = "answer-eval-v3"

ANSWER_JUDGE_PROMPT_VERSION = "answer-judge-v1"


RouteName = Literal[
    "vector",
    "graph",
    "hybrid",
]


class ExpectedCalculation(BaseModel):
    """Expected deterministic calculation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    start_year: int

    end_year: int

    start_value: str

    end_value: str

    absolute_change: str

    percentage_change: str | None = None


class AnswerGoldenCase(BaseModel):
    """One answer-level golden evaluation case."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    id: str = Field(
        min_length=1,
    )

    question: str = Field(
        min_length=1,
    )

    expected_route: RouteName | None = None

    expected_insufficient: bool = False

    min_citations: int = Field(
        default=1,
        ge=0,
    )

    expected_fiscal_years: tuple[
        int,
        ...,
    ] = ()

    expected_calculation: ExpectedCalculation | None = None

    reference_notes: tuple[
        str,
        ...,
    ] = ()

    semantic_judge: bool = True


class AnswerJudgeResult(BaseModel):
    """Structured semantic judgment of one answer."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    correct: bool

    grounded: bool

    complete: bool

    temporally_correct: bool

    unsupported_claims: list[str]

    reason: str


class AnswerContractResult(BaseModel):
    """Deterministic answer-contract result."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    passed: bool

    route_passed: bool

    insufficient_passed: bool

    citations_passed: bool

    citation_entity_passed: bool

    temporal_passed: bool

    calculation_passed: bool

    failure_categories: list[str]


JUDGE_SYSTEM_INSTRUCTIONS = """
You evaluate final answers produced by FinGraph, a financial intelligence
system grounded exclusively in supplied SEC evidence.

Evaluate ONLY against:
1. the user question;
2. the supplied evidence;
3. deterministic calculations;
4. reference notes supplied by the golden evaluation case.

Rules:

1. Do not use outside knowledge.
2. A factual statement is grounded only if supported by supplied evidence
   or a supplied deterministic calculation.
3. Do not penalize concise answers merely for being concise.
4. Mark correct=false if the answer contradicts the evidence, deterministic
   calculation, or reference notes.
5. Mark grounded=false if the answer contains factual claims not supported
   by the supplied evidence or deterministic calculations.
6. unsupported_claims must list unsupported factual claims briefly.
7. Mark complete=false only when the answer misses information necessary to
   answer the question based on the supplied evidence.
8. For temporal questions, verify that the answer uses the requested period
   or correctly describes the chronology represented by the evidence.
9. If expected_insufficient=true, the correct behavior is to state that
   sufficient grounded evidence is unavailable rather than invent an answer.
10. If expected_insufficient=false, do not mark an answer incorrect merely
    because it acknowledges reasonable evidence limitations.
11. Never judge hidden reasoning, Cypher style, implementation style, or prose
    elegance.
12. Return only the requested structured schema.
""".strip()


def _canonical_json(
    value: Any,
) -> str:
    """Serialize deterministically."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256(
    value: str,
) -> str:
    """Return deterministic SHA-256."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def case_fingerprint(
    case: AnswerGoldenCase,
) -> str:
    """Hash one golden case deterministically."""

    return _sha256(_canonical_json(case.model_dump(mode="json")))


def dataset_fingerprint(
    cases: list[AnswerGoldenCase],
) -> str:
    """Hash the complete ordered dataset."""

    return _sha256(_canonical_json([case.model_dump(mode="json") for case in cases]))


def evaluation_fingerprint(
    configuration: dict[str, Any],
) -> str:
    """Hash the effective answer-evaluation system configuration."""

    return _sha256(_canonical_json(configuration))


def load_report(
    path: Path,
) -> dict[str, Any] | None:
    """Load a previous evaluation report."""

    if not path.exists():
        return None

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if not isinstance(
        raw,
        dict,
    ):
        return None

    return raw


def write_report_atomic(
    *,
    path: Path,
    report: dict[str, Any],
) -> None:
    """Persist evaluation state atomically."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(path.suffix + ".tmp")

    temp_path.write_text(
        json.dumps(
            report,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temp_path.replace(path)


def reusable_result(
    *,
    previous: dict[str, Any] | None,
    case: AnswerGoldenCase,
    evaluation_hash: str,
    rerun_failed: bool,
) -> bool:
    """Return whether one prior case may safely be skipped."""

    if previous is None:
        return False

    if previous.get("case_hash") != case_fingerprint(case):
        return False

    if previous.get("evaluator_version") != ANSWER_EVALUATOR_VERSION:
        return False

    if previous.get("evaluation_hash") != evaluation_hash:
        return False

    if previous.get("status") != "completed":
        return False

    return not (
        rerun_failed
        and not bool(
            previous.get(
                "passed",
                False,
            )
        )
    )


def _decimal_equal(
    first: str,
    second: str,
) -> bool:
    """Compare decimal strings without formatting sensitivity."""

    try:
        return Decimal(first) == Decimal(second)

    except InvalidOperation:
        return False


def _evidence_years(
    evidence: list[Evidence],
    cited_ids: set[str],
) -> set[int]:
    """Collect economic/provenance years represented by cited evidence."""

    years: set[int] = set()

    year_keys = {
        "year",
        "fiscal_year",
        "economic_year",
        "period_year",
    }

    for item in evidence:
        if item.evidence_id not in cited_ids:
            continue

        if item.source_type == "vector":
            raw_year = item.provenance.get("fiscal_year")

            if raw_year is not None:
                with contextlib.suppress(ValueError):
                    years.add(int(str(raw_year)))

            continue

        payload = item.structured_payload or {}

        rows = payload.get(
            "rows",
            [],
        )

        if not isinstance(
            rows,
            list,
        ):
            continue

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            for key, raw in row.items():
                if key.lower() not in year_keys:
                    continue

                with contextlib.suppress(
                    TypeError,
                    ValueError,
                ):
                    years.add(int(str(raw)))

    return years


def _calculation_matches(
    *,
    expected: ExpectedCalculation,
    calculations: list[DeterministicCalculation],
) -> bool:
    """Verify deterministic financial arithmetic exactly."""

    for actual in calculations:
        if actual.start_year != expected.start_year:
            continue

        if actual.end_year != expected.end_year:
            continue

        if actual.start_value is None or actual.end_value is None or actual.absolute_change is None:
            continue

        if not _decimal_equal(
            actual.start_value,
            expected.start_value,
        ):
            continue

        if not _decimal_equal(
            actual.end_value,
            expected.end_value,
        ):
            continue

        if not _decimal_equal(
            actual.absolute_change,
            expected.absolute_change,
        ):
            continue

        if expected.percentage_change is None:
            return actual.percentage_change is None

        if actual.percentage_change is None:
            continue

        if not _decimal_equal(
            actual.percentage_change,
            expected.percentage_change,
        ):
            continue

        return True

    return False


def evaluate_answer_contract(
    *,
    case: AnswerGoldenCase,
    route: str,
    plan: SynthesisPlan,
    citations: list[AnswerCitation],
    calculations: list[DeterministicCalculation],
    evidence: list[Evidence],
) -> AnswerContractResult:
    """Run deterministic answer-level checks."""

    route_passed = case.expected_route is None or route == case.expected_route

    insufficient_passed = plan.insufficient_evidence == case.expected_insufficient

    evidence_ids = {item.evidence_id for item in evidence}

    citation_ids = {item.evidence_id for item in citations}

    citations_passed = len(citations) >= case.min_citations and citation_ids.issubset(evidence_ids)

    for citation in citations:
        if not citation.filing_id or not citation.source_url:
            citations_passed = False

            break

    citation_entity_result = evaluate_company_citation_grounding(
        answer=plan.answer,
        cited_evidence_ids=plan.cited_evidence_ids,
        evidence=evidence,
    )

    citation_entity_passed = citation_entity_result.passed

    temporal_passed = True

    if case.expected_fiscal_years:
        represented_years = _evidence_years(
            evidence,
            citation_ids,
        )

        temporal_passed = set(case.expected_fiscal_years).issubset(represented_years)

    calculation_passed = True

    if case.expected_calculation is not None:
        calculation_passed = _calculation_matches(
            expected=(case.expected_calculation),
            calculations=(calculations),
        )

    failures: list[str] = []

    if not route_passed:
        failures.append("route_mismatch")

    if not insufficient_passed:
        failures.append("insufficient_evidence_mismatch")

    if not citations_passed:
        failures.append("citation_failure")

    if not citation_entity_passed:
        failures.append("citation_entity_failure")

    if not temporal_passed:
        failures.append("temporal_failure")

    if not calculation_passed:
        failures.append("calculation_failure")

    return AnswerContractResult(
        passed=not failures,
        route_passed=(route_passed),
        insufficient_passed=(insufficient_passed),
        citations_passed=(citations_passed),
        citation_entity_passed=(citation_entity_passed),
        temporal_passed=(temporal_passed),
        calculation_passed=(calculation_passed),
        failure_categories=(failures),
    )


def _bounded_evidence(
    evidence: list[Evidence],
) -> list[dict[str, Any]]:
    """Build bounded evidence input for semantic judging."""

    payload: list[dict[str, Any]] = []

    for item in evidence:
        text = item.text

        if text is not None and len(text) > 3000:
            text = text[:3000] + "..."

        payload.append(
            {
                "evidence_id": (item.evidence_id),
                "source_type": (item.source_type),
                "ticker": (item.ticker),
                "filing_id": (item.filing_id),
                "section_name": (item.section_name),
                "text": text,
                "structured_payload": (item.structured_payload),
                "source_url": (item.source_url),
                "provenance": (item.provenance),
            }
        )

    return payload


def _retry_after_seconds(
    error: groq.RateLimitError,
) -> float | None:
    """Extract provider retry-after header."""

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


def _is_retryable_judge_validation_error(error: object) -> bool:
    """Classify structured judge-output failures that can succeed on regeneration."""

    text = str(error).lower()

    markers = (
        "json_validate_failed",
        "output_parse_failed",
        "parsing failed",
        "failed_generation",
        "groq returned no structured judge content",
        "validation error",
    )

    return any(marker in text for marker in markers)


class GroqAnswerJudge:
    """Resumable semantic judge for answer evaluation."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        state_store: LLMStateStore,
        reasoning_effort: str = "low",
        infrastructure_max_attempts: int = 4,
        max_retry_wait_seconds: float = 30.0,
    ) -> None:
        self.model = model

        self.state_store = state_store

        self.reasoning_effort = reasoning_effort

        self.infrastructure_max_attempts = infrastructure_max_attempts

        self.max_retry_wait_seconds = max_retry_wait_seconds

        self.client = Groq(
            api_key=api_key,
            max_retries=0,
            timeout=60.0,
        )

    def _backoff_seconds(
        self,
        attempt: int,
    ) -> float:
        """Calculate bounded retry delay."""

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

    def _request_identity(
        self,
        *,
        case: AnswerGoldenCase,
        answer: str,
        insufficient_evidence: bool,
        citations: list[AnswerCitation],
        calculations: list[DeterministicCalculation],
        evidence: list[Evidence],
    ) -> tuple[
        str,
        str,
        dict[str, Any],
    ]:
        """Build deterministic semantic-judge request identity."""

        response_schema = AnswerJudgeResult.model_json_schema()

        response_schema_hash = _sha256(_canonical_json(response_schema))

        payload = {
            "task_type": ("answer_evaluation_judge"),
            "provider": "groq",
            "model": self.model,
            "prompt_version": (ANSWER_JUDGE_PROMPT_VERSION),
            "response_schema_hash": (response_schema_hash),
            "case": (case.model_dump(mode="json")),
            "answer": answer,
            "insufficient_evidence": (insufficient_evidence),
            "citations": [item.model_dump(mode="json") for item in citations],
            "calculations": [item.model_dump(mode="json") for item in calculations],
            "evidence": (_bounded_evidence(evidence)),
        }

        canonical = _canonical_json(payload)

        input_hash = _sha256(canonical)

        request_id = _sha256(f"fingraph:answer_evaluation_judge:{input_hash}")

        return (
            request_id,
            input_hash,
            payload,
        )

    def evaluate(
        self,
        *,
        case: AnswerGoldenCase,
        answer: str,
        insufficient_evidence: bool,
        citations: list[AnswerCitation],
        calculations: list[DeterministicCalculation],
        evidence: list[Evidence],
    ) -> AnswerJudgeResult:
        """Evaluate or resume semantic answer judgment."""

        (
            request_id,
            input_hash,
            payload,
        ) = self._request_identity(
            case=case,
            answer=answer,
            insufficient_evidence=(insufficient_evidence),
            citations=(citations),
            calculations=(calculations),
            evidence=evidence,
        )

        self.state_store.register(
            request_id=request_id,
            task_type=("answer_evaluation_judge"),
            provider="groq",
            model=self.model,
            prompt_version=(ANSWER_JUDGE_PROMPT_VERSION),
            input_hash=input_hash,
            request_payload=payload,
        )

        existing = self.state_store.get(request_id)

        if (
            existing is not None
            and existing.status == "succeeded"
            and existing.response_json is not None
        ):
            return AnswerJudgeResult.model_validate_json(existing.response_json)

        if existing is not None and existing.status == "failed":
            if _is_retryable_judge_validation_error(existing.last_error):
                raise LLMQualityError(
                    "This deterministic "
                    "answer-judge request previously "
                    "exhausted bounded structured-output "
                    "validation: "
                    f"{existing.last_error}"
                )

            raise LLMPermanentError(
                "This deterministic "
                "answer-judge request "
                "previously failed permanently: "
                f"{existing.last_error}"
            )

        owner_id = f"answer-judge:{uuid4()}"

        if not self.state_store.claim_execution(
            request_id=request_id,
            owner_id=owner_id,
        ):
            current = self.state_store.get(request_id)

            if (
                current is not None
                and current.status == "succeeded"
                and current.response_json is not None
            ):
                return AnswerJudgeResult.model_validate_json(current.response_json)

            if current is not None and current.status == "failed":
                if _is_retryable_judge_validation_error(current.last_error):
                    raise LLMQualityError(
                        "This deterministic "
                        "answer-judge request previously "
                        "exhausted bounded structured-output "
                        "validation: "
                        f"{current.last_error}"
                    )

                raise LLMPermanentError(
                    "This deterministic "
                    "answer-judge request "
                    "previously failed permanently: "
                    f"{current.last_error}"
                )

            raise LLMDeferredError(
                "This deterministic answer-judge request is already being executed "
                "by another worker and remains resumable."
            )

        user_prompt = _canonical_json(
            {
                "question": (case.question),
                "expected_insufficient": (case.expected_insufficient),
                "reference_notes": (list(case.reference_notes)),
                "answer": answer,
                "insufficient_evidence": (insufficient_evidence),
                "citations": [item.model_dump(mode="json") for item in citations],
                "deterministic_calculations": [
                    item.model_dump(mode="json") for item in calculations
                ],
                "evidence": (_bounded_evidence(evidence)),
            }
        )

        for attempt in range(
            1,
            self.infrastructure_max_attempts + 1,
        ):
            if not self.state_store.begin_attempt(
                request_id,
                owner_id=owner_id,
            ):
                raise LLMDeferredError(
                    "Answer-judge execution ownership was lost before the provider call; "
                    "the deterministic request remains resumable."
                )

            try:
                completion = cast(Any, self.client.chat.completions.create)(
                    model=self.model,
                    messages=[
                        {
                            "role": ("system"),
                            "content": (JUDGE_SYSTEM_INSTRUCTIONS),
                        },
                        {
                            "role": ("user"),
                            "content": (user_prompt),
                        },
                    ],
                    reasoning_effort=(self.reasoning_effort),
                    response_format={
                        "type": ("json_schema"),
                        "json_schema": {
                            "name": ("answer_judgment"),
                            "strict": True,
                            "schema": (AnswerJudgeResult.model_json_schema()),
                        },
                    },
                )

                content = completion.choices[0].message.content

                if content is None:
                    raise ValueError("Groq returned no structured judge content.")

                result = AnswerJudgeResult.model_validate_json(content)

                if not self.state_store.mark_succeeded(
                    request_id=(request_id),
                    response_json=(result.model_dump_json()),
                    owner_id=owner_id,
                ):
                    raise LLMDeferredError(
                        "Answer-judge execution ownership was lost before success "
                        "could be persisted; the deterministic request remains resumable."
                    )

                return result

            except groq.RateLimitError as exc:
                retry_after = _retry_after_seconds(exc)

                if retry_after is None:
                    retry_after = self._backoff_seconds(attempt)

                if (
                    attempt < self.infrastructure_max_attempts
                    and retry_after <= self.max_retry_wait_seconds
                ):
                    time.sleep(retry_after)

                    continue

                self.state_store.mark_pending(
                    request_id=(request_id),
                    error=(f"Groq rate limit: {exc}"),
                    owner_id=owner_id,
                )

                raise LLMDeferredError(
                    "Answer judge was interrupted "
                    "by the Groq quota/rate limit. "
                    "The request remains resumable."
                ) from exc

            except (
                groq.APITimeoutError,
                groq.APIConnectionError,
                groq.InternalServerError,
            ) as exc:
                if attempt < self.infrastructure_max_attempts:
                    time.sleep(self._backoff_seconds(attempt))

                    continue

                self.state_store.mark_pending(
                    request_id=(request_id),
                    error=(f"{type(exc).__name__}: {exc}"),
                    owner_id=owner_id,
                )

                raise LLMDeferredError("Transient answer-judge failure remains resumable.") from exc

            except groq.APIStatusError as exc:
                error_message = f"Groq HTTP {exc.status_code}: {exc}"

                if exc.status_code == 400 and _is_retryable_judge_validation_error(exc):
                    if attempt < self.infrastructure_max_attempts:
                        time.sleep(self._backoff_seconds(attempt))

                        continue

                    self.state_store.mark_failed(
                        request_id=request_id,
                        error=error_message,
                        owner_id=owner_id,
                    )

                    raise LLMQualityError(
                        "Groq answer-judge structured-output "
                        "validation exhausted bounded retry attempts."
                    ) from exc

                if exc.status_code >= 500:
                    if attempt < self.infrastructure_max_attempts:
                        time.sleep(self._backoff_seconds(attempt))

                        continue

                    self.state_store.mark_pending(
                        request_id=(request_id),
                        error=(error_message),
                        owner_id=owner_id,
                    )

                    raise LLMDeferredError(
                        "Groq answer-judge server failure remains resumable."
                    ) from exc

                self.state_store.mark_failed(
                    request_id=(request_id),
                    error=(error_message),
                    owner_id=owner_id,
                )

                raise LLMPermanentError(error_message) from exc

            except (
                ValueError,
                TypeError,
            ) as exc:
                error_message = f"{type(exc).__name__}: {exc}"

                if (
                    _is_retryable_judge_validation_error(exc)
                    and attempt < self.infrastructure_max_attempts
                ):
                    time.sleep(self._backoff_seconds(attempt))

                    continue

                if _is_retryable_judge_validation_error(exc):
                    self.state_store.mark_failed(
                        request_id=request_id,
                        error=error_message,
                        owner_id=owner_id,
                    )

                    raise LLMQualityError(
                        "Answer judge failed structured validation after bounded retry attempts."
                    ) from exc

                self.state_store.mark_failed(
                    request_id=(request_id),
                    error=(error_message),
                    owner_id=owner_id,
                )

                raise LLMPermanentError(error_message) from exc

        raise AssertionError("Unreachable answer-judge retry state.")
