"""Langfuse tracing adapters for FinGraph application components."""

from __future__ import annotations

from typing import Literal

from langfuse import Langfuse

from fingraph.agent.evidence import Evidence
from fingraph.answer.models import (
    AnswerCitation,
    DeterministicCalculation,
    SynthesisPlan,
)
from fingraph.answer.synthesis import (
    SYNTHESIS_PROMPT_VERSION,
    GroqAnswerSynthesizer,
)
from fingraph.graph.qa.models import CypherPlan
from fingraph.llm.groq_client import (
    PROMPT_VERSION,
    GroqCypherGenerator,
)
from fingraph.llm.state import LLMStateStore


class TracedGroqCypherGenerator(GroqCypherGenerator):
    """Add Langfuse tracing around resumable Text-to-Cypher execution."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        state_store: LLMStateStore,
        reasoning_effort: str = "low",
        infrastructure_max_attempts: int = 4,
        max_retry_wait_seconds: float = 30.0,
        langfuse: Langfuse | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            state_store=state_store,
            reasoning_effort=reasoning_effort,
            infrastructure_max_attempts=(infrastructure_max_attempts),
            max_retry_wait_seconds=(max_retry_wait_seconds),
        )

        self._langfuse = langfuse

    def generate_cypher(
        self,
        *,
        question: str,
        schema: str,
        previous_cypher: str | None = None,
        error: str | None = None,
    ) -> CypherPlan:
        """Trace one resumable Text-to-Cypher task."""

        if self._langfuse is None:
            return super().generate_cypher(
                question=question,
                schema=schema,
                previous_cypher=previous_cypher,
                error=error,
            )

        with self._langfuse.start_as_current_observation(
            name="text-to-cypher",
            as_type="generation",
            model=self.model,
            model_parameters={
                "reasoning_effort": self.reasoning_effort,
            },
            input={
                "question": question,
                "repair": (previous_cypher is not None and error is not None),
            },
            metadata={
                "provider": "groq",
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "reasoning_effort": (self.reasoning_effort),
                "resumable": True,
            },
            version=PROMPT_VERSION,
        ) as observation:
            plan = super().generate_cypher(
                question=question,
                schema=schema,
                previous_cypher=previous_cypher,
                error=error,
            )

            usage = self.last_usage
            output = {
                "cypher": plan.cypher,
                "parameter_count": len(plan.parameters),
                "unsupported": (not plan.cypher.strip()),
            }

            if usage is None:
                observation.update(output=output)
            else:
                observation.update(
                    output=output,
                    usage_details=usage.langfuse_usage_details(),
                    cost_details=usage.langfuse_cost_details(),
                )

            return plan


class TracedGroqAnswerSynthesizer(GroqAnswerSynthesizer):
    """Add Langfuse tracing around grounded answer synthesis."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        state_store: LLMStateStore,
        reasoning_effort: str = "low",
        infrastructure_max_attempts: int = 4,
        max_retry_wait_seconds: float = 30.0,
        langfuse: Langfuse | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            state_store=state_store,
            reasoning_effort=reasoning_effort,
            infrastructure_max_attempts=(infrastructure_max_attempts),
            max_retry_wait_seconds=(max_retry_wait_seconds),
        )

        self._langfuse = langfuse

    def synthesize(
        self,
        *,
        question: str,
        evidence: list[Evidence],
    ) -> tuple[
        SynthesisPlan,
        list[DeterministicCalculation],
        list[AnswerCitation],
    ]:
        """Trace one resumable grounded-answer task."""

        if self._langfuse is None:
            return super().synthesize(
                question=question,
                evidence=evidence,
            )

        with self._langfuse.start_as_current_observation(
            name="grounded-answer-synthesis",
            as_type="generation",
            model=self.model,
            model_parameters={
                "reasoning_effort": self.reasoning_effort,
            },
            input={
                "question": question,
                "evidence_count": len(evidence),
                "evidence_ids": [item.evidence_id for item in evidence],
            },
            metadata={
                "provider": "groq",
                "model": self.model,
                "prompt_version": (SYNTHESIS_PROMPT_VERSION),
                "reasoning_effort": (self.reasoning_effort),
                "resumable": True,
            },
            version=SYNTHESIS_PROMPT_VERSION,
        ) as observation:
            (
                plan,
                calculations,
                citations,
            ) = super().synthesize(
                question=question,
                evidence=evidence,
            )

            usage = self.last_usage
            output = {
                "answer": plan.answer,
                "insufficient_evidence": (plan.insufficient_evidence),
                "citation_count": len(citations),
                "calculation_count": len(calculations),
            }

            if usage is None:
                observation.update(output=output)
            else:
                observation.update(
                    output=output,
                    usage_details=usage.langfuse_usage_details(),
                    cost_details=usage.langfuse_cost_details(),
                )

            return (
                plan,
                calculations,
                citations,
            )


ObservationType = Literal[
    "retriever",
    "agent",
]


class TracedEvidenceTool:
    """Trace an existing FinGraph evidence tool without changing its logic."""

    def __init__(
        self,
        *,
        tool: object,
        langfuse: Langfuse | None,
        name: str,
        observation_type: ObservationType,
    ) -> None:
        if not callable(tool):
            raise TypeError("tool must be callable.")

        self._tool = tool
        self._langfuse = langfuse
        self._name = name
        self._observation_type = observation_type

    def __call__(
        self,
        question: str,
    ) -> list[Evidence]:
        """Execute and trace one evidence tool."""

        tool = self._tool

        if not callable(tool):
            raise TypeError("tool must be callable.")

        if self._langfuse is None:
            result = tool(question)

            return list(result)

        with self._langfuse.start_as_current_observation(
            name=self._name,
            as_type=self._observation_type,
            input={
                "question": question,
            },
        ) as observation:
            result = list(tool(question))

            source_types = sorted({item.source_type for item in result})

            tickers = sorted({item.ticker for item in result if item.ticker is not None})

            row_count = 0

            for item in result:
                payload = item.structured_payload or {}

                rows = payload.get(
                    "rows",
                    [],
                )

                if isinstance(
                    rows,
                    list,
                ):
                    row_count += len(rows)

            observation.update(
                output={
                    "evidence_count": len(result),
                    "evidence_ids": [item.evidence_id for item in result],
                    "source_types": (source_types),
                    "tickers": tickers,
                    "row_count": row_count,
                }
            )

            return result
