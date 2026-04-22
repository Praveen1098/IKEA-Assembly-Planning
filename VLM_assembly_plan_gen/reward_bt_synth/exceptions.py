"""Named exceptions for the reward_bt_synth pipeline.

Every exception carries a structured message identifying the specific offending
artefact (AST lineno, predicate name, part id, connection pair) so the failure
is directly actionable. There are no silent fallbacks anywhere in this package.
"""

from __future__ import annotations


class RewardBTSynthError(Exception):
    """Base class for all errors raised by this package."""


class ExtractionError(RewardBTSynthError):
    """Raised when the AST extractor cannot produce a sound STRIPS tuple.

    Attributes:
        skill_name: Which skill's reward spec was being analysed.
        phase: Which of the six extraction phases failed (1..6).
        lineno: Source line number of the offending AST node (None if N/A).
        predicate: Name of the offending predicate (None if N/A).
    """

    def __init__(
        self,
        message: str,
        *,
        skill_name: str,
        phase: int,
        lineno: int | None = None,
        predicate: str | None = None,
    ):
        detail = f"[skill={skill_name}][phase={phase}]"
        if lineno is not None:
            detail += f"[line={lineno}]"
        if predicate is not None:
            detail += f"[predicate={predicate}]"
        super().__init__(f"{detail} {message}")
        self.skill_name = skill_name
        self.phase = phase
        self.lineno = lineno
        self.predicate = predicate


class ExpansionFailure(RewardBTSynthError):
    """Raised when Cai AAAI 2021 BT Expansion terminates without reaching s0.

    Attributes:
        unreached: The condition set that remained unexpanded.
        iterations: How many outer-loop iterations ran before termination.
    """

    def __init__(
        self,
        message: str,
        *,
        unreached: frozenset,
        iterations: int,
    ):
        super().__init__(f"[iters={iterations}][unreached={sorted(str(x) for x in unreached)}] {message}")
        self.unreached = unreached
        self.iterations = iterations


class VerificationFailure(RewardBTSynthError):
    """Raised when the stochastic verifier detects a structural-property violation.

    Attributes:
        property_name: Which of P1..P5 failed.
        violations: The offending action / transition list.
    """

    def __init__(
        self,
        message: str,
        *,
        property_name: str,
        violations: list,
    ):
        super().__init__(f"[property={property_name}] {message}")
        self.property_name = property_name
        self.violations = violations


class StructuredPlanError(RewardBTSynthError):
    """Raised during Coverage-Conservation Structured Plan Extraction.

    Attributes:
        stage: 'enumerate' | 'assign' | 'repair' | 'verify'.
        missing_parts: Parts absent from the final plan (None if N/A).
        unassigned_connections: Connections that no repair could place (None if N/A).
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        missing_parts: frozenset[int] | None = None,
        unassigned_connections: frozenset[tuple[int, int]] | None = None,
    ):
        detail = f"[stage={stage}]"
        if missing_parts is not None:
            detail += f"[missing_parts={sorted(missing_parts)}]"
        if unassigned_connections is not None:
            detail += f"[unassigned_connections={sorted(unassigned_connections)}]"
        super().__init__(f"{detail} {message}")
        self.stage = stage
        self.missing_parts = missing_parts
        self.unassigned_connections = unassigned_connections


class IncompleteCoverageError(StructuredPlanError):
    """Specific subclass for coverage-invariant violations. Kept for clarity."""


class VocabularyError(RewardBTSynthError):
    """Raised when the predicate or initializer registry is misused.

    E.g. duplicate registration, resolution of an unregistered function call.
    """
