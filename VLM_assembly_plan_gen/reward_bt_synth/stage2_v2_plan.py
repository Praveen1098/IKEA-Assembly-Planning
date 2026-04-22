"""Coverage-Conservation Structured Plan Extraction (C²SPE).

Secondary contribution. Replaces the free-form prose VLM call in
`inference/stage2_planning.create_plan` with a three-call decomposition whose
intermediate outputs are constrained to known finite sets. Coverage invariants
(every part participates, graph is connected, every connection is assigned
exactly once, steps are contiguous 1..N) are checked algorithmically between
calls. The silent part-omission failure mode of the existing Stage 2 is
structurally eliminated by this design.

The module accepts an injectable `vlm_client` that satisfies the
`VLMClient` protocol — in production this is a thin wrapper over
`VLM_assembly_plan_gen/llm/utils.py::invoke_multimodal`; in tests it is a
deterministic mock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from .coverage import (
    check_all_parts_participate,
    check_assignment_coverage,
    check_connection_graph_connected,
    check_temporal_validity,
)
from .exceptions import IncompleteCoverageError, StructuredPlanError

# ---------- Public types ----------


@runtime_checkable
class VLMClient(Protocol):
    """Minimal duck-typed contract for a multimodal LLM client.

    The existing `llm/utils.py::invoke_multimodal` satisfies this shape; so
    does any pytest mock that implements `invoke_multimodal(prompt, images)`.
    """

    def invoke_multimodal(self, prompt: str, base64_images: list[str]) -> str: ...


@dataclass(frozen=True)
class StructuredPlan:
    """Fully-covered assembly plan produced by C²SPE.

    Invariants guaranteed by construction:
      - every part in `parts` appears in ≥ 1 element of `connections`
      - the graph (parts, connections) is a single connected component
      - every element of `connections` is a key of `step_assignments`
      - values of `step_assignments` form the contiguous set 1..num_steps
    """

    parts: frozenset[int]
    connections: tuple[tuple[int, int], ...]
    step_assignments: dict[tuple[int, int], int]
    num_steps: int

    def connections_at_step(self, step: int) -> list[tuple[int, int]]:
        return sorted(c for c, s in self.step_assignments.items() if s == step)

    def to_dict(self) -> dict:
        return {
            "parts": sorted(self.parts),
            "connections": [list(c) for c in self.connections],
            "step_assignments": {
                f"{c[0]},{c[1]}": s for c, s in sorted(self.step_assignments.items())
            },
            "num_steps": self.num_steps,
        }


# ---------- Prompt loading ----------

_PROMPTS_DIR = Path(__file__).parent / "prompts_v2"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


# ---------- Shared JSON parsing ----------

def _strip_code_fence(raw: str) -> str:
    """Remove ``` code fences if present. Tolerant of language tags."""
    txt = raw.strip()
    if txt.startswith("```"):
        # Drop any line that starts with ```
        lines = [line for line in txt.splitlines() if not line.startswith("```")]
        txt = "\n".join(lines).strip()
    return txt


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


# ---------- VLM Call 1 — enumerate connections ----------

def _parse_connections_json(raw: str) -> list[tuple[int, int]]:
    txt = _strip_code_fence(raw)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError as exc:
        raise StructuredPlanError(
            f"Call 1 output is not valid JSON: {exc.msg}",
            stage="enumerate",
        ) from exc
    if not isinstance(data, list):
        raise StructuredPlanError(
            f"Call 1 output must be a JSON array; got {type(data).__name__}",
            stage="enumerate",
        )
    result: list[tuple[int, int]] = []
    for item in data:
        if not (isinstance(item, list) and len(item) == 2):
            raise StructuredPlanError(
                f"connection entry {item!r} is not a 2-element array",
                stage="enumerate",
            )
        a, b = item
        if not (isinstance(a, int) and isinstance(b, int)):
            raise StructuredPlanError(
                f"connection {item!r} contains non-integer part ids",
                stage="enumerate",
            )
        result.append((int(a), int(b)))
    return result


def _enumerate_connections(
    vlm: VLMClient,
    parts: frozenset[int],
    final_assembly_image_b64: str | list[str],
    retry_hint: str,
) -> list[tuple[int, int]]:
    template = _load_prompt("enumerate_connections.txt")
    parts_str = ", ".join(str(p) for p in sorted(parts))
    prompt = template.format(
        parts_inventory=parts_str,
        retry_hint=retry_hint,
    )
    # Permit a list of context images (e.g. labeled scene + all manual pages).
    # Backwards-compatible with the single-image call form used in tests.
    images = (
        [final_assembly_image_b64]
        if isinstance(final_assembly_image_b64, str)
        else list(final_assembly_image_b64)
    )
    raw = vlm.invoke_multimodal(prompt, images)
    return _parse_connections_json(raw)


# ---------- VLM Call 2 — assign connections to steps ----------

def _parse_assignments_json(raw: str) -> dict[tuple[int, int], int]:
    """Accepts either [{{"connection": [a,b], "step": n}}, ...]
    or {{\"a,b\": n, ...}}. Canonicalises edge order."""
    txt = _strip_code_fence(raw)
    try:
        data = json.loads(txt)
    except json.JSONDecodeError as exc:
        raise StructuredPlanError(
            f"Call 2 / 3 output is not valid JSON: {exc.msg}",
            stage="assign",
        ) from exc

    result: dict[tuple[int, int], int] = {}
    if isinstance(data, list):
        for item in data:
            if not (isinstance(item, dict) and "connection" in item and "step" in item):
                raise StructuredPlanError(
                    f"assignment entry {item!r} missing 'connection' or 'step' key",
                    stage="assign",
                )
            conn = item["connection"]
            step = item["step"]
            if not (isinstance(conn, list) and len(conn) == 2
                    and all(isinstance(x, int) for x in conn)):
                raise StructuredPlanError(
                    f"assignment connection {conn!r} malformed",
                    stage="assign",
                )
            if not isinstance(step, int):
                raise StructuredPlanError(
                    f"assignment step {step!r} is not an integer",
                    stage="assign",
                )
            result[_canonical_edge(conn[0], conn[1])] = step
    elif isinstance(data, dict):
        for key, val in data.items():
            parts_of_key = [int(x.strip()) for x in str(key).strip("[]").split(",")]
            if len(parts_of_key) != 2:
                raise StructuredPlanError(
                    f"assignment key {key!r} must decode to two ints",
                    stage="assign",
                )
            if not isinstance(val, int):
                raise StructuredPlanError(
                    f"assignment value for {key!r} is not an integer",
                    stage="assign",
                )
            result[_canonical_edge(parts_of_key[0], parts_of_key[1])] = val
    else:
        raise StructuredPlanError(
            f"Call 2 / 3 output must be list-of-dicts or dict; "
            f"got {type(data).__name__}",
            stage="assign",
        )
    return result


def _assign_connections_to_steps(
    vlm: VLMClient,
    connections: list[tuple[int, int]],
    manual_page_images_b64: list[str],
) -> dict[tuple[int, int], int]:
    template = _load_prompt("assign_connections_to_steps.txt")
    connections_str = json.dumps([list(c) for c in connections])
    prompt = template.format(
        connections=connections_str,
        num_pages=len(manual_page_images_b64),
    )
    raw = vlm.invoke_multimodal(prompt, manual_page_images_b64)
    return _parse_assignments_json(raw)


# ---------- VLM Call 3 — targeted repair ----------

def _repair_assignment(
    vlm: VLMClient,
    missing_connections: frozenset[tuple[int, int]],
    manual_page_images_b64: list[str],
) -> dict[tuple[int, int], int]:
    template = _load_prompt("repair_connection_assignment.txt")
    missing_str = json.dumps([list(c) for c in sorted(missing_connections)])
    prompt = template.format(
        missing_connections=missing_str,
        num_pages=len(manual_page_images_b64),
    )
    raw = vlm.invoke_multimodal(prompt, manual_page_images_b64)
    return _parse_assignments_json(raw)


# ---------- Public entry point ----------

def generate_plan_v2(
    vlm: VLMClient,
    parts: frozenset[int],
    final_assembly_image_b64: str | list[str],
    manual_page_images_b64: list[str],
    *,
    max_enum_retries: int = 1,
) -> StructuredPlan:
    """Run the full C²SPE pipeline and return a coverage-guaranteed plan.

    Raises StructuredPlanError (or subclass IncompleteCoverageError) with the
    specific missing parts / unassigned connections on any irreparable deficit.
    """
    # ---- Phase 1: enumerate (with optional retry on disconnectedness) ----
    retry_hint = ""
    connections: list[tuple[int, int]] | None = None
    last_err: StructuredPlanError | None = None
    for _attempt in range(max_enum_retries + 1):
        connections = _enumerate_connections(
            vlm, parts, final_assembly_image_b64, retry_hint
        )
        try:
            check_all_parts_participate(parts, connections)
            check_connection_graph_connected(parts, connections)
            break  # enumeration is good
        except StructuredPlanError as exc:
            last_err = exc
            missing = sorted(exc.missing_parts or [])
            retry_hint = (
                f"\n\nNOTE: a previous attempt isolated these parts from the "
                f"assembly graph: {missing}. Ensure each of them appears in at "
                "least one connection pair."
            )
    else:
        # for-else: loop finished without break
        assert last_err is not None
        raise StructuredPlanError(
            f"after {max_enum_retries + 1} attempt(s), enumeration still "
            f"produces an incomplete assembly graph: {last_err}",
            stage="enumerate",
            missing_parts=last_err.missing_parts,
        ) from last_err

    assert connections is not None

    # ---- Phase 2: assign connections to steps ----
    assignments = _assign_connections_to_steps(vlm, connections, manual_page_images_b64)

    # ---- Phase 3: coverage + targeted repair ----
    # Recompute missing/extras independently of exception attributes so we
    # distinguish the two failure modes reliably. Missing → targeted repair;
    # extras (hallucinated connections) → raise immediately (no heuristic fix).
    try:
        check_assignment_coverage(connections, assignments)
    except StructuredPlanError:
        enumerated_set = {_canonical_edge(a, b) for (a, b) in connections}
        assigned_set = {_canonical_edge(k[0], k[1]) for k in assignments.keys()}
        missing = enumerated_set - assigned_set
        extras = assigned_set - enumerated_set
        if extras:
            # VLM hallucinated connections — fail loud, no repair.
            raise
        if not missing:
            # Defensive: shouldn't reach here if coverage check raised.
            raise
        repaired = _repair_assignment(vlm, frozenset(missing), manual_page_images_b64)
        for k, v in repaired.items():
            if k in missing:
                assignments[k] = v
        # Second-pass check — no silent tolerance of remaining deficits.
        check_assignment_coverage(connections, assignments)

    check_temporal_validity(assignments)

    num_steps = max(assignments.values()) if assignments else 0
    return StructuredPlan(
        parts=parts,
        connections=tuple(sorted(_canonical_edge(a, b) for a, b in connections)),
        step_assignments=dict(assignments),
        num_steps=num_steps,
    )


# ---------- Re-export for convenience ----------
__all__ = [
    "StructuredPlan",
    "VLMClient",
    "generate_plan_v2",
]

# Silence an unused-import warning in strict linters (IncompleteCoverageError
# is documented in the module docstring as a possible raise but is caught by
# the StructuredPlanError base class in generate_plan_v2).
_ = IncompleteCoverageError
