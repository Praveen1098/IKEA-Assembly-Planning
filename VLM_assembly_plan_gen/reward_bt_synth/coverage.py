"""Coverage-invariant checks for the Stage-2-v2 structured plan pipeline.

Pure functions, no side effects, no VLM calls. Each either returns cleanly or
raises a named error that carries the offending parts/connections.

Called from stage2_v2_plan at two points:
  1. After VLM Call 1 (enumerate connections) — check every part participates
     in ≥ 1 connection AND the induced graph is connected.
  2. After VLM Call 2 (assign connections to steps) — check every connection
     got exactly one step assignment and step numbering is contiguous.
"""

from __future__ import annotations

from collections import deque

from .exceptions import IncompleteCoverageError, StructuredPlanError


def _canonical(edge: tuple[int, int]) -> tuple[int, int]:
    """Order-independent edge representation: (min, max)."""
    a, b = edge
    return (a, b) if a <= b else (b, a)


# ---------- After VLM Call 1 ----------

def check_all_parts_participate(
    parts: frozenset[int],
    connections: list[tuple[int, int]],
) -> None:
    """Raise IncompleteCoverageError if any part appears in no connection.

    This is the primary guard against Stage 2's silent part-omission failure.
    """
    present = {p for (a, b) in connections for p in (a, b)}
    missing = parts - present
    if missing:
        raise IncompleteCoverageError(
            f"{len(missing)} part(s) do not participate in any enumerated "
            f"connection; the assembly graph is incomplete",
            stage="enumerate",
            missing_parts=frozenset(missing),
        )


def check_connection_graph_connected(
    parts: frozenset[int],
    connections: list[tuple[int, int]],
) -> None:
    """Raise StructuredPlanError if the graph on `parts` with edges `connections`
    is not a single connected component.

    Stronger than participation alone: two disconnected sub-assemblies (e.g.
    a chair with two disjoint seat / backrest clusters) would pass
    participation but fail this connectedness check. Physical assembly
    requires a connected graph, so we treat disconnectedness as a fatal
    enumeration error.
    """
    if not parts:
        return
    if len(parts) == 1:
        # Trivially connected
        return

    adjacency: dict[int, set[int]] = {p: set() for p in parts}
    for a, b in connections:
        if a in adjacency and b in adjacency and a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    start = next(iter(parts))
    visited: set[int] = {start}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for nb in adjacency[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    disconnected = parts - visited
    if disconnected:
        raise StructuredPlanError(
            f"assembly graph over {len(parts)} parts is disconnected; "
            f"{len(disconnected)} part(s) not reachable from the rest",
            stage="enumerate",
            missing_parts=frozenset(disconnected),
        )


# ---------- After VLM Call 2 ----------

def check_assignment_coverage(
    connections: list[tuple[int, int]],
    step_assignments: dict[tuple[int, int], int],
) -> None:
    """Every enumerated connection must have exactly one step assignment.

    Canonical (min, max) ordering is used so (2, 5) and (5, 2) are treated
    as the same edge. Raises StructuredPlanError listing the unassigned or
    extraneous connections.
    """
    enumerated = {_canonical(c) for c in connections}
    assigned = {_canonical(c) for c in step_assignments.keys()}

    missing = enumerated - assigned
    if missing:
        raise StructuredPlanError(
            f"{len(missing)} connection(s) were enumerated but not assigned "
            "to any step",
            stage="assign",
            unassigned_connections=frozenset(missing),
        )

    extras = assigned - enumerated
    if extras:
        raise StructuredPlanError(
            f"{len(extras)} connection(s) were assigned but not in the "
            "enumerated set — VLM hallucinated connections",
            stage="assign",
            unassigned_connections=frozenset(extras),
        )


def check_temporal_validity(
    step_assignments: dict[tuple[int, int], int],
) -> None:
    """Step numbers must be positive integers.

    VLM Call 2 is told ``step ∈ 1..N`` where N = number of manual pages.
    Some pages may have no connection-forming content (parts laid out,
    hardware inventory, final-assembly illustration), so the set of
    step values we receive is a subset of ``{1..N}``; it need not start
    at 1 or be contiguous.

    We reject only clearly-corrupt outputs (0 or negative step ids).
    Gaps are legitimate — they indicate that some manual pages carry no
    connection work, which is common.
    """
    if not step_assignments:
        return
    steps = sorted(set(step_assignments.values()))
    if steps[0] < 1:
        raise StructuredPlanError(
            f"step numbers must be >= 1; smallest observed is {steps[0]}",
            stage="assign",
        )
