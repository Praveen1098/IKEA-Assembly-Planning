"""main_data.json entry → list[Literal] connection goals.

Main-data connections are undirected pairs. For BT synthesis we must orient
each edge so the Insert action's peg (the part that was lifted and moved) is
distinct across edges; otherwise a single peg is "reused" multiple times,
which breaks the STRIPS model (Insert deletes holding(peg), so successive
Inserts need a re-Lift of a part that is no longer on_surface — physically
the part is now inside an assembled subassembly).

Orientation rule (simple, deterministic, principled):
  peg    = the endpoint with the LOWER vertex degree in the connection graph
  socket = the other endpoint
  (ties broken by the lower part id)

Rationale: a leaf (degree 1) is a part that appears in only one connection —
the natural "peg" to lift and insert. A hub (high degree) is the natural
"socket" that accumulates leaves. This orientation matches how IKEA manuals
typically present assembly: you attach parts to a frame, not the frame to
parts.
"""

from __future__ import annotations

from collections import Counter

from .action_model import Literal
from .exceptions import IncompleteCoverageError


def connection_goals(entry: dict) -> list[Literal]:
    """Convert a main_data.json entry's `connection_relation` into oriented
    `inserted(peg, socket)` literals, one per undirected connection pair.

    Args:
        entry: one element of main_data.json (keys: name, parts_ct,
            connection_relation, ...).

    Returns:
        A list of Literals, same length as `entry["connection_relation"]`.

    Raises:
        IncompleteCoverageError if the connection set doesn't cover every
        part (0..parts_ct-1). This is the same guard C²SPE applies to the
        VLM-extracted plan.
    """
    connections = entry.get("connection_relation", [])
    parts_ct = int(entry.get("parts_ct", 0))
    parts = frozenset(range(parts_ct))

    # Coverage check — refuse to produce goals over an incomplete graph
    present = {p for pair in connections for p in pair}
    missing = parts - present
    if missing:
        raise IncompleteCoverageError(
            f"main_data entry {entry.get('name')!r}: "
            f"connection_relation misses parts {sorted(missing)}",
            stage="enumerate",
            missing_parts=frozenset(missing),
        )

    # Degree-based orientation
    degree: Counter[int] = Counter()
    for a, b in connections:
        degree[int(a)] += 1
        degree[int(b)] += 1

    goals: list[Literal] = []
    for pair in connections:
        a, b = int(pair[0]), int(pair[1])
        if degree[a] < degree[b]:
            peg, socket = a, b
        elif degree[a] > degree[b]:
            peg, socket = b, a
        else:
            # Tie-break: smaller id is the peg
            peg, socket = (a, b) if a < b else (b, a)
        goals.append(Literal("inserted", (peg, socket)))
    return goals


def initial_state_for(entry: dict) -> frozenset[Literal]:
    """Build the s0 frozenset for a given furniture entry.

    Every part starts `on_surface` and `accessible`. Gripper is empty.
    Mirrors what `reset_to_init_lift` and `reset_to_init_insert` establish
    per-skill, generalised across the whole furniture.
    """
    parts_ct = int(entry.get("parts_ct", 0))
    lits: set[Literal] = {Literal("gripper_empty", ())}
    for p in range(parts_ct):
        lits.add(Literal("on_surface", (p,)))
        lits.add(Literal("accessible", (p,)))
    return frozenset(lits)
