"""Compare a synthesised BT's executed plan vs ground-truth IKEA steps.

Ground truth in `main_data.json`:
  - connection_relation: list of [a, b] pairs (undirected edges; order-agnostic)
  - steps: per-step dicts with a `connections` field — each step forms one or
    more connections simultaneously
  - assembly_tree: nested list representing hierarchical subassemblies

Our BT: after Tick from s0, fires a deterministic sequence of Lift and
Insert actions. Each Insert forms exactly one connection.

The meaningful comparisons are:
  1. Connection-set match: does our set of connections equal the ground-truth
     edge set (mod direction)?
  2. Connection count parity: do we synthesise exactly `len(connection_relation)`
     Insert actions?
  3. Precedence consistency: for every pair (c_i, c_j) where ground truth
     places c_i before c_j (across steps), does our Tick order also place
     c_i before c_j?
  4. Per-ground-truth-step alignment: which of our actions land inside each
     ground-truth step's connection set, and in what order.

These are the well-defined tests one can run without paraphrasing ground truth
into BT semantics — neither side has to be serialised into the other's format.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analysis import ExecutedAction


@dataclass(frozen=True)
class ComparisonReport:
    """Head-to-head comparison of BT plan vs ground-truth steps.

    Attributes:
        bt_connection_count: number of Insert actions our BT fires
        gt_connection_count: number of edges in main_data.connection_relation
        connection_set_match: our {inserted pairs} == ground-truth {edges} ?
        extra_bt_connections: pairs we form but ground truth does not
        missing_bt_connections: pairs ground truth has but we don't
        precedence_violations: (c_before, c_after) pairs where ground truth
            orders c_before < c_after but our Tick fires c_after first
        gt_step_coverage: list of dicts, one per ground-truth step, describing
            which of our Inserts land in it and at what Tick index
        bt_action_count: total Lift + Insert count
        lift_count: Lift action count in our Tick
        insert_count: Insert action count in our Tick
    """

    bt_connection_count: int
    gt_connection_count: int
    connection_set_match: bool
    extra_bt_connections: list[tuple[int, int]]
    missing_bt_connections: list[tuple[int, int]]
    precedence_violations: list[tuple[tuple[int, int], tuple[int, int]]]
    gt_step_coverage: list[dict]
    bt_action_count: int
    lift_count: int
    insert_count: int

    def to_dict(self) -> dict:
        return {
            "bt_connection_count": self.bt_connection_count,
            "gt_connection_count": self.gt_connection_count,
            "connection_set_match": self.connection_set_match,
            "extra_bt_connections": [list(c) for c in self.extra_bt_connections],
            "missing_bt_connections": [list(c) for c in self.missing_bt_connections],
            "precedence_violations": [
                {"before": list(b), "after": list(a)}
                for b, a in self.precedence_violations
            ],
            "gt_step_coverage": self.gt_step_coverage,
            "bt_action_count": self.bt_action_count,
            "lift_count": self.lift_count,
            "insert_count": self.insert_count,
        }


def _canonical(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _parse_part_refs(ref) -> list[int]:
    """Parse a part reference from main_data.json.

    Connection entries sometimes use comma-separated strings like ``"0,1,2"``
    to mean "subassembly containing parts 0, 1, 2". We unpack those into the
    individual part ids. Single-part references like ``3`` or ``"3"`` are
    returned as a one-element list.
    """
    if isinstance(ref, (list, tuple)):
        out: list[int] = []
        for x in ref:
            out.extend(_parse_part_refs(x))
        return out
    s = str(ref).strip()
    if "," in s:
        return [int(tok.strip()) for tok in s.split(",") if tok.strip()]
    return [int(s)]


def _expand_pair(pair) -> list[tuple[int, int]]:
    """Expand a raw connection pair (possibly subassembly-style) into
    canonical (int, int) edges.

    For ``["0,1,2", "3"]`` we emit ``[(0,3), (1,3), (2,3)]`` — every part of
    the subassembly connects to the single other part.
    """
    lhs = _parse_part_refs(pair[0])
    rhs = _parse_part_refs(pair[1])
    return [_canonical(a, b) for a in lhs for b in rhs if a != b]


def _normalise_connections(raw: list) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for pair in raw:
        out.extend(_expand_pair(pair))
    return out


def compare_with_ground_truth(
    bt_actions: list[ExecutedAction],
    gt_entry: dict,
) -> ComparisonReport:
    """Build a full comparison report for one furniture item.

    Args:
        bt_actions: output of `analysis.extract_action_sequence` — deterministic
            Tick trace.
        gt_entry: one entry from main_data.json (must contain
            `connection_relation` and ideally `steps`).
    """
    # BT side
    bt_inserts = [a for a in bt_actions if a.name == "insert"]
    bt_lifts = [a for a in bt_actions if a.name == "lift"]

    bt_connection_seq: list[tuple[int, int]] = [
        _canonical(a.args[0], a.args[1]) for a in bt_inserts
    ]
    bt_connection_set = set(bt_connection_seq)

    # Ground-truth side
    gt_edges_list = _normalise_connections(gt_entry.get("connection_relation", []))
    gt_connection_set = set(gt_edges_list)

    # (1) Set equality + deltas
    extras = sorted(bt_connection_set - gt_connection_set)
    missings = sorted(gt_connection_set - bt_connection_set)
    set_match = len(extras) == 0 and len(missings) == 0

    # (2) Precedence check — derive a partial order from ground-truth steps
    # Ground truth orders: connections in step_i come before connections in step_j
    # whenever i < j. Within the same step, no precedence between them.
    gt_steps = gt_entry.get("steps", [])
    gt_edge_step: dict[tuple[int, int], int] = {}
    for step_idx, step in enumerate(gt_steps):
        for pair in step.get("connections", []):
            for edge in _expand_pair(pair):
                gt_edge_step[edge] = step_idx

    # For edges that appear in connection_relation but not in any step (rare),
    # assign them the step index equal to len(gt_steps) — they come "after" everything
    for e in gt_edges_list:
        gt_edge_step.setdefault(e, len(gt_steps))

    # Our Tick order index per connection
    bt_order: dict[tuple[int, int], int] = {
        _canonical(a.args[0], a.args[1]): a.at_step_index for a in bt_inserts
    }

    # Precedence violations: any (c_i, c_j) with gt_edge_step[c_i] <
    # gt_edge_step[c_j] but bt_order[c_i] > bt_order[c_j]
    precedence_violations: list[tuple[tuple[int, int], tuple[int, int]]] = []
    common = [c for c in bt_order if c in gt_edge_step]
    for i in range(len(common)):
        for j in range(len(common)):
            if i == j:
                continue
            ci, cj = common[i], common[j]
            if gt_edge_step[ci] < gt_edge_step[cj] and bt_order[ci] > bt_order[cj]:
                precedence_violations.append((ci, cj))

    # (3) Per-ground-truth-step coverage
    gt_step_coverage: list[dict] = []
    for step_idx, step in enumerate(gt_steps):
        step_edges: list[tuple[int, int]] = []
        for pair in step.get("connections", []):
            step_edges.extend(_expand_pair(pair))
        our_inserts_in_step = [
            {
                "connection": list(e),
                "bt_step_index": bt_order[e]
                if e in bt_order
                else None,
            }
            for e in step_edges
        ]
        gt_step_coverage.append({
            "gt_step_index": step_idx,
            "page_id": step.get("page_id"),
            "gt_connections": [list(e) for e in step_edges],
            "our_inserts": our_inserts_in_step,
        })

    return ComparisonReport(
        bt_connection_count=len(bt_inserts),
        gt_connection_count=len(gt_edges_list),
        connection_set_match=set_match,
        extra_bt_connections=extras,
        missing_bt_connections=missings,
        precedence_violations=precedence_violations,
        gt_step_coverage=gt_step_coverage,
        bt_action_count=len(bt_actions),
        lift_count=len(bt_lifts),
        insert_count=len(bt_inserts),
    )
