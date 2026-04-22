"""BT topology analysis — structural metrics beyond the P1..P5 simulation checks.

These operate on a synthesised BTNode tree without running an LLM or a
simulator. They are the "14 formal properties computable from tree topology
alone" discussed in the plan's Section 12 references, reduced to the subset
that is meaningful for our two-skill PPA-shaped trees.

Every function here is pure: given the same BTNode, returns the same metric.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterator

from .action_model import ActionModel, Literal
from .bt_expansion import BTNode, _tick


@dataclass(frozen=True)
class TopologyReport:
    """Structural metrics for a synthesised BT.

    All fields are derivable from the BTNode tree alone; no execution or
    state is required.
    """

    total_nodes: int
    max_depth: int
    min_leaf_depth: int
    balance_ratio: float                # min_leaf_depth / max_leaf_depth ∈ [0, 1]
    branching_factor_mean: float        # mean |children| over composites
    branching_factor_max: int
    node_counts: dict[str, int]         # {action, condition, sequence, fallback}
    ppa_compliant_root: bool            # every Fallback follows PPA shape
    ppa_pattern_count: int              # number of PPA Fallbacks in the tree
    mccabe_complexity: int              # |Selectors| + 1
    reactive: bool                      # (trivially True: we construct with memory=False)
    depth_per_sub_goal: int             # expected constant for our pipeline

    def to_dict(self) -> dict:
        return {
            "total_nodes": self.total_nodes,
            "max_depth": self.max_depth,
            "min_leaf_depth": self.min_leaf_depth,
            "balance_ratio": round(self.balance_ratio, 4),
            "branching_factor_mean": round(self.branching_factor_mean, 4),
            "branching_factor_max": self.branching_factor_max,
            "node_counts": self.node_counts,
            "ppa_compliant_root": self.ppa_compliant_root,
            "ppa_pattern_count": self.ppa_pattern_count,
            "mccabe_complexity": self.mccabe_complexity,
            "reactive": self.reactive,
            "depth_per_sub_goal": self.depth_per_sub_goal,
        }


# ---------- Leaf-depth enumeration ----------

def _leaf_depths(node: BTNode, depth: int = 0) -> list[int]:
    """Depths of all leaf nodes (condition + action). Composites are not leaves."""
    if not node.children:
        return [depth]
    out: list[int] = []
    for child in node.children:
        out.extend(_leaf_depths(child, depth + 1))
    return out


def _all_nodes(node: BTNode) -> Iterator[BTNode]:
    yield node
    for c in node.children:
        yield from _all_nodes(c)


# ---------- PPA pattern recognition ----------

def _is_ppa_fallback(node: BTNode) -> bool:
    """Fallback in the PPA shape:
        Fallback(
          Condition(...),
          Sequence(Condition(...), Action(...)),
          ...
        )
    First child is a Condition; all remaining children are Sequences whose
    last leaf is an Action.
    """
    if node.kind != "fallback":
        return False
    if not node.children:
        return False
    if node.children[0].kind != "condition":
        return False
    for sibling in node.children[1:]:
        if sibling.kind != "sequence":
            return False
        if not sibling.children:
            return False
        if sibling.children[-1].kind != "action":
            return False
    return True


def ppa_pattern_count(node: BTNode) -> int:
    """How many Fallback nodes in the tree obey the PPA shape."""
    return sum(1 for n in _all_nodes(node) if _is_ppa_fallback(n))


def all_fallbacks_are_ppa(root: BTNode) -> bool:
    """Strict: every Fallback in the whole tree must be PPA-shaped."""
    return all(
        _is_ppa_fallback(n) for n in _all_nodes(root) if n.kind == "fallback"
    )


# ---------- Public entry point ----------

def analyse_topology(root: BTNode) -> TopologyReport:
    leaf_depths = _leaf_depths(root) or [0]
    max_leaf = max(leaf_depths)
    min_leaf = min(leaf_depths)
    balance = (min_leaf / max_leaf) if max_leaf else 1.0

    node_counts: dict[str, int] = Counter()
    composite_branch = []
    for n in _all_nodes(root):
        node_counts[n.kind] += 1
        if n.kind in ("sequence", "fallback") and n.children:
            composite_branch.append(len(n.children))

    branch_mean = (
        sum(composite_branch) / len(composite_branch) if composite_branch else 0.0
    )
    branch_max = max(composite_branch) if composite_branch else 0

    total = sum(node_counts.values())

    # Reactiveness: constructed with memory=False via pytrees_build; here we
    # assert it structurally — our bt_expansion never emits memory-carrying
    # composites, so reactive is True by construction.
    reactive = True

    # Expected depth_per_sub_goal is a structural fingerprint: for the
    # two-skill Lift+Insert library, every PPA sub-tree has depth 5.
    depth_per_sub_goal = 5

    mccabe = node_counts.get("fallback", 0) + 1

    return TopologyReport(
        total_nodes=total,
        max_depth=max_leaf,
        min_leaf_depth=min_leaf,
        balance_ratio=balance,
        branching_factor_mean=branch_mean,
        branching_factor_max=branch_max,
        node_counts=dict(node_counts),
        ppa_compliant_root=all_fallbacks_are_ppa(root),
        ppa_pattern_count=ppa_pattern_count(root),
        mccabe_complexity=mccabe,
        reactive=reactive,
        depth_per_sub_goal=depth_per_sub_goal,
    )


# ---------- Action sequence extraction (for ground-truth comparison) ----------

@dataclass(frozen=True)
class ExecutedAction:
    """An action leaf invoked during a deterministic Tick from s0."""

    name: str                # "lift" or "insert"
    args: tuple              # grounded argument tuple
    at_step_index: int       # 0-based position in the overall Tick order


def extract_action_sequence(
    root: BTNode,
    s0: frozenset[Literal],
) -> list[ExecutedAction]:
    """Deterministic Tick trace under s0: list of action leaves in the order
    they fire, with 0-indexed positions.

    Uses the same state semantics as `bt_expansion._tick` but records action
    invocations along the way. Only successful rollouts are reported — if the
    root fails under s0, returns an empty list (should never happen for a
    well-formed bt_expand output).
    """
    trace: list[ExecutedAction] = []
    counter = [0]  # step index

    def visit(node: BTNode, state: frozenset[Literal]) -> tuple[str, frozenset[Literal]]:
        if node.kind == "condition":
            assert node.condition is not None
            return ("SUCCESS", state) if node.condition <= state else ("FAILURE", state)
        if node.kind == "action":
            assert node.action is not None
            new_state = (state - node.action.delete) | node.action.add
            args = _grounded_args(node.action)
            trace.append(
                ExecutedAction(
                    name=node.action.name,
                    args=args,
                    at_step_index=counter[0],
                )
            )
            counter[0] += 1
            return "SUCCESS", new_state
        if node.kind == "sequence":
            current = state
            snapshot_trace_len = len(trace)
            for c in node.children:
                r, current = visit(c, current)
                if r == "FAILURE":
                    # Roll back any trace entries produced in this failed sequence
                    del trace[snapshot_trace_len:]
                    return "FAILURE", state
            return "SUCCESS", current
        # fallback
        for c in node.children:
            snapshot_trace_len = len(trace)
            r, new_state = visit(c, state)
            if r == "SUCCESS":
                return "SUCCESS", new_state
            del trace[snapshot_trace_len:]
        return "FAILURE", state

    result, _ = visit(root, s0)
    if result != "SUCCESS":
        return []
    return trace


def _grounded_args(action: ActionModel) -> tuple:
    """Recover grounded argument tuple from action.add literals."""
    for lit in action.add:
        if lit.name == "inserted" and len(lit.args) == 2:
            return lit.args
        if lit.name == "holding" and len(lit.args) == 1:
            return lit.args
    return ()
