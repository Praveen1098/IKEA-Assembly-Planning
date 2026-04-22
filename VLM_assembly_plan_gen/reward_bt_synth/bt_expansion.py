"""BT Expansion (Cai et al., AAAI 2021) — direct, faithful implementation.

Reference: Algorithm 1 (Expand) and Algorithm 2 (main loop) from
"Behavior Tree Building for Flexible Cognitive Systems" (AAAI 2021).

Correctness: the published theorems give soundness (if the algorithm returns a
BT, that BT is Finite-Time Successful with ROA containing s0) and completeness
(if the planning problem is solvable, the algorithm returns a BT). Both hold
when the input action models are STRIPS-consistent ⟨pre, add, del⟩ tuples,
which is exactly what `ast_extractor.extract_action_model` produces.

Lifted grounding: ActionModels in our pipeline are lifted (formals ?peg,
?socket). Rather than pre-enumerate all O(|parts|^arity) grounded actions, we
instantiate on demand: for each literal in the condition being expanded, we
attempt to unify it with each ActionModel's add-set literals, deriving a
formal-parameter binding. Only bindings that cover ALL of the model's formals
produce a valid grounded instance. For Lift(?p) and Insert(?peg, ?socket) this
is always the case (inserted(a, b) covers both, holding(a) covers ?p).

No heuristics, no cost optimisation, no pruning beyond the published
algorithm's Prune step (which we omit because the two-skill domain does not
accumulate dead subtrees in practice; see `test_bt_expansion.py` for size
bounds).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .action_model import ActionModel, Literal, LiteralArg
from .exceptions import ExpansionFailure


# ---------- BT node representation ----------

_Kind = str  # "condition" | "action" | "sequence" | "fallback"
_ALL_KINDS = {"condition", "action", "sequence", "fallback"}


@dataclass(frozen=True)
class BTNode:
    """Lightweight BT node.

    Exactly one of `condition` or `action` is non-None (for leaf nodes); both
    are None for composite nodes, whose children live in `children`.
    """

    kind: _Kind
    condition: frozenset[Literal] | None = None
    action: ActionModel | None = None
    children: tuple["BTNode", ...] = ()

    def __post_init__(self):
        if self.kind not in _ALL_KINDS:
            raise ValueError(f"BTNode.kind={self.kind!r} invalid")
        if self.kind == "condition" and self.condition is None:
            raise ValueError("condition node must carry `condition` frozenset")
        if self.kind == "action" and self.action is None:
            raise ValueError("action node must carry `action` ActionModel")


# ---------- Literal unification (used for lifted grounding) ----------

def _unify_literals(
    template: Literal, concrete: Literal
) -> dict[str, LiteralArg] | None:
    """Return a formal → concrete binding if template matches concrete, else None."""
    if template.name != concrete.name:
        return None
    if template.negated != concrete.negated:
        return None
    if len(template.args) != len(concrete.args):
        return None
    binding: dict[str, LiteralArg] = {}
    for t_arg, c_arg in zip(template.args, concrete.args):
        if isinstance(t_arg, str) and t_arg.startswith("?"):
            if t_arg in binding:
                if binding[t_arg] != c_arg:
                    return None
            else:
                binding[t_arg] = c_arg
        else:
            if t_arg != c_arg:
                return None
    return binding


def _ground_model_for_literal(
    model: ActionModel, target: Literal
) -> ActionModel | None:
    """Attempt to ground `model` so `target` ∈ grounded.add.

    Returns the grounded ActionModel if a binding covering ALL formals is
    found; otherwise None (insufficient information to fully ground).
    """
    for add_lit in model.add:
        binding = _unify_literals(add_lit, target)
        if binding is None:
            continue
        if all(f in binding for f in model.params):
            return model.ground_all(binding)
    return None


# ---------- Tick simulation (with forward-propagating state) ----------

def _tick(
    node: BTNode, state: frozenset[Literal]
) -> tuple[str, frozenset[Literal]]:
    """Textbook BT Tick over STRIPS states.

    Returns (SUCCESS | FAILURE, new_state). Sequence propagates state left to
    right; Fallback tries children left to right from the incoming state;
    Action leaves commit their (add, delete) effects.
    """
    if node.kind == "condition":
        assert node.condition is not None
        return ("SUCCESS", state) if node.condition <= state else ("FAILURE", state)

    if node.kind == "action":
        assert node.action is not None
        new_state = (state - node.action.delete) | node.action.add
        return "SUCCESS", new_state

    if node.kind == "sequence":
        current = state
        for child in node.children:
            result, current = _tick(child, current)
            if result == "FAILURE":
                return "FAILURE", state  # overall sequence failed; revert
        return "SUCCESS", current

    # fallback
    for child in node.children:
        result, new_state = _tick(child, state)
        if result == "SUCCESS":
            return "SUCCESS", new_state
    return "FAILURE", state


# ---------- Condition traversal ----------

def _next_unexpanded_condition(
    root: BTNode, s0: frozenset[Literal], expanded: set[frozenset[Literal]]
) -> frozenset[Literal] | None:
    """First unexpanded Condition visited in Tick order that returns FAILURE."""
    found: list[frozenset[Literal] | None] = [None]

    def visit(node: BTNode, state: frozenset[Literal]) -> tuple[str, frozenset[Literal]]:
        if found[0] is not None:
            return "SUCCESS", state  # short-circuit
        if node.kind == "condition":
            assert node.condition is not None
            if node.condition <= state:
                return "SUCCESS", state
            if node.condition not in expanded:
                found[0] = node.condition
            return "FAILURE", state
        if node.kind == "action":
            assert node.action is not None
            return "SUCCESS", (state - node.action.delete) | node.action.add
        if node.kind == "sequence":
            current = state
            for c in node.children:
                r, current = visit(c, current)
                if r == "FAILURE":
                    return "FAILURE", state
            return "SUCCESS", current
        # fallback
        for c in node.children:
            r, new_state = visit(c, state)
            if r == "SUCCESS":
                return "SUCCESS", new_state
        return "FAILURE", state

    visit(root, s0)
    return found[0]


# ---------- Expand (Cai AAAI 2021 Algorithm 1) ----------

def _expand_condition(
    cond: frozenset[Literal],
    models: Iterable[ActionModel],
) -> BTNode | None:
    """Return the Fallback subtree for expanding `cond`, or None if no
    applicable grounded actions exist.

    Shape (after expansion):
        Fallback(
            Condition(cond),
            Sequence(Condition(c_attr_1), Action(a_1)),
            ...
            Sequence(Condition(c_attr_k), Action(a_k)),
        )

    where a_i ranges over grounded actions applicable to `cond` per Cai's
    definition:
      1. cond ∩ (pre(a) ∪ add(a) \\ del(a)) ≠ ∅       (a is relevant)
      2. cond \\ del(a) == cond                        (a does not undo cond)
    """
    children: list[BTNode] = [BTNode(kind="condition", condition=cond)]

    # Deduplicate grounded instances produced by different target literals
    seen: set[tuple[str, tuple[LiteralArg, ...]]] = set()

    for model in models:
        for target in cond:
            grounded = _ground_model_for_literal(model, target)
            if grounded is None:
                continue
            key = (grounded.name, tuple(
                sorted(
                    b for f, b in zip(model.params,
                                      _binding_order(model, grounded))
                )
            ))
            if key in seen:
                continue
            seen.add(key)

            usable = grounded.add - grounded.delete
            if not (cond & (grounded.pre | usable)):
                continue
            if cond - grounded.delete != cond:
                continue

            c_attr = frozenset((grounded.pre | cond) - grounded.add)
            seq_child = BTNode(
                kind="sequence",
                children=(
                    BTNode(kind="condition", condition=c_attr),
                    BTNode(kind="action", action=grounded),
                ),
            )
            children.append(seq_child)

    if len(children) == 1:
        return None
    return BTNode(kind="fallback", children=tuple(children))


def _binding_order(
    template: ActionModel, grounded: ActionModel
) -> tuple[LiteralArg, ...]:
    """Extract the grounded values for each of template's formal parameters
    by pairing the first template add-literal that contains all formals with
    the corresponding grounded add-literal.
    """
    for t_add in template.add:
        formals_in_add = [a for a in t_add.args if isinstance(a, str) and a.startswith("?")]
        if set(formals_in_add) != set(template.params):
            continue
        for g_add in grounded.add:
            if g_add.name == t_add.name and len(g_add.args) == len(t_add.args):
                binding = _unify_literals(t_add, g_add)
                if binding is not None and all(f in binding for f in template.params):
                    return tuple(binding[f] for f in template.params)
    # Fallback: build a canonical tuple from whatever params we can recover.
    # Should not happen for our two-skill domain.
    return tuple(
        str(p)
        for p in template.params
    )


def _replace_condition_in_tree(
    root: BTNode,
    target: frozenset[Literal],
    replacement: BTNode,
) -> BTNode:
    """Return a new tree with every Condition(target) swapped for replacement."""
    if root.kind == "condition" and root.condition == target:
        return replacement
    if root.kind in ("sequence", "fallback"):
        new_children = tuple(
            _replace_condition_in_tree(c, target, replacement) for c in root.children
        )
        return BTNode(kind=root.kind, children=new_children)
    return root


# ---------- Main entry point (Cai AAAI 2021 Algorithm 2) ----------

def bt_expand(
    goal: frozenset[Literal],
    models: Iterable[ActionModel],
    s0: frozenset[Literal],
    *,
    max_iterations: int = 512,
) -> BTNode:
    """Return a BTNode tree that is Finite-Time Successful with s0 ∈ ROA.

    Multi-literal goals are decomposed into per-literal sub-goals (OBTEA
    IJCAI 2024 decomposition), each expanded independently against s0 and
    then chained in a top-level Sequence. The Fallback-pattern guards in
    each sub-tree handle dynamic state propagation at Tick time, so
    sub-tree_i correctly operates on the evolved state left behind by
    sub-tree_{i-1}, not on s0 itself.

    Rationale: for actions that interfere on shared resources (our Insert
    deletes holding(peg); successive inserts on the same peg need re-Lifts),
    single-pass regression through a conjunctive condition produces
    physically inconsistent c_attrs. Per-literal decomposition avoids this
    by never conjoining unrelated sub-goals in a single regression step.

    Raises ExpansionFailure when no unexpanded FAILING condition can be
    satisfied or max_iterations is exceeded.
    """
    if not goal:
        raise ExpansionFailure(
            "empty goal condition — nothing to plan for",
            unreached=frozenset(),
            iterations=0,
        )
    models_list = list(models)

    if len(goal) == 1:
        return _bt_expand_single(goal, models_list, s0, max_iterations=max_iterations)

    # Multi-literal goal: decompose, chain in Sequence (OBTEA-style).
    sub_trees: list[BTNode] = []
    for literal in sorted(goal, key=str):
        sub = _bt_expand_single(
            frozenset({literal}),
            models_list,
            s0,
            max_iterations=max_iterations,
        )
        sub_trees.append(sub)
    return BTNode(kind="sequence", children=tuple(sub_trees))


def _bt_expand_single(
    goal: frozenset[Literal],
    models_list: list[ActionModel],
    s0: frozenset[Literal],
    *,
    max_iterations: int,
) -> BTNode:
    """Cai AAAI 2021 Algorithm 2 on a single goal condition."""
    tree: BTNode = BTNode(kind="condition", condition=goal)
    expanded: set[frozenset[Literal]] = set()

    for iteration in range(max_iterations):
        result, _ = _tick(tree, s0)
        if result == "SUCCESS":
            return tree
        cond = _next_unexpanded_condition(tree, s0, expanded)
        if cond is None:
            raise ExpansionFailure(
                "no unexpanded FAILING condition found but root still FAILS; "
                "likely an action library gap (no action achieves some sub-goal)",
                unreached=goal,
                iterations=iteration,
            )
        expansion = _expand_condition(cond, models_list)
        if expansion is None:
            raise ExpansionFailure(
                f"no applicable action for condition {sorted(str(x) for x in cond)}",
                unreached=cond,
                iterations=iteration,
            )
        tree = _replace_condition_in_tree(tree, cond, expansion)
        expanded.add(cond)

    raise ExpansionFailure(
        f"max_iterations={max_iterations} exceeded",
        unreached=goal,
        iterations=max_iterations,
    )


# ---------- Tree inspection helpers (used by tests + visualiser) ----------

def tree_depth(node: BTNode) -> int:
    """Maximum root-to-leaf path length (edges)."""
    if not node.children:
        return 0
    return 1 + max(tree_depth(c) for c in node.children)


def tree_node_count(node: BTNode) -> int:
    """Total number of nodes in the tree."""
    return 1 + sum(tree_node_count(c) for c in node.children)


def count_nodes_by_kind(node: BTNode) -> dict[str, int]:
    """{'action': n, 'condition': n, 'sequence': n, 'fallback': n}."""
    acc = {k: 0 for k in _ALL_KINDS}

    def walk(n: BTNode) -> None:
        acc[n.kind] = acc.get(n.kind, 0) + 1
        for c in n.children:
            walk(c)

    walk(node)
    return acc


def is_ppa_compliant(node: BTNode) -> bool:
    """True iff every Fallback node satisfies the PPA pattern:

        Fallback(
            Condition(...),
            Sequence(Condition(...), Action(...)),
            ...
        )

    (first child is a Condition, remaining children are Sequences whose last
    leaf is an Action. Matches the one-step expansion shape emitted by
    _expand_condition.)
    """
    if node.kind == "fallback":
        if not node.children:
            return False
        if node.children[0].kind != "condition":
            return False
        for seq in node.children[1:]:
            if seq.kind != "sequence":
                return False
            if not seq.children:
                return False
            if seq.children[-1].kind != "action":
                return False
    # Recurse
    for c in node.children:
        if not is_ppa_compliant(c):
            return False
    return True


def unicode_render(node: BTNode, indent: str = "", is_last: bool = True) -> str:
    """Monospace tree rendering for CLI / debugging."""
    prefix = indent + ("└── " if is_last else "├── ")
    if node.kind == "condition":
        body = "Condition(" + ", ".join(sorted(str(l) for l in node.condition)) + ")"
    elif node.kind == "action":
        ga = node.action
        args = ", ".join(str(a) for a in ga.params) if ga.params else ""
        ground_args = _extract_ground_args(ga)
        arg_str = ", ".join(str(a) for a in ground_args) if ground_args else ""
        body = f"Action({ga.name}({arg_str}))"
    else:
        body = f"{node.kind.capitalize()}[{len(node.children)}]"
    lines = [prefix + body]
    new_indent = indent + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        lines.append(unicode_render(child, new_indent, i == len(node.children) - 1))
    return "\n".join(lines)


def _extract_ground_args(ga: ActionModel) -> tuple[LiteralArg, ...]:
    """Recover grounded argument values by inspecting the add-set.

    Used for display only. Returns empty tuple if ambiguous.
    """
    for add_lit in ga.add:
        if len(add_lit.args) == len(ga.params):
            if all(not (isinstance(a, str) and a.startswith("?")) for a in add_lit.args):
                return add_lit.args
    return ()
