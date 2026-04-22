"""Grounded STRIPS action model + literal representation + pretty-printing.

A `Literal` is a first-order atom carrying a predicate name, a tuple of
positional arguments (strings for skill-formal parameters like "?peg", or ints
for grounded part ids), and a `negated` flag. Equality and hashing treat
a literal as an immutable tuple, so frozenset-set operations (used all over the
extractor and expander) behave correctly.

An `ActionModel` is the tuple ⟨pre, add, delete⟩ produced by the AST extractor
for one skill. Frozensets are required — the BT Expansion algorithm performs
set intersections and differences at every iteration.

Both classes are `frozen=True` dataclasses so instances are hashable and can
be stored in sets / used as dict keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# A literal argument is either a formal parameter name (starts with "?") or an
# integer part id. Runtime code uses this as the single source of truth for
# argument types. Kept as a type alias, not an enum, because Python `ast`
# gives us strings/ints directly.
LiteralArg = Union[str, int]


@dataclass(frozen=True, order=True)
class Literal:
    """Immutable first-order atom.

    Examples:
        Literal("holding", ("?p",), False)           → holding(?p)
        Literal("inserted", ("?peg", "?socket"), False) → inserted(?peg, ?socket)
        Literal("gripper_empty", (), False)          → gripper_empty
        Literal("on_surface", ("?p",), True)         → ¬on_surface(?p)
    """

    name: str
    args: tuple[LiteralArg, ...] = field(default_factory=tuple)
    negated: bool = False

    def negate(self) -> "Literal":
        """Return the logical negation of this literal (flip the `negated` flag)."""
        return Literal(self.name, self.args, not self.negated)

    def ground(self, binding: dict[str, LiteralArg]) -> "Literal":
        """Substitute formal parameters (strings starting with '?') per binding.

        Unmapped formals are left as-is. This preserves symbolic literals when
        only some arguments are known; full grounding requires a total binding.
        """
        new_args = tuple(binding.get(a, a) if isinstance(a, str) else a for a in self.args)
        return Literal(self.name, new_args, self.negated)

    def __str__(self) -> str:
        body = f"{self.name}({', '.join(str(a) for a in self.args)})" if self.args else self.name
        return f"¬{body}" if self.negated else body


@dataclass(frozen=True)
class ActionModel:
    """STRIPS action model ⟨pre, add, delete⟩ for one skill.

    pre, add, delete are frozensets of Literals. params is the tuple of formal
    parameter names in declaration order (matches the skill's reward function
    signature, excluding the implicit first `obs` argument).

    Invariant (checked by the extractor, not re-checked here):
        add & delete == frozenset()
        add & pre - delete == frozenset()
    """

    name: str
    params: tuple[str, ...]
    pre: frozenset[Literal]
    add: frozenset[Literal]
    delete: frozenset[Literal]
    cost: float = 1.0

    def to_pretty_string(self) -> str:
        """Multi-line derivation-style display for the CLI `extract` subcommand.

        Matches the slide-trace format used in the project plan.
        """
        params_str = ", ".join(self.params) if self.params else "—"

        def block(title: str, literals: frozenset[Literal]) -> str:
            if not literals:
                return f"    {title:<8}= ∅"
            items = sorted(str(lit) for lit in literals)
            if len(items) == 1:
                return f"    {title:<8}= {{ {items[0]} }}"
            body = "\n               ".join(items)
            return f"    {title:<8}= {{ {body} }}"

        lines = [
            f"{self.name.capitalize()}({params_str})",
            block("pre", self.pre),
            block("add", self.add),
            block("delete", self.delete),
            f"    cost    = {self.cost}",
        ]
        return "\n".join(lines)

    def ground_all(self, binding: dict[str, LiteralArg]) -> "ActionModel":
        """Return a copy with every literal grounded via `binding`."""
        return ActionModel(
            name=self.name,
            params=self.params,
            pre=frozenset(l.ground(binding) for l in self.pre),
            add=frozenset(l.ground(binding) for l in self.add),
            delete=frozenset(l.ground(binding) for l in self.delete),
            cost=self.cost,
        )
