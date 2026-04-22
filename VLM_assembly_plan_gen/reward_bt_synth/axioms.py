"""Domain axioms for IKEA assembly physics + Horn-chain entailment.

The axioms encode rigid physical facts over the predicate vocabulary defined
in reward_specs.py — they are not heuristics. They are used by the AST
extractor to derive the STRIPS `delete` set from the interaction between
preconditions and the terminal-success literals.

Representation:
  Each axiom is a single Horn implication of the form
      body_literal  ⇒  head_literal
  where body and head are `Literal` instances over formal parameters (?p, ?q).

Entailment:
  `entails(lhs, rhs)` returns True iff there is a one-hop Horn match: some
  axiom whose body unifies with `lhs` and whose head unifies with `rhs`,
  under a parameter binding consistent on both sides. One hop is sufficient
  for the IKEA assembly domain; deeper chaining is intentionally unsupported
  to keep the algorithm polynomial and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .action_model import Literal, LiteralArg


@dataclass(frozen=True)
class Axiom:
    """A single Horn rule: body  ⇒  head."""

    body: Literal
    head: Literal

    def __str__(self) -> str:  # pragma: no cover - diagnostic only
        return f"{self.body}  ⇒  {self.head}"


# ---------- Parameter unification ----------
def _unify(
    template: Literal,
    concrete: Literal,
) -> dict[str, LiteralArg] | None:
    """Return a binding {?p: arg} that makes template equal concrete, or None.

    Names and arities must match exactly (no up-casting). Negation must match.
    Each formal ?-parameter in `template` is bound to the corresponding
    concrete argument in `concrete`. Repeated formals must bind consistently.
    """
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
            # Non-variable position — require literal equality
            if t_arg != c_arg:
                return None
    return binding


def _apply(binding: dict[str, LiteralArg], lit: Literal) -> Literal:
    """Substitute the binding through `lit`'s arguments."""
    return lit.ground(binding)


# ---------- Canonical IKEA axiom set ----------
# Variables: ?p ranges over parts; ?q over parts. Axioms are kept minimal and
# rigid so their consequences are easy to audit.

# A1: holding(?p) ⇒ ¬on_surface(?p)
# A2: holding(?p) ⇒ ¬gripper_empty
# A3: gripper_empty ⇒ ¬holding(?p)  (dual of A2 — needed for Insert's delete)
# A4: inserted(?p, ?q) ⇒ ¬holding(?p)
# A5: inserted(?p, ?q) ⇒ gripper_empty  (successful release implies gripper empty)
# A6: on_surface(?p) ⇒ ¬holding(?p)  (dual of A1 — symmetric closure)

IKEA_AXIOMS: tuple[Axiom, ...] = (
    Axiom(
        body=Literal("holding", ("?p",)),
        head=Literal("on_surface", ("?p",), negated=True),
    ),
    Axiom(
        body=Literal("holding", ("?p",)),
        head=Literal("gripper_empty", (), negated=True),
    ),
    Axiom(
        body=Literal("gripper_empty", ()),
        head=Literal("holding", ("?p",), negated=True),
    ),
    Axiom(
        body=Literal("inserted", ("?p", "?q")),
        head=Literal("holding", ("?p",), negated=True),
    ),
    Axiom(
        body=Literal("inserted", ("?p", "?q")),
        head=Literal("gripper_empty", ()),
    ),
    Axiom(
        body=Literal("on_surface", ("?p",)),
        head=Literal("holding", ("?p",), negated=True),
    ),
)


# ---------- Public entailment API ----------
def entails(
    lhs: Literal,
    rhs: Literal,
    axioms: tuple[Axiom, ...] = IKEA_AXIOMS,
) -> bool:
    """Return True iff `lhs ⇒ rhs` under a one-hop Horn match over `axioms`.

    Algorithm:
      For each axiom a, try binding a.body to lhs. If it unifies, apply the
      binding to a.head. If the grounded head equals rhs, return True.

    Time: O(|axioms|), negligible given our six rules.
    """
    for ax in axioms:
        binding = _unify(ax.body, lhs)
        if binding is None:
            continue
        grounded_head = _apply(binding, ax.head)
        # Account for formals in the head that were not bound by the body
        # (e.g. A3's head mentions ?p which isn't in the 0-arity body). In
        # that case, unify the still-symbolic head against rhs.
        if all(
            not (isinstance(a, str) and a.startswith("?"))
            for a in grounded_head.args
        ):
            if grounded_head == rhs:
                return True
        else:
            # Second-stage unification: match head against rhs, extending binding
            second = _unify(grounded_head, rhs)
            if second is not None:
                return True
    return False


def negate(lit: Literal) -> Literal:
    """Convenience — the ast_extractor calls this directly."""
    return lit.negate()
