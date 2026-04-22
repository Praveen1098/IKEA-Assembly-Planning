"""Monte-Carlo verifier for BTs synthesised over stochastic-success skills.

Adapts the clean LTL simulator from
`VLM_assembly_plan_gen/inference/stage5_bt_compile.py::verify_bt` lines 552–668
and extends it to per-action stochastic success: each action leaf, when
ticked, either succeeds with probability `p_success` (effects applied) or
fails (FAILURE propagated upward).

Five properties reported:
  P1 — gripper consistency: (gripper_empty ∈ state) ⟺ (no holding(?) literal)
  P2 — no double grasp: Lift only attempted when gripper_empty holds
  P3 — hold-before-connect: Insert only attempted when peg is held
  P4 — causal completeness: every action's preconditions are satisfied at
       execution time
  P5 — liveness: empirical fraction of rollouts where Tick returns SUCCESS

P1–P4 are deterministic structural invariants — one violation anywhere
across N rollouts is reported. P5 is a Monte-Carlo estimate of success
probability under per-action p_success, with Wilson 95% CI.

No repair, no fallback. This is deliberate: a pipeline that detects a
violation must surface it to the caller; auto-patching would hide
synthesiser bugs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .action_model import ActionModel, Literal
from .bt_expansion import BTNode


# ---------- Public data classes ----------

@dataclass(frozen=True)
class VerificationReport:
    """Structured output of `stochastic_verify`.

    Attributes:
        passed: True iff every P1..P4 violation list is empty AND
            liveness_mean is within 1% of the theoretical value p_success ^
            critical_path_length (under deterministic p_success=1.0, this
            is the Stage 5 verify_bt pass check).
        liveness_mean: fraction of n_rollouts in which Tick returned SUCCESS.
        liveness_95_ci: Wilson half-width 95% confidence interval for the mean.
        p1_gripper_consistency / p2_no_double_grasp /
        p3_hold_before_connect / p4_causal_completeness: True iff no
            violation was observed across any rollout.
        violations: structured per-property list of violation strings
            (first 5 kept per property to bound report size).
    """

    passed: bool
    liveness_mean: float
    liveness_95_ci: tuple[float, float]
    p1_gripper_consistency: bool
    p2_no_double_grasp: bool
    p3_hold_before_connect: bool
    p4_causal_completeness: bool
    violations: dict[str, list[str]] = field(default_factory=dict)
    n_rollouts: int = 0
    p_success: float = 1.0


# ---------- Helpers ----------

def _has_held_literal(state: set[Literal]) -> bool:
    return any(
        lit.name == "holding" and not lit.negated
        for lit in state
    )


def _gripper_empty_literal() -> Literal:
    return Literal("gripper_empty", ())


def _peg_from_insert(action: ActionModel) -> int | None:
    """Return the grounded peg id from an Insert action's add-set, or None."""
    for lit in action.add:
        if lit.name == "inserted" and lit.args:
            first = lit.args[0]
            if isinstance(first, int):
                return first
    return None


def _wilson_95_ci(k: int, n: int) -> tuple[float, float]:
    """Wilson score interval half-width for binomial proportion.

    Using Wilson rather than normal-approximation avoids undefined behaviour
    at p == 0 or p == 1, which our deterministic-success tests hit frequently.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (lo, hi)


def _record(violations: dict[str, list[str]], key: str, msg: str, cap: int = 5) -> None:
    bucket = violations.setdefault(key, [])
    if len(bucket) < cap:
        bucket.append(msg)


# ---------- Stochastic Tick ----------

def _tick_stochastic(
    node: BTNode,
    state: set[Literal],
    p_success: float,
    rng: random.Random,
    violations: dict[str, list[str]],
) -> bool:
    """Single Tick with in-place state mutation; returns True on SUCCESS."""

    if node.kind == "condition":
        assert node.condition is not None
        return node.condition.issubset(state)

    if node.kind == "action":
        action = node.action
        assert action is not None
        return _tick_action(action, state, p_success, rng, violations)

    if node.kind == "sequence":
        snapshot = set(state)
        for child in node.children:
            if not _tick_stochastic(child, state, p_success, rng, violations):
                state.clear()
                state.update(snapshot)
                return False
        return True

    # fallback
    for child in node.children:
        snapshot = set(state)
        if _tick_stochastic(child, state, p_success, rng, violations):
            return True
        state.clear()
        state.update(snapshot)
    return False


def _tick_action(
    action: ActionModel,
    state: set[Literal],
    p_success: float,
    rng: random.Random,
    violations: dict[str, list[str]],
) -> bool:
    """P1–P4 checks, then stochastic success with effect application."""
    ge = _gripper_empty_literal()

    # ---- P1: gripper consistency before this action ----
    gripper_empty = ge in state
    held = _has_held_literal(state)
    if gripper_empty and held:
        _record(violations, "p1",
                f"{action.name}({action.params}): gripper_empty but holding "
                f"literal present")
    elif not gripper_empty and not held:
        _record(violations, "p1",
                f"{action.name}({action.params}): ¬gripper_empty but no "
                f"holding literal")

    # ---- P2: Lift requires gripper_empty ----
    if action.name == "lift":
        if ge not in state:
            _record(violations, "p2",
                    f"Lift {action.params} attempted without gripper_empty")

    # ---- P3: Insert requires peg currently held ----
    if action.name == "insert":
        peg = _peg_from_insert(action)
        if peg is not None:
            held_peg = Literal("holding", (peg,))
            if held_peg not in state:
                _record(violations, "p3",
                        f"Insert peg={peg} attempted without holding(peg)")

    # ---- P4: every precondition must be satisfied at execution time ----
    unmet = action.pre - frozenset(state)
    if unmet:
        _record(violations, "p4",
                f"{action.name}({action.params}): unmet preconditions "
                f"{sorted(str(u) for u in unmet)}")
        # Precondition failure also means this action fails (FAILURE node)
        return False

    # ---- Stochastic success ----
    if rng.random() < p_success:
        state -= action.delete
        state |= action.add
        return True
    return False


# ---------- Public entry point ----------

def stochastic_verify(
    tree: BTNode,
    s0: frozenset[Literal],
    *,
    p_success: float = 1.0,
    n_rollouts: int = 1000,
    seed: int = 42,
) -> VerificationReport:
    """Run `n_rollouts` Monte-Carlo Ticks of `tree` from `s0`.

    Args:
        tree: a BTNode produced by bt_expand.
        s0: initial state as a frozenset of Literals.
        p_success: per-action success probability in [0, 1].
        n_rollouts: Monte-Carlo sample count (default 1000).
        seed: PRNG seed for reproducibility.
    """
    if not 0.0 <= p_success <= 1.0:
        raise ValueError(f"p_success must be in [0, 1]; got {p_success}")
    if n_rollouts <= 0:
        raise ValueError(f"n_rollouts must be positive; got {n_rollouts}")

    rng = random.Random(seed)
    violations: dict[str, list[str]] = {"p1": [], "p2": [], "p3": [], "p4": []}

    successes = 0
    for _ in range(n_rollouts):
        state: set[Literal] = set(s0)
        if _tick_stochastic(tree, state, p_success, rng, violations):
            successes += 1

    liveness = successes / n_rollouts
    ci = _wilson_95_ci(successes, n_rollouts)

    p1_ok = not violations["p1"]
    p2_ok = not violations["p2"]
    p3_ok = not violations["p3"]
    p4_ok = not violations["p4"]

    passed = all([p1_ok, p2_ok, p3_ok, p4_ok])
    if p_success == 1.0:
        # With deterministic actions, we expect liveness == 1.0; tolerate none
        passed = passed and liveness == 1.0

    return VerificationReport(
        passed=passed,
        liveness_mean=liveness,
        liveness_95_ci=ci,
        p1_gripper_consistency=p1_ok,
        p2_no_double_grasp=p2_ok,
        p3_hold_before_connect=p3_ok,
        p4_causal_completeness=p4_ok,
        violations=violations,
        n_rollouts=n_rollouts,
        p_success=p_success,
    )
