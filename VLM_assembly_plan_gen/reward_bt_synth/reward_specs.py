"""Skill specifications for IKEA assembly in robosuite/IsaacLab convention.

Each skill is specified by three top-level functions:

  _check_success_<skill>(obs, *params) -> bool
      The terminal-success predicate. The AST extractor analyses its AST to
      derive the skill's add-set. Mirrors robosuite's env `_check_success`
      method (see robosuite/environments/manipulation/lift.py::Lift).

  reward_<skill>(obs, action, *params) -> float
      The reward function. Returns TERMINAL_REWARD when `_check_success_*`
      holds, otherwise a dense shaping term. Mirrors robosuite's env
      `reward(action)` method. The AST extractor does NOT analyse this
      function; it only uses the existence of the terminal-return branch as
      a structural invariant.

  reset_to_init_<skill>(obs, *params) -> None
      Initial-state distribution. Calls a sequence of @initializer-registered
      setter functions that establish the training-time curriculum's initial
      predicates. Mirrors robosuite's env `_reset_internal` method. The AST
      extractor analyses it to derive the skill's pre-set.

Predicates and initialisers are registered via decorators from vocabulary.py.
Their Python bodies are runtime code that would be used on a real robot; the
AST extractor only uses their registered names + templates.

IMPORTANT: modifying a literal in `_check_success_lift` or `_check_success_insert`
will cause the extracted ActionModel to change. That is the design — it is the
falsifiable claim that the extractor reads the code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .vocabulary import initializer, predicate

# Robosuite's standard sparse-completion reward value (see lift.py line ~180).
TERMINAL_REWARD: float = 2.25


# ============================================================================
# Predicates — runtime evaluators used by reward / check functions.
# The AST extractor ONLY uses @predicate templates, not bodies.
# ============================================================================

@predicate("holding(?p)")
def holding(obs: Any, p: int) -> bool:
    """True iff the gripper currently grips part `p`."""
    return getattr(obs, "held_id", None) == p


@predicate("oriented(?p)")
def oriented(obs: Any, p: int) -> bool:
    """True iff part `p` is within orientation tolerance of its target pose."""
    return obs.orientation_error(p) < 0.05  # 0.05 rad ~ 3 deg


@predicate("on_surface(?p)")
def on_surface(obs: Any, p: int) -> bool:
    """True iff part `p` rests on the work surface."""
    return obs.part_on_table(p)


@predicate("accessible(?p)")
def accessible(obs: Any, p: int) -> bool:
    """True iff the gripper has a collision-free path to grasp part `p`."""
    return obs.reachable(p)


@predicate("gripper_empty")
def gripper_empty(obs: Any) -> bool:
    """True iff the gripper holds no part."""
    return getattr(obs, "held_id", None) is None


@predicate("inserted(?peg, ?socket)")
def inserted(obs: Any, peg: int, socket: int) -> bool:
    """True iff `peg` is fully seated in `socket` within positional tolerance."""
    return obs.peg_hole_distance(peg, socket) < 0.005  # 5 mm


# ============================================================================
# Initialisers — establish initial-state predicates during env reset.
# @initializer templates identify which predicate each call asserts.
# ============================================================================

@initializer("on_surface(?p)")
def init_on_surface(obs: Any, p: int) -> None:
    obs.place_on_table(p)


@initializer("accessible(?p)")
def init_accessible(obs: Any, p: int) -> None:
    obs.mark_reachable(p)


@initializer("gripper_empty")
def init_gripper_empty(obs: Any) -> None:
    obs.held_id = None


@initializer("holding(?p)")
def init_held(obs: Any, p: int) -> None:
    obs.held_id = p


@initializer("oriented(?p)")
def init_oriented(obs: Any, p: int) -> None:
    obs.orient_to_target(p)


# ============================================================================
# Dense shaping helpers (not analysed by extractor).
# ============================================================================

def _grip_distance(obs: Any, part: int) -> float:
    return obs.distance(obs.gripper_pos, obs.part_pos(part))


def _peg_hole_distance(obs: Any, peg: int, socket: int) -> float:
    return obs.distance(obs.part_pos(peg), obs.hole_pos(socket))


def shaping_lift(obs: Any, part: int) -> float:
    return 1.0 - float(np.tanh(10.0 * _grip_distance(obs, part)))


def shaping_insert(obs: Any, peg: int, socket: int) -> float:
    return 1.0 - float(np.tanh(10.0 * _peg_hole_distance(obs, peg, socket)))


# ============================================================================
# Lift skill — bundled reach + grasp + reorient.
# Matches robosuite.environments.manipulation.Lift structure.
# ============================================================================

def _check_success_lift(obs: Any, part: int) -> bool:
    """Terminal success predicate for Lift. AST-extracted for add(Lift)."""
    return holding(obs, part) and oriented(obs, part)


def reward_lift(obs: Any, action: Any, part: int) -> float:
    """Mirrors robosuite's Lift.reward(action) convention."""
    if _check_success_lift(obs, part):
        return TERMINAL_REWARD
    return shaping_lift(obs, part)


def reset_to_init_lift(obs: Any, part: int) -> None:
    """Initial-state distribution for Lift. AST-extracted for pre(Lift)."""
    init_on_surface(obs, part)
    init_accessible(obs, part)
    init_gripper_empty(obs)


# ============================================================================
# Insert skill — peg-in-hole with gripper release.
# Matches robosuite.environments.manipulation.Assembly peg-in-hole structure.
# ============================================================================

def _check_success_insert(obs: Any, peg: int, socket: int) -> bool:
    """Terminal success predicate for Insert. AST-extracted for add(Insert).

    The `on_surface(obs, peg)` conjunct encodes a physical reality: at the
    moment insertion succeeds, the peg is mechanically fixed into the socket,
    which is itself resting on the work surface — so the peg is on_surface
    (as a rigid member of the subassembly). Without this conjunct, the
    extracted STRIPS tuple loses the ability to re-Lift parts that appear
    as pegs in multiple connections (e.g. hub-and-spoke topologies).
    """
    return (
        inserted(obs, peg, socket)
        and gripper_empty(obs)
        and on_surface(obs, peg)
    )


def reward_insert(obs: Any, action: Any, peg: int, socket: int) -> float:
    """Mirrors robosuite's Assembly.reward(action) convention."""
    if _check_success_insert(obs, peg, socket):
        return TERMINAL_REWARD
    return shaping_insert(obs, peg, socket)


def reset_to_init_insert(obs: Any, peg: int, socket: int) -> None:
    """Initial-state distribution for Insert. AST-extracted for pre(Insert)."""
    init_held(obs, peg)
    init_oriented(obs, peg)
    init_accessible(obs, socket)
