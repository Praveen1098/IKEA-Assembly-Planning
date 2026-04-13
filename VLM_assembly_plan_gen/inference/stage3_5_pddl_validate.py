"""Stage 3.5: PDDL domain auto-generation and logical consistency validation.

Reads actions.json (output of Stage 3) and verifies that the extracted assembly
sequence is logically executable: every connection goal can be reached from the
initial state using the defined skill primitives.

Algorithm:
  1. Auto-generate a PDDL domain from the static SKILL_LIBRARY
     (predicates: holding, gripper_empty, accessible, inserted, pressed, screwed)
  2. Auto-generate a PDDL problem from actions.json
     (init: all parts accessible + gripper empty; goal: all connections achieved)
  3. Run PyPerPlan (pure Python PDDL planner) to check solvability
  4. Report success or diagnose which goals are unreachable

Dependency: pyperplan  (pip install pyperplan)

Literature & open-source references:
  - PMC11504948 — "Autonomous Robot Task Execution in Flexible Manufacturing:
    Integrating PDDL and Behavior Trees in ARIAC 2023"
  - SoarGroup assembly.pddl + gripper.pddl PDDL domain templates:
    github.com/SoarGroup/Domains-Planning-Domain-Definition-Language
  - PyPerPlan pure Python PDDL planner: github.com/aibasel/pyperplan
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Allow running as a standalone script from inference/
sys.path.append(str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Skill library — PDDL-style pre/postconditions per primitive
# Adapted from SoarGroup gripper.pddl and assembly.pddl templates
# ---------------------------------------------------------------------------

SKILL_LIBRARY = {
    "GRASP": {
        "pre":  ["accessible(?p)", "gripper-empty"],
        "post": ["holding(?p)", "not(gripper-empty)"],
    },
    "INSERT": {
        "pre":  ["holding(?active)", "accessible(?passive)"],
        "post": ["inserted(?active,?passive)", "not(holding(?active))", "gripper-empty"],
    },
    "PRESS": {
        "pre":  ["holding(?active)", "accessible(?passive)"],
        "post": ["pressed(?active,?passive)", "not(holding(?active))", "gripper-empty"],
    },
    "SCREW": {
        "pre":  ["holding(?active)", "accessible(?passive)"],
        "post": ["screwed(?active,?passive)", "not(holding(?active))", "gripper-empty"],
    },
    "REORIENT": {
        "pre":  ["holding(?p)"],
        "post": ["holding(?p)"],   # grip unchanged; orientation updated externally
    },
}


# ---------------------------------------------------------------------------
# PDDL domain / problem generation
# ---------------------------------------------------------------------------

def generate_pddl_domain() -> str:
    """Return a PDDL domain string for IKEA assembly skills.

    Domain adapted from:
      - SoarGroup gripper.pddl (pick/drop with gripper-empty tracking)
      - SoarGroup assembly.pddl (part incorporation predicates)
    Predicates cover all primitives in SKILL_LIBRARY.
    """
    return """\
(define (domain ikea-assembly)
  (:requirements :strips :negative-preconditions)
  (:predicates
    (holding ?p)
    (gripper-empty)
    (accessible ?p)
    (inserted ?a ?b)
    (pressed ?a ?b)
    (screwed ?a ?b))

  (:action grasp
    :parameters (?p)
    :precondition (and (accessible ?p) (gripper-empty))
    :effect (and (holding ?p) (not (gripper-empty))))

  (:action release
    :parameters (?p)
    :precondition (holding ?p)
    :effect (and (not (holding ?p)) (gripper-empty)))

  (:action insert
    :parameters (?active ?passive)
    :precondition (and (holding ?active) (accessible ?passive))
    :effect (and (inserted ?active ?passive)
                 (not (holding ?active)) (gripper-empty)))

  (:action press
    :parameters (?active ?passive)
    :precondition (and (holding ?active) (accessible ?passive))
    :effect (and (pressed ?active ?passive)
                 (not (holding ?active)) (gripper-empty)))

  (:action screw
    :parameters (?active ?passive)
    :precondition (and (holding ?active) (accessible ?passive))
    :effect (and (screwed ?active ?passive)
                 (not (holding ?active)) (gripper-empty)))

  (:action reorient
    :parameters (?p)
    :precondition (holding ?p)
    :effect (holding ?p)))
"""


def generate_pddl_problem(parts: list[int], connections: list[dict]) -> str:
    """Return a PDDL problem string derived from actions.json.

    Initial state: all parts accessible, gripper empty.
    Goal: every connection in all_connections is achieved.

    Adapted from SoarGroup assembly problem template.
    """
    objects_str = " ".join(f"part{p}" for p in sorted(parts))
    init_lines = ["(gripper-empty)"] + [f"(accessible part{p})" for p in sorted(parts)]
    init_str = "\n    ".join(init_lines)

    goal_atoms = []
    for conn in connections:
        pred = conn.get("predicate")   # "inserted", "pressed", "screwed"
        p1 = conn.get("part1")
        p2 = conn.get("part2")
        if pred and p1 is not None and p2 is not None:
            goal_atoms.append(f"({pred} part{p1} part{p2})")

    if not goal_atoms:
        # No goals — trivially satisfied
        goal_str = "(gripper-empty)"
    else:
        goal_str = "\n    ".join(goal_atoms)

    return f"""\
(define (problem ikea-assembly-problem)
  (:domain ikea-assembly)
  (:objects {objects_str})
  (:init
    {init_str})
  (:goal
    (and
    {goal_str})))
"""


# ---------------------------------------------------------------------------
# PyPerPlan runner
# ---------------------------------------------------------------------------

def _run_pyperplan(domain_file: str, problem_file: str):
    """Run PyPerPlan and return the plan (list of operators) or None.

    Uses pyperplan's search_plan(domain, problem, search, heuristic_class) API.
    search   — greedy_best_first_search callable from pyperplan.search
    heuristic_class — hFFHeuristic class from pyperplan.heuristics.relaxation
    """
    try:
        from pyperplan import planner as pp
        from pyperplan.search import greedy_best_first_search
        from pyperplan.heuristics.relaxation import hFFHeuristic

        plan = pp.search_plan(
            domain_file, problem_file,
            search=greedy_best_first_search,
            heuristic_class=hFFHeuristic,
        )
        return plan
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Diagnosis helper
# ---------------------------------------------------------------------------

def _diagnose(parts: list[int], connections: list[dict]) -> None:
    """Print which goal predicates cannot be produced from the skill library."""
    achievable_predicates = set()
    for skill in SKILL_LIBRARY.values():
        for eff in skill["post"]:
            # Extract predicate name (e.g. "inserted(?active,?passive)" → "inserted")
            pred_name = eff.split("(")[0].lstrip("not(").strip()
            if pred_name and not pred_name.startswith("not"):
                achievable_predicates.add(pred_name)

    print("\n[PDDL Diagnosis]")
    for conn in connections:
        pred = conn.get("predicate", "")
        p1 = conn.get("part1")
        p2 = conn.get("part2")
        if pred not in achievable_predicates:
            print(f"  WARNING: predicate '{pred}' (parts {p1},{p2}) has no "
                  f"producing action in SKILL_LIBRARY — check connection type.")
        else:
            print(f"  OK: {pred}(part{p1}, part{p2}) is achievable")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_actions_json(actions_path: str, verbose: bool = True) -> bool:
    """Validate actions.json logical consistency via PDDL + PyPerPlan.

    Generates a PDDL domain + problem from the file, runs PyPerPlan, and
    reports whether a valid plan exists.

    Returns True if the assembly is logically consistent (plan found).
    """
    with open(actions_path, "r") as f:
        data = json.load(f)

    furniture = data.get("furniture", "unknown")
    parts = data.get("parts", [])
    connections = data.get("all_connections", [])

    if not connections:
        if verbose:
            print(f"[PDDL] {furniture}: no connections found — skipping validation.")
        return True

    domain_str = generate_pddl_domain()
    problem_str = generate_pddl_problem(parts, connections)

    with tempfile.TemporaryDirectory() as tmpdir:
        domain_file = os.path.join(tmpdir, "domain.pddl")
        problem_file = os.path.join(tmpdir, "problem.pddl")
        with open(domain_file, "w") as f:
            f.write(domain_str)
        with open(problem_file, "w") as f:
            f.write(problem_str)

        if verbose:
            print(f"[PDDL] Validating {furniture} ({len(parts)} parts, "
                  f"{len(connections)} connections) ...")

        plan = _run_pyperplan(domain_file, problem_file)

    if plan is not None:
        if verbose:
            print(f"[PDDL] {furniture}: VALID — plan found ({len(plan)} actions). "
                  "Assembly sequence is logically consistent.")
        return True
    else:
        print(f"[PDDL] {furniture}: INVALID — no valid plan found. "
              "The connection sequence may be unreachable from the initial state.")
        if verbose:
            _diagnose(parts, connections)
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Validate actions.json logical consistency via PDDL + PyPerPlan."
    )
    p.add_argument("--actions", required=True,
                   help="Path to actions.json produced by Stage 3")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress detailed output")
    args = p.parse_args()

    ok = validate_actions_json(args.actions, verbose=not args.quiet)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
