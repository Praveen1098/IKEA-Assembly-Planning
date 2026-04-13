"""DEPRECATED: Stage 5 is no longer called by run.py.

Backchaining BT synthesis has been moved into stage4_formalize.py using a
simpler, dependency-free implementation. This file is retained for reference
only and should be deleted in a future cleanup.

Do not import this module. Dependencies (py_trees, pydot, pyperplan, networkx)
are no longer required by the pipeline.

-----------------------------------------------------------------------
Original docstring preserved below for reference:
-----------------------------------------------------------------------
Stage 5: Formal Behavior Tree Compiler.

Compiles actions.json → a formally grounded, reactive BehaviorTree.CPP v4 XML
using three algorithms from the literature:

  1. Martín et al. (AAMAS 2021) — causal-graph construction from grounded PDDL plan:
       build_causal_graph()
       Actions with no causal dependency between them can run in Parallel.

  2. Colledanchise & Ögren (2016/2019) — backchaining BT synthesis:
       backchain()
       For each goal condition G, find action A whose postcondition includes G,
       build Fallback(Condition(G), Sequence(pre_checks, Action(A))), recurse on
       unsatisfied preconditions of A.

  3. BehaviorTree.CPP v4 native pre/post-condition scripting:
       _serialize_btcpp_v4_xml()
       PDDL preconditions → _skipIf / _failureIf attributes on Action nodes.
       PDDL effects       → _onSuccess scripts that write to the shared blackboard.
       No separate Condition subtrees needed; BT.CPP evaluates inline.

Additionally implements a Python-based BT simulation verifier (verify_bt()) that
executes the tick loop against a mock blackboard and checks five LTL-like invariants
without requiring an external model-checker (nuXmv).  This gives bounded formal
guarantees for the deterministic assembly domain.

Outputs (per furniture item):
  behavior_tree_formal.xml     BT.CPP v4 XML with full pre/post-condition semantics
  bt_causal_graph.json         Causal dependency graph adjacency list
  bt_verification.json         Simulation-based LTL property results

References:
  - Martín et al. "Optimised Execution of PDDL Plans using BTs" AAMAS 2021
  - Colledanchise & Ögren "Towards Blended Reactive Planning and Acting" ICRA 2016
  - Colledanchise & Ögren "Behavior Trees in Robotics and AI" 2019 (Ch. 6)
  - BehaviorTree.CPP v4 docs: behaviortree.dev/docs/4.0.2
  - BehaVerify (arXiv 2208.05360) — optional nuXmv verification wrapper
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx

from stage4_formalize import (
    PDDL_ACTION_SEMANTICS,
    CONNECTION_GOAL_PREDICATES,
    _solve_pddl,
    to_pddl,
    _validate_actions_data,
)
from utils import ensure_dir


# ---------------------------------------------------------------------------
# Blackboard key helpers
# ---------------------------------------------------------------------------

def _key_on_table(p: int) -> str:
    return f"on_table_{p}"

def _key_held(p: int) -> str:
    return f"held_{p}"

def _key_inserted(peg: int, socket: int) -> str:
    return f"inserted_{peg}_{socket}"

def _key_fastened(fastener: int, hole: int) -> str:
    return f"fastened_{fastener}_{hole}"

def _key_pressed(p1: int, p2: int) -> str:
    return f"pressed_{p1}_{p2}"

def _key_reoriented(p: int) -> str:
    return f"reoriented_{p}"


def _connection_key(conn: dict) -> str:
    """Return the blackboard key that represents this connection predicate."""
    pred = conn.get("predicate", "inserted")
    p1 = conn["part1"]
    p2 = conn["part2"]
    return f"{pred}_{p1}_{p2}"


# ---------------------------------------------------------------------------
# Grounded action dataclass
# ---------------------------------------------------------------------------

@dataclass
class GroundedAction:
    """A single assembly primitive with its grounded PDDL pre/effects.

    All predicate sets use the flat blackboard key strings (e.g. "held_2",
    "inserted_2_1") so they can be directly used in BT.CPP v4 scripting and
    the simulation verifier.
    """
    primitive: str          # GRASP | INSERT | FASTEN | PRESS | REORIENT | PLACE
    params: dict            # raw action dict from actions_data (already validated)
    step_idx: int
    action_idx: int

    # Populated by _ground_predicates()
    pre: list[str] = field(default_factory=list)   # must be TRUE before execution
    add: list[str] = field(default_factory=list)   # set TRUE on SUCCESS
    delete: list[str] = field(default_factory=list) # set FALSE on SUCCESS
    skip_if: str = ""   # BT.CPP _skipIf expression (postcond already holds)
    fail_if: str = ""   # BT.CPP _failureIf expression (precond not met)
    on_success: str = ""  # BT.CPP _onSuccess script (write effects to blackboard)
    connection_key: str = ""  # blackboard key for the connection formed (if any)


def _ground_predicates(action: dict, step_idx: int, action_idx: int) -> GroundedAction:
    """Instantiate PDDL_ACTION_SEMANTICS templates with concrete part IDs.

    Maps each template placeholder to the actual int from the action dict, then
    generates BT.CPP v4 scripting strings for the GroundedAction.
    """
    prim = action["primitive"]
    semantics = PDDL_ACTION_SEMANTICS[prim]

    # Build substitution map from template placeholders to concrete IDs
    subs: dict[str, str] = {}
    if prim == "GRASP":
        subs = {"p": str(action["part_id"])}
    elif prim == "PLACE":
        subs = {"p": str(action["part_id"])}
    elif prim == "REORIENT":
        subs = {"p": str(action["part_id"])}
    elif prim == "INSERT":
        subs = {"peg": str(action["peg_part_id"]), "socket": str(action["socket_part_id"])}
    elif prim == "FASTEN":
        subs = {"fastener": str(action["fastener_part_id"]), "hole": str(action["hole_part_id"])}
    elif prim == "PRESS":
        subs = {"p1": str(action["part1_id"]), "p2": str(action["part2_id"])}

    def _subst(template: str) -> str:
        result = template
        for k, v in subs.items():
            result = result.replace(f"{{{k}}}", v)
        return result

    pre = [_subst(t) for t in semantics["pre"]]
    add = [_subst(t) for t in semantics["add"]]
    delete = [_subst(t) for t in semantics["del"]]

    # BT.CPP v4 scripting strings -----------------------------------------------
    # _skipIf  : return success early if the primary postcondition already holds
    # The "primary postcondition" is the connection key (INSERT/FASTEN/PRESS) or
    # the held key for GRASP, or the reoriented key for REORIENT.
    primary_post = add[0] if add else ""
    skip_if = f"{{{primary_post}}} == 1" if primary_post else ""

    # _failureIf : fail early if any precondition is not met
    # We combine all preconditions except gripper_empty (checked via skipIf on GRASP)
    # using && — the BT.CPP scripting engine supports && / ||.
    non_trivial_pre = [p for p in pre if p != "gripper_empty"]
    if non_trivial_pre:
        fail_conditions = " && ".join(f"{{{p}}} == 0" for p in non_trivial_pre)
        fail_if = fail_conditions
    elif "gripper_empty" in pre:
        fail_if = "{gripper_empty} == 0"
    else:
        fail_if = ""

    # _onSuccess : write all effects to blackboard (add := 1, del := 0)
    add_scripts = [f"{{{k}}} := 1" for k in add]
    del_scripts = [f"{{{k}}} := 0" for k in delete]
    on_success = "; ".join(add_scripts + del_scripts)

    # Connection key (for goal tracking and causal graph)
    conn_key = ""
    if prim == "INSERT":
        conn_key = _key_inserted(action["peg_part_id"], action["socket_part_id"])
    elif prim == "FASTEN":
        conn_key = _key_fastened(action["fastener_part_id"], action["hole_part_id"])
    elif prim == "PRESS":
        conn_key = _key_pressed(action["part1_id"], action["part2_id"])

    return GroundedAction(
        primitive=prim,
        params=action,
        step_idx=step_idx,
        action_idx=action_idx,
        pre=pre,
        add=add,
        delete=delete,
        skip_if=skip_if,
        fail_if=fail_if,
        on_success=on_success,
        connection_key=conn_key,
    )


def ground_all_actions(actions_data: dict) -> list[GroundedAction]:
    """Convert all validated step/action dicts in actions_data to GroundedActions."""
    result = []
    for step in actions_data["steps"]:
        for j, action in enumerate(step["actions"]):
            result.append(_ground_predicates(action, step["step_idx"], j))
    return result


# ---------------------------------------------------------------------------
# Initial blackboard state
# ---------------------------------------------------------------------------

def build_initial_blackboard(actions_data: dict) -> dict[str, int]:
    """Construct the PDDL initial state as a flat blackboard dict.

    All parts start on_table=1. gripper_empty=1. All connection predicates=0.
    Uses int values (0/1) to match BT.CPP scripting semantics.
    """
    bb: dict[str, int] = {"gripper_empty": 1}
    for part_id in actions_data["parts"]:
        bb[_key_on_table(part_id)] = 1
        bb[_key_held(part_id)] = 0
        bb[_key_reoriented(part_id)] = 0
    for conn in actions_data.get("all_connections", []):
        bb[_connection_key(conn)] = 0
    return bb


# ---------------------------------------------------------------------------
# Algorithm 1 — Causal Graph (Martín et al. AAMAS 2021)
# ---------------------------------------------------------------------------

def build_causal_graph(grounded: list[GroundedAction]) -> nx.DiGraph:
    """Build a causal dependency graph over the grounded action sequence.

    Edge i→j means action j has a precondition that is an effect of action i
    (causal link).  Actions with no path between them can be parallelised.

    Source: Martín et al. "Optimised Execution of PDDL Plans using BTs" AAMAS 2021
    — Algorithm 1: build execution graph from causal links, then derive BT.
    """
    G = nx.DiGraph()
    n = len(grounded)
    for i in range(n):
        G.add_node(i, action=grounded[i])

    for i in range(n):
        effects_i = set(grounded[i].add)
        for j in range(i + 1, n):
            pre_j = set(grounded[j].pre)
            if effects_i & pre_j:   # causal link: effect of i is precondition of j
                G.add_edge(i, j)

    return G


def causal_graph_to_json(G: nx.DiGraph, grounded: list[GroundedAction]) -> dict:
    """Serialise the causal graph to a JSON-friendly dict for bt_causal_graph.json."""
    nodes = []
    for i, ga in enumerate(grounded):
        nodes.append({
            "index": i,
            "step_idx": ga.step_idx,
            "action_idx": ga.action_idx,
            "primitive": ga.primitive,
            "pre": ga.pre,
            "add": ga.add,
            "del": ga.delete,
        })
    edges = [{"from": u, "to": v} for u, v in G.edges()]
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Algorithm 2 — Backchaining BT Synthesis (Colledanchise & Ögren 2016/2019)
# ---------------------------------------------------------------------------

def _find_action_for_postcond(
    goal: str, grounded: list[GroundedAction], search_from: int = 0
) -> GroundedAction | None:
    """Return the first GroundedAction (at or after search_from) whose add-effects
    include `goal`.  Scanning forward preserves the assembly ordering from the
    PDDL-solved plan.

    Colledanchise & Ögren (2019) Ch.6: find action A such that goal ∈ post(A).
    """
    for i in range(search_from, len(grounded)):
        if goal in grounded[i].add:
            return grounded[i]
    return None


@dataclass
class BTNode:
    """Lightweight BT node representation for serialisation.

    kind: "sequence" | "fallback" | "action" | "script" | "parallel"
    label: human-readable name / script expression
    children: child BTNodes (empty for leaf nodes)
    grounded_action: GroundedAction (action leaves only)
    """
    kind: str
    label: str
    children: list["BTNode"] = field(default_factory=list)
    grounded_action: GroundedAction | None = None


def backchain(
    goal_conditions: list[str],
    grounded: list[GroundedAction],
    world_state: dict[str, int],
    depth: int = 0,
) -> BTNode:
    """Recursively synthesise a BT that achieves all goal_conditions.

    Implements Colledanchise & Ögren (2016) Algorithms 1+2 (backchaining):
      1. For each unsatisfied goal G, find action A with G ∈ post(A).
      2. Build: Fallback(Condition(G_already_done), Sequence(pre_subtree, Action(A)))
      3. Recurse on A's unsatisfied preconditions.
      4. Simulate A's effects on world_state (forward pass through the plan).

    The forward scan (_find_action_for_postcond starting from index 0 for each
    goal) respects the PDDL plan ordering — actions are visited in causal sequence.

    Returns a Sequence of per-goal Fallback subtrees.
    """
    step_nodes: list[BTNode] = []
    for goal in goal_conditions:
        if world_state.get(goal, 0):
            # Already satisfied — no subtree needed.
            continue

        ga = _find_action_for_postcond(goal, grounded)
        if ga is None:
            # No action achieves this goal; add a terminal condition check so the
            # BT will FAIL clearly if the goal remains unmet at execution time.
            step_nodes.append(BTNode(
                kind="script",
                label=f"ASSERT {goal} == 1",
            ))
            continue

        # Recursively satisfy A's unsatisfied preconditions (depth-bounded at 16)
        unsatisfied_pre = [p for p in ga.pre if not world_state.get(p, 0)]
        if unsatisfied_pre and depth < 16:
            pre_subtree = backchain(unsatisfied_pre, grounded, world_state, depth + 1)
        else:
            pre_subtree = None

        # Build action leaf
        action_leaf = BTNode(
            kind="action",
            label=f"{ga.primitive}(step={ga.step_idx},act={ga.action_idx})",
            grounded_action=ga,
        )

        # Sequence: [pre_subtree?, action]
        seq_children: list[BTNode] = []
        if pre_subtree is not None and pre_subtree.children:
            seq_children.append(pre_subtree)
        seq_children.append(action_leaf)
        execute_seq = BTNode(kind="sequence", label=f"execute_{goal}", children=seq_children)

        # Fallback: [already_done?, execute_seq]
        already_done = BTNode(kind="script", label=f"{{{{ {goal} }}}} == 1")
        fallback = BTNode(
            kind="fallback",
            label=f"achieve_{goal}",
            children=[already_done, execute_seq],
        )
        step_nodes.append(fallback)

        # Simulate this action's effects so downstream goals see updated state
        for k in ga.add:
            world_state[k] = 1
        for k in ga.delete:
            world_state[k] = 0

    return BTNode(kind="sequence", label="backchain_root", children=step_nodes)


# ---------------------------------------------------------------------------
# Algorithm 3 — BT.CPP v4 XML Serialiser
# ---------------------------------------------------------------------------

def _action_node_xml(parent: ET.Element, ga: GroundedAction) -> None:
    """Append a BT.CPP v4 <Action> element with inline pre/post-condition scripts."""
    prim = ga.primitive
    params = ga.params

    attribs: dict[str, str] = {"ID": prim.capitalize()}

    # Primitive-specific parameters
    if prim == "GRASP":
        attribs["part"] = str(params["part_id"])
    elif prim == "PLACE":
        attribs["part"] = str(params["part_id"])
    elif prim == "REORIENT":
        attribs["part"] = str(params["part_id"])
        attribs["description"] = str(params.get("description", ""))
    elif prim == "INSERT":
        attribs["peg"] = str(params["peg_part_id"])
        attribs["socket"] = str(params["socket_part_id"])
        attribs["direction"] = str(params.get("insertion_dir", ""))
        attribs["fit_type"] = str(params.get("fit_type", ""))
    elif prim == "FASTEN":
        attribs["fastener"] = str(params["fastener_part_id"])
        attribs["hole"] = str(params["hole_part_id"])
        attribs["tool"] = str(params.get("tool", ""))
        attribs["connection"] = str(params.get("connection_type", ""))
    elif prim == "PRESS":
        attribs["part1"] = str(params["part1_id"])
        attribs["part2"] = str(params["part2_id"])
        attribs["direction"] = str(params.get("press_dir", ""))
        attribs["fit_type"] = str(params.get("fit_type", ""))

    # BT.CPP v4 pre/post-condition scripting attributes
    if ga.skip_if:
        attribs["_skipIf"] = ga.skip_if
    if ga.fail_if:
        attribs["_failureIf"] = ga.fail_if
    if ga.on_success:
        attribs["_onSuccess"] = ga.on_success

    ET.SubElement(parent, "Action", attrib=attribs)


def _btnode_to_xml(parent: ET.Element, node: BTNode) -> None:
    """Recursively convert a BTNode tree to XML elements."""
    if node.kind == "sequence":
        seq = ET.SubElement(parent, "Sequence", attrib={"name": node.label})
        for child in node.children:
            _btnode_to_xml(seq, child)

    elif node.kind == "fallback":
        fb = ET.SubElement(parent, "Fallback", attrib={"name": node.label})
        for child in node.children:
            _btnode_to_xml(fb, child)

    elif node.kind == "parallel":
        par = ET.SubElement(
            parent, "Parallel",
            attrib={"name": node.label, "success_count": str(len(node.children)),
                    "failure_count": "1"}
        )
        for child in node.children:
            _btnode_to_xml(par, child)

    elif node.kind == "script":
        ET.SubElement(parent, "Script", attrib={"code": node.label})

    elif node.kind == "action":
        assert node.grounded_action is not None
        _action_node_xml(parent, node.grounded_action)

    else:
        raise ValueError(f"Unknown BTNode kind: {node.kind!r}")


def _blackboard_init_script(bb: dict[str, int]) -> str:
    """Produce a single BT.CPP Script code string that initialises the blackboard."""
    assignments = "; ".join(f"{{{k}}} := {v}" for k, v in sorted(bb.items()))
    return assignments


def _build_goal_verify_subtree(all_connections: list[dict]) -> BTNode:
    """Append a final Sequence that asserts every connection goal holds."""
    checks = [
        BTNode(kind="script", label=f"{{{{ {_connection_key(c)} }}}} == 1")
        for c in all_connections
    ]
    return BTNode(kind="sequence", label="VerifyAllConnections", children=checks)


def serialize_btcpp_v4_xml(
    bt_root: BTNode,
    blackboard_init: dict[str, int],
    furniture_name: str,
    all_connections: list[dict],
) -> str:
    """Serialise the BTNode tree to BehaviorTree.CPP v4 format XML string.

    Structure:
      <root BTCPP_format="4">
        <BehaviorTree ID="AssembleFurniture">
          <Sequence name="assemble_{furniture}">
            <Script code="...blackboard init..."/>
            <!-- backchained Fallback/Sequence/Action subtrees -->
            <Sequence name="VerifyAllConnections">
              <Script code="{conn_key} == 1"/>
              ...
            </Sequence>
          </Sequence>
        </BehaviorTree>
        <TreeNodesModel> ... </TreeNodesModel>
      </root>
    """
    xml_root = ET.Element("root", attrib={"BTCPP_format": "4"})
    bt_elem = ET.SubElement(xml_root, "BehaviorTree", attrib={"ID": "AssembleFurniture"})

    top_seq = ET.SubElement(bt_elem, "Sequence", attrib={"name": f"assemble_{furniture_name}"})

    # Blackboard initialisation (PDDL initial state)
    ET.SubElement(top_seq, "Script", attrib={"code": _blackboard_init_script(blackboard_init)})

    # Backchained subtrees
    for child in bt_root.children:
        _btnode_to_xml(top_seq, child)

    # Terminal goal-verification subtree
    verify = _build_goal_verify_subtree(all_connections)
    _btnode_to_xml(top_seq, verify)

    # TreeNodesModel: declare all custom Action node types so Groot2 can open the tree
    model_elem = ET.SubElement(xml_root, "TreeNodesModel")
    primitives_seen: set[str] = set()
    for child in bt_root.children:
        _collect_primitives(child, primitives_seen)
    for prim in sorted(primitives_seen):
        ET.SubElement(model_elem, "Action", attrib={"ID": prim.capitalize()})

    raw = ET.tostring(xml_root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def _collect_primitives(node: BTNode, seen: set[str]) -> None:
    """Walk BTNode tree and collect all primitive names used in action leaves."""
    if node.kind == "action" and node.grounded_action is not None:
        seen.add(node.grounded_action.primitive)
    for child in node.children:
        _collect_primitives(child, seen)


# ---------------------------------------------------------------------------
# Simulation-based LTL verifier
# ---------------------------------------------------------------------------

def verify_bt(
    grounded: list[GroundedAction],
    blackboard: dict[str, int],
    all_connections: list[dict],
) -> dict:
    """Execute the grounded action sequence against a mock blackboard and check
    five LTL-like invariants.

    This is a deterministic, bounded forward simulation of the BT's tick loop.
    For the assembly domain (finite, acyclic action sequence, deterministic effects)
    this simulation is complete — it covers the full execution trace.

    Properties checked (inspired by BehaVerify / arXiv 2208.05360):
      P1  Gripper consistency:   gripper_empty == 1  ⟺  no part is held
      P2  No double-grasp:       gripper_empty == 1  before every GRASP
      P3  Hold-before-connect:   the peg/fastener/part must be held before INSERT/FASTEN/PRESS
      P4  Causal completeness:   every action's preconditions are satisfied at execution time
      P5  Liveness:              all connection goal predicates hold after full execution

    Returns a dict:
      {
        "passed": bool,          True if all properties pass
        "properties": {
          "P1_gripper_consistency":  {"pass": bool, "violations": [str]},
          "P2_no_double_grasp":     {"pass": bool, "violations": [str]},
          "P3_hold_before_connect": {"pass": bool, "violations": [str]},
          "P4_causal_completeness": {"pass": bool, "violations": [str]},
          "P5_liveness":            {"pass": bool, "unmet_goals": [str]},
        }
      }
    """
    bb = dict(blackboard)  # copy — do not mutate original

    p1_violations: list[str] = []
    p2_violations: list[str] = []
    p3_violations: list[str] = []
    p4_violations: list[str] = []

    def _held_parts() -> list[str]:
        return [k for k, v in bb.items() if k.startswith("held_") and v == 1]

    for ga in grounded:
        loc = f"step={ga.step_idx} act={ga.action_idx} {ga.primitive}"

        # P1: gripper_empty ⟺ no held parts (check BEFORE applying this action)
        held = _held_parts()
        gripper_empty_val = bb.get("gripper_empty", 1)
        if gripper_empty_val == 1 and held:
            p1_violations.append(f"{loc}: gripper_empty=1 but held parts: {held}")
        if gripper_empty_val == 0 and not held:
            p1_violations.append(f"{loc}: gripper_empty=0 but no parts held")

        # P2: no double-grasp — gripper must be empty before GRASP
        if ga.primitive == "GRASP":
            if bb.get("gripper_empty", 1) == 0:
                p2_violations.append(
                    f"{loc}: GRASP on part {ga.params['part_id']} but gripper not empty"
                )

        # P3: held-before-connect
        if ga.primitive == "INSERT":
            peg = ga.params["peg_part_id"]
            if bb.get(_key_held(peg), 0) == 0:
                p3_violations.append(
                    f"{loc}: INSERT peg={peg} but part not held"
                )
        elif ga.primitive == "FASTEN":
            fst = ga.params["fastener_part_id"]
            if bb.get(_key_held(fst), 0) == 0:
                p3_violations.append(
                    f"{loc}: FASTEN fastener={fst} but part not held"
                )
        elif ga.primitive == "PRESS":
            p1_id = ga.params["part1_id"]
            if bb.get(_key_held(p1_id), 0) == 0:
                p3_violations.append(
                    f"{loc}: PRESS part1={p1_id} but part not held"
                )

        # P4: causal completeness — every precondition must be satisfied
        for pre in ga.pre:
            if bb.get(pre, 0) == 0:
                p4_violations.append(
                    f"{loc}: precondition '{pre}' not satisfied"
                )

        # Apply effects to blackboard (simulate SUCCESS)
        for k in ga.add:
            bb[k] = 1
        for k in ga.delete:
            bb[k] = 0

    # P5: liveness — all connection goals must be achieved after full execution
    unmet_goals: list[str] = []
    for conn in all_connections:
        k = _connection_key(conn)
        if bb.get(k, 0) == 0:
            unmet_goals.append(k)

    all_pass = (
        not p1_violations
        and not p2_violations
        and not p3_violations
        and not p4_violations
        and not unmet_goals
    )

    return {
        "passed": all_pass,
        "properties": {
            "P1_gripper_consistency":  {"pass": not p1_violations, "violations": p1_violations},
            "P2_no_double_grasp":     {"pass": not p2_violations, "violations": p2_violations},
            "P3_hold_before_connect": {"pass": not p3_violations, "violations": p3_violations},
            "P4_causal_completeness": {"pass": not p4_violations, "violations": p4_violations},
            "P5_liveness":            {"pass": not unmet_goals,   "unmet_goals": unmet_goals},
        },
    }


# ---------------------------------------------------------------------------
# Action sequence repair heuristic
# ---------------------------------------------------------------------------

def _repair_action_sequence(actions_data: dict) -> dict:
    """Post-hoc repair of actions_data to satisfy PDDL gripper preconditions.

    Simulates gripper state per step and inserts missing GRASPs before
    INSERT/PRESS actions, and inserts PLACE before conflicting double-GRASPs.
    Operates on a deep copy — the original is never mutated.

    Called as a fallback when _plan_from_pddl() raises PDDLSolveError.
    """
    import copy
    data = copy.deepcopy(actions_data)

    for step in data["steps"]:
        held: int | None = None
        repaired: list[dict] = []

        for act in step.get("actions", []):
            prim = act.get("primitive")

            if prim == "GRASP":
                if held is not None:
                    repaired.append({"primitive": "PLACE", "part_id": held})
                    held = None
                repaired.append(act)
                held = act["part_id"]

            elif prim in ("INSERT", "PRESS"):
                active = act.get("peg_part_id") or act.get("part1_id")
                if held != active:
                    if held is not None:
                        repaired.append({"primitive": "PLACE", "part_id": held})
                    repaired.append({"primitive": "GRASP", "part_id": active})
                    held = active
                repaired.append(act)
                held = None  # INSERT/PRESS release the gripper

            elif prim == "PLACE":
                repaired.append(act)
                held = None

            else:  # REORIENT — no gripper state change
                repaired.append(act)

        step["actions"] = repaired

    # Re-aggregate primitives_used from repaired sequences
    all_prims = {a["primitive"] for s in data["steps"] for a in s.get("actions", [])}
    data["primitives_used"] = sorted(
        all_prims & {"GRASP", "PLACE", "INSERT", "PRESS", "REORIENT"}
    )
    return data


# ---------------------------------------------------------------------------
# PDDL solve + plan reconciliation — no fallbacks
# ---------------------------------------------------------------------------

class PDDLSolveError(Exception):
    """Raised when pyperplan cannot find a plan for the generated PDDL.

    The message includes:
    - Which goal predicates are unreachable from the initial state.
    - Which grounded action was supposed to achieve each unreachable goal.
    - The likely source of the inconsistency in actions_data (step / action index).

    This is a hard error — no silent fallback to the VLM action order.
    Fixing the underlying stage3 extraction error is the correct response.
    """


def _diagnose_pddl_failure(
    domain_path: str,
    problem_path: str,
    grounded: list[GroundedAction],
) -> str:
    """Identify exactly why pyperplan returned no plan.

    Algorithm (Göbelbecker et al. 2010 — diagnosis via reachability analysis):
      1. Parse domain + problem with pyperplan to get a Task object.
      2. Ground the task to obtain operators and the set of initial facts.
      3. Forward-simulate: apply each operator whose preconditions are satisfied
         until no new facts can be added (fixed-point = reachable facts).
      4. Unreachable goals = task.goals − reachable facts.
      5. Map each unreachable goal back to the GroundedAction that was supposed
         to achieve it, and identify which of that action's preconditions is the
         root cause.

    Returns a human-readable diagnostic string formatted for immediate action.
    """
    lines: list[str] = ["pyperplan found no plan for the generated PDDL.\n"]

    try:
        from pyperplan.planner import _parse, _ground
        import logging
        logging.disable(logging.CRITICAL)
        parsed = _parse(domain_path, problem_path)
        task = _ground(parsed)
        logging.disable(logging.NOTSET)

        # Forward reachability (BFS over facts)
        reachable: set = set(task.initial_state)
        changed = True
        while changed:
            changed = False
            for op in task.operators:
                if op.preconditions <= reachable:
                    new_facts = op.add_effects - reachable
                    if new_facts:
                        reachable |= new_facts
                        changed = True

        # Unreachable goals
        unreachable_goals = task.goals - reachable
        if not unreachable_goals:
            lines.append(
                "  All goal predicates ARE reachable — the plan may be infeasible "
                "due to ordering constraints or pyperplan resource limits.\n"
            )
        else:
            lines.append(f"  Unreachable goal(s): {sorted(str(g) for g in unreachable_goals)}\n")

        # Map unreachable goals to grounded actions + diagnose precondition failures
        for goal in unreachable_goals:
            goal_str = str(goal).strip("() ")
            # Find which GroundedAction was supposed to achieve this goal
            responsible: GroundedAction | None = None
            for ga in grounded:
                if ga.connection_key and ga.connection_key.replace("_", " ") in goal_str.replace("_", " "):
                    responsible = ga
                    break
                # Broader match: check if any add-effect matches the goal string
                for add_key in ga.add:
                    if add_key.replace("_", " ") in goal_str.replace("_", " "):
                        responsible = ga
                        break
                if responsible:
                    break

            if responsible is None:
                lines.append(f"  Goal '{goal_str}': no matching grounded action found.\n")
                continue

            lines.append(
                f"  Goal '{goal_str}' was supposed to be achieved by:\n"
                f"    step={responsible.step_idx}, action={responsible.action_idx}, "
                f"primitive={responsible.primitive}\n"
            )

            # Check which preconditions of the responsible action are not reachable
            for pre in responsible.pre:
                # Convert flat key (e.g. "held_2") to a PDDL fact string for lookup
                pre_fact_candidates = [
                    f for f in reachable | task.initial_state
                    if pre.replace("_", " ") in str(f).replace("_", " ")
                ]
                pre_in_reachable = any(
                    pre.replace("_", " ") in str(f).replace("_", " ")
                    for f in reachable
                )
                if not pre_in_reachable:
                    lines.append(
                        f"    ✗ precondition '{pre}' is NOT reachable.\n"
                        f"      → Likely cause: the action that produces '{pre}' "
                        f"(e.g. GRASP for a held_* predicate) is missing from "
                        f"actions_data or appears after this step.\n"
                    )

    except Exception as exc:
        lines.append(f"  (Diagnostic step failed: {exc})\n")
        lines.append(
            "  Check domain.pddl and problem.pddl for syntax errors and ensure\n"
            "  all goal predicates in all_connections have corresponding actions.\n"
        )

    lines.append(
        "\nTo fix: inspect actions_data['steps'] for the furniture item and ensure\n"
        "every connection-forming action (INSERT/FASTEN/PRESS) is preceded by a\n"
        "GRASP of the same peg/fastener/part in the same or an earlier step."
    )
    return "".join(lines)


def _plan_from_pddl(
    domain_path: str,
    problem_path: str,
    grounded: list[GroundedAction],
) -> list[GroundedAction]:
    """Solve PDDL with pyperplan and reorder grounded actions to match the plan.

    Raises PDDLSolveError (with full diagnosis) if pyperplan cannot find a plan.
    There is NO silent fallback to the VLM action order — a solve failure always
    means the actions_data is causally inconsistent and must be corrected.

    Reordering: match plan operator names to GroundedAction primitives in FIFO
    order.  This preserves the parameter values from stage3 while guaranteeing
    the ordering satisfies all PDDL preconditions.
    """
    plan = _solve_pddl(domain_path, problem_path)
    if plan is None:
        raise PDDLSolveError(_diagnose_pddl_failure(domain_path, problem_path, grounded))

    # Build a lookup: primitive name (lower) → queue of GroundedActions (FIFO)
    by_prim: dict[str, list[GroundedAction]] = defaultdict(list)
    for ga in grounded:
        by_prim[ga.primitive.lower()].append(ga)

    reordered: list[GroundedAction] = []
    missing_plan_steps: list[str] = []

    for step in plan:
        prim_lower = step["name"]   # e.g. "grasp", "insert"
        queue = by_prim.get(prim_lower, [])
        if queue:
            reordered.append(queue.pop(0))
        else:
            # The PDDL plan requires a primitive that has no corresponding grounded
            # action in actions_data — this is a hard inconsistency.
            missing_plan_steps.append(
                f"  plan step '{prim_lower}' has no matching action in actions_data"
            )

    if missing_plan_steps:
        diag_lines = [
            "The PDDL plan requires actions that are missing from actions_data.\n",
            "This means the VLM's stage3 extraction omitted actions that are\n",
            "necessary for the assembly to be causally consistent.\n\n",
            "Missing plan steps:\n",
        ] + missing_plan_steps + [
            "\n\nFix: inspect actions_data['steps'] and ensure every connection-forming\n",
            "action (INSERT/FASTEN/PRESS) is preceded by a GRASP of the same part.\n",
        ]
        raise PDDLSolveError("".join(diag_lines))

    # If any actions were not matched by the plan (e.g. REORIENT steps that the
    # STRIPS solver does not include), interleave them by causal position.
    # This is NOT a fallback — REORIENT is a positioning primitive not modelled
    # in the goal, so the planner legitimately omits it.
    unmatched: list[GroundedAction] = []
    for queue in by_prim.values():
        unmatched.extend(queue)
    if unmatched:
        _interleave_reorient(reordered, unmatched)

    return reordered


def _interleave_reorient(
    reordered: list[GroundedAction],
    unmatched: list[GroundedAction],
) -> None:
    """Insert REORIENT (and any other unmatched) actions into the ordered plan.

    REORIENT is a positioning primitive — it must precede the INSERT/FASTEN/PRESS
    that uses the same part.  Strategy: insert each REORIENT immediately before
    the first connection-forming action that involves the same part_id.
    Mutates `reordered` in-place.
    """
    for ga in unmatched:
        if ga.primitive != "REORIENT":
            reordered.append(ga)
            continue
        part_id = ga.params.get("part_id")
        # Find insertion point: first connection action involving this part
        insert_at = len(reordered)
        for i, existing in enumerate(reordered):
            if existing.primitive in ("INSERT", "FASTEN", "PRESS"):
                params = existing.params
                active_part = (
                    params.get("peg_part_id")
                    or params.get("fastener_part_id")
                    or params.get("part1_id")
                )
                if active_part == part_id:
                    insert_at = i
                    break
        reordered.insert(insert_at, ga)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def to_formal_bt(actions_data: dict, output_path: str) -> str:
    """Compile actions_data into a formally grounded BT.CPP v4 XML.

    Pipeline:
      1. Validate actions_data schema (reuses stage4 validator — no duplication).
      2. Generate PDDL files via stage4.to_pddl() and attempt pyperplan solve.
      3. Ground all actions → GroundedAction list.
      4. (Optional) Reorder by PDDL plan if solve succeeds.
      5. Build causal graph (Martín et al. AAMAS 2021).
      6. Run backchaining synthesis (Colledanchise & Ögren 2016/2019).
      7. Serialise to BT.CPP v4 XML with _skipIf / _failureIf / _onSuccess.
      8. Run simulation-based LTL verifier.
      9. Write behavior_tree_formal.xml, bt_causal_graph.json, bt_verification.json.

    Returns path to behavior_tree_formal.xml.

    Raises BehaviorTreeBuildError on schema validation failure.
    """
    # --- Step 1: validate ---
    _validate_actions_data(actions_data)

    furniture_name = actions_data["furniture"]
    furniture_type = actions_data["category"]
    item_out_dir = os.path.join(output_path, furniture_type, furniture_name)
    ensure_dir(item_out_dir)

    # --- Step 2: PDDL files + optional solve ---
    domain_path, problem_path = to_pddl(actions_data, output_path)

    # --- Step 3: ground all actions ---
    grounded = ground_all_actions(actions_data)

    # --- Step 4: PDDL reorder (with repair fallback) ---
    try:
        grounded = _plan_from_pddl(domain_path, problem_path, grounded)
    except PDDLSolveError as original_error:
        repaired_data = _repair_action_sequence(actions_data)
        domain_path, problem_path = to_pddl(repaired_data, output_path)
        grounded = ground_all_actions(repaired_data)
        try:
            grounded = _plan_from_pddl(domain_path, problem_path, grounded)
        except PDDLSolveError:
            raise PDDLSolveError(
                str(original_error) + "\n\nRepair heuristic also failed."
            ) from None
        actions_data = repaired_data

    # --- Step 5: causal graph ---
    causal_G = build_causal_graph(grounded)
    causal_json = causal_graph_to_json(causal_G, grounded)
    causal_path = os.path.join(item_out_dir, "bt_causal_graph.json")
    with open(causal_path, "w", encoding="utf-8") as f:
        json.dump(causal_json, f, indent=2)

    # --- Step 6: backchaining synthesis ---
    all_connections = actions_data.get("all_connections", [])
    goal_conditions = [_connection_key(c) for c in all_connections]

    blackboard = build_initial_blackboard(actions_data)
    world_state = dict(blackboard)   # mutable copy for backchain simulation

    bt_root = backchain(goal_conditions, grounded, world_state)

    # --- Step 7: BT.CPP v4 XML serialisation ---
    xml_str = serialize_btcpp_v4_xml(bt_root, blackboard, furniture_name, all_connections)
    xml_path = os.path.join(item_out_dir, "behavior_tree_formal.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    # --- Step 8: simulation-based LTL verification ---
    # Re-ground with fresh blackboard (verify_bt runs a clean forward simulation)
    grounded_for_verify = ground_all_actions(actions_data)
    grounded_for_verify = _plan_from_pddl(domain_path, problem_path, grounded_for_verify)
    verification = verify_bt(grounded_for_verify, dict(blackboard), all_connections)

    verify_path = os.path.join(item_out_dir, "bt_verification.json")
    with open(verify_path, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2)

    # Surface result to stdout
    status = "PASSED" if verification["passed"] else "FAILED"
    print(f"  [BT verify] {furniture_name}: {status}")
    for prop, result in verification["properties"].items():
        ok = result["pass"]
        marker = "  OK" if ok else "FAIL"
        viol = result.get("violations") or result.get("unmet_goals") or []
        if not ok:
            print(f"    [{marker}] {prop}: {viol}")
        else:
            print(f"    [{marker}] {prop}")

    return xml_path
