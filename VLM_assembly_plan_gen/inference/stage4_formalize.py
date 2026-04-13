"""Stage 4: Compile actions.json into an executable Behavior Tree (XML + ASCII).

Generates a goal-oriented backchaining BT (Colledanchise & Ögren 2016/2019) —
one Fallback subtree per assembly connection goal.  Independent connections
(no causal dependency between them) are wrapped in a BT.CPP Parallel node;
dependent connections remain in a Sequence.

BT structure per connection goal:
  Fallback [achieve_{predicate}_{part1}_{part2}]
  ├── Condition: IsConnected(part1, part2, description=<visual cue>)
  └── Sequence [do_{predicate}_{part1}_{part2}]
      ├── Fallback [achieve_held_{active_part}]
      │   ├── Condition: IsHeld(part, description=<visual cue>)
      │   └── Action: Grasp(part)
      ├── Action: Reorient(part, description)  [only if needs_reorient=True]
      └── Action: Insert / Press / Screw (active_part, passive_part, direction)

The `description` attribute on Condition nodes is a natural language visual cue
drawn from Stage 3's VLM reasoning. At runtime a robot VLM can evaluate these
conditions against camera images (VLM-driven BT, arXiv 2501.03968).

Output files (per furniture item):
  behavior_tree.xml   — BehaviorTree.CPP v4 XML (Groot2-compatible)
  behavior_tree.txt   — ASCII tree printed to terminal for quick inspection

No LLM calls, no py_trees, no PDDL, no networkx — pure stdlib only.

Design references:
  - Colledanchise & Ögren (2016/2019) — backchaining synthesis
  - Martín et al., AAMAS 2021 (arXiv 2101.01964) — causal graph → Parallel BT nodes
    Open-source reference: PlanSys2 compute_bt.cpp (github.com/PlanSys2/ros2_planning_system)
  - VLM-driven Behavior Trees (arXiv 2501.03968, Microsoft) — visual condition nodes
  - BTGenBot (arXiv 2403.12761, IROS 2024) — BT.CPP v4 XML format conventions
  - BehaviorTree.CPP v4 — TreeNodesModel port declarations for Groot2
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from utils import ensure_dir


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class BehaviorTreeBuildError(ValueError):
    pass


REQUIRED_ACTION_FIELDS: dict[str, dict[str, type]] = {
    "GRASP":    {"part_id": int},
    "INSERT":   {"active_part": int, "passive_part": int, "direction": str},
    "PRESS":    {"active_part": int, "passive_part": int, "direction": str},
    "SCREW":    {"active_part": int, "passive_part": int, "direction": str},
    "REORIENT": {"part_id": int, "description": str},
}


def _validate_actions_data(actions_data: dict) -> None:
    required_top = {"furniture", "category", "all_connections", "steps"}
    missing = required_top - actions_data.keys()
    if missing:
        raise BehaviorTreeBuildError(f"actions_data missing top-level keys: {missing}")

    if not actions_data["all_connections"]:
        raise BehaviorTreeBuildError("all_connections is empty — no assembly goals to build BT for")

    for step in actions_data.get("steps", []):
        for action in step.get("actions", []):
            prim = action.get("primitive")
            if prim not in REQUIRED_ACTION_FIELDS:
                continue
            for field, ftype in REQUIRED_ACTION_FIELDS[prim].items():
                if field not in action:
                    raise BehaviorTreeBuildError(
                        f"Step {step.get('step_idx')}: {prim} action missing field '{field}'"
                    )
                if not isinstance(action[field], ftype):
                    raise BehaviorTreeBuildError(
                        f"Step {step.get('step_idx')}: {prim}.{field} must be {ftype.__name__}, "
                        f"got {type(action[field]).__name__}"
                    )


# ---------------------------------------------------------------------------
# Causal graph + parallel partitioning (Martín et al., AAMAS 2021)
#
# Algorithm adapted from PlanSys2 compute_bt.cpp:
#   github.com/PlanSys2/ros2_planning_system
#
# Two connections are causally dependent if:
#   Rule 1: C_i.active_part == C_j.passive_part
#           (the part inserted in C_i becomes the mounting base for C_j)
#   Rule 2: C_i.active_part == C_j.active_part
#           (same gripper target — robot must release and re-grasp, so serialize)
#
# Independent connection groups (no edges between them) are candidates for
# a BT.CPP Parallel node; dependent chains stay in a Sequence.
# ---------------------------------------------------------------------------

def _build_causal_graph(connections: list[dict]) -> dict[int, set[int]]:
    """Return adjacency dict: graph[i] = set of indices that must run after i."""
    graph: dict[int, set[int]] = {i: set() for i in range(len(connections))}
    for i, ci in enumerate(connections):
        ai = ci.get("active_part")
        for j, cj in enumerate(connections):
            if i == j:
                continue
            aj = cj.get("active_part")
            pj = cj.get("passive_part")
            # Rule 1: ci's active_part becomes cj's passive_part
            if ai is not None and ai == pj:
                graph[i].add(j)
            # Rule 2: same active_part — serialize (only forward edges to avoid cycles)
            if ai is not None and ai == aj and i < j:
                graph[i].add(j)
    return graph


def _partition_parallel(connections: list[dict], graph: dict[int, set[int]]) \
        -> list[list[int]]:
    """Partition connection indices into groups for Parallel vs Sequence emission.

    Connections in the same group are causally related (Sequence); connections
    in different groups are independent (Parallel).

    Returns list of groups, each group being a topologically-sorted chain.
    """
    num_conn = len(connections)
    # Union-Find: merge any two nodes that share a directed edge
    parent = list(range(num_conn))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(num_conn):
        for j in graph[i]:
            union(i, j)

    # Collect groups by root
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(num_conn):
        groups[find(i)].append(i)

    # Topological sort within each group
    result = []
    for group_nodes in groups.values():
        group_set = set(group_nodes)
        in_degree = {n: 0 for n in group_nodes}
        for n in group_nodes:
            for dep in graph[n]:
                if dep in group_set:
                    in_degree[dep] += 1
        queue = sorted(n for n in group_nodes if in_degree[n] == 0)
        chain: list[int] = []
        while queue:
            node = queue.pop(0)
            chain.append(node)
            for dep in sorted(graph[node]):
                if dep in group_set:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        queue.append(dep)
        result.append(chain)

    # Sort groups by their first element for deterministic XML output
    result.sort(key=lambda g: g[0])
    return result


# ---------------------------------------------------------------------------
# BT XML builder (backchaining pattern)
# ---------------------------------------------------------------------------

def _build_connection_subtree(conn: dict, parent: ET.Element) -> None:
    """Append a single backchaining Fallback subtree for one connection goal."""
    predicate = conn["predicate"]
    p1 = str(conn["part1"])
    p2 = str(conn["part2"])
    active = str(conn.get("active_part", p1))
    passive = str(conn.get("passive_part", p2))
    direction = conn.get("direction", "downward")
    needs_reorient = conn.get("needs_reorient", False)
    reorient_desc = conn.get("reorient_description", "reorient before connecting")
    goal_key = f"{predicate}_{p1}_{p2}"

    # Fallback: skip if goal already achieved
    fallback = ET.SubElement(parent, "Fallback", attrib={"name": f"achieve_{goal_key}"})

    # Condition: IsConnected — VLM-evaluable from camera at runtime
    visual_desc = conn.get(
        "visual_description",
        f"Part {p1} is {predicate} into part {p2} (direction: {direction})",
    )
    ET.SubElement(fallback, "Condition", attrib={
        "ID": "IsConnected",
        "part1": p1,
        "part2": p2,
        "predicate": predicate,
        "description": visual_desc,
    })

    # Sequence: achieve preconditions then execute connection
    do_seq = ET.SubElement(fallback, "Sequence", attrib={"name": f"do_{goal_key}"})

    # Fallback: ensure active part is held
    held_fallback = ET.SubElement(do_seq, "Fallback",
                                  attrib={"name": f"achieve_held_{active}"})
    ET.SubElement(held_fallback, "Condition", attrib={
        "ID": "IsHeld",
        "part": active,
        "description": f"Gripper is holding part {active}",
    })
    ET.SubElement(held_fallback, "Action", attrib={"ID": "Grasp", "part": active})

    # Optional Reorient (only when VLM flagged needs_reorient)
    if needs_reorient:
        ET.SubElement(do_seq, "Action", attrib={
            "ID": "Reorient",
            "part": active,
            "description": reorient_desc,
        })

    # Connection action
    action_id = {"inserted": "Insert", "pressed": "Press", "screwed": "Screw"}.get(predicate)
    if action_id:
        ET.SubElement(do_seq, "Action", attrib={
            "ID": action_id,
            "active_part": active,
            "passive_part": passive,
            "direction": direction,
        })


def _build_backchain_bt_xml(actions_data: dict) -> str:
    """Build BehaviorTree.CPP v4 XML using goal-oriented backchaining.

    Independent connections (detected via causal graph) are wrapped in a
    Parallel node; causally-chained connections remain in a Sequence.

    Causal graph algorithm: Martín et al., AAMAS 2021 (arXiv 2101.01964).
    Open-source reference: PlanSys2 compute_bt.cpp.
    """
    furniture = actions_data["furniture"]
    all_connections = actions_data["all_connections"]

    root = ET.Element("root", attrib={"BTCPP_format": "4"})
    bt = ET.SubElement(root, "BehaviorTree", attrib={"ID": "AssembleFurniture"})
    main_seq = ET.SubElement(bt, "Sequence", attrib={"name": f"assemble_{furniture}"})

    # Partition connections into causally independent groups
    graph = _build_causal_graph(all_connections)
    groups = _partition_parallel(all_connections, graph)

    if len(groups) == 1:
        # All connections are causally dependent — flat Sequence (common case)
        for idx in groups[0]:
            _build_connection_subtree(all_connections[idx], main_seq)
    else:
        # Multiple independent groups — emit a Parallel node
        # success_count = number of groups (all must succeed)
        par = ET.SubElement(main_seq, "Parallel", attrib={
            "name": f"parallel_assembly_{furniture}",
            "success_count": str(len(groups)),
            "failure_count": "1",
        })
        for group in groups:
            chain_seq = ET.SubElement(par, "Sequence",
                                      attrib={"name": f"chain_{group[0]}"})
            for idx in group:
                _build_connection_subtree(all_connections[idx], chain_seq)

    # TreeNodesModel — required by Groot2 for port declarations
    model = ET.SubElement(root, "TreeNodesModel")

    def _cond(parent, node_id, *ports):
        node = ET.SubElement(parent, "Condition", attrib={"ID": node_id})
        for p in ports:
            ET.SubElement(node, "input_port", attrib={"name": p})

    def _action(parent, node_id, *ports):
        node = ET.SubElement(parent, "Action", attrib={"ID": node_id})
        for p in ports:
            ET.SubElement(node, "input_port", attrib={"name": p})

    _cond(model, "IsConnected", "part1", "part2", "predicate", "description")
    _cond(model, "IsHeld", "part", "description")
    _action(model, "Grasp", "part")
    _action(model, "Insert", "active_part", "passive_part", "direction")
    _action(model, "Press", "active_part", "passive_part", "direction")
    _action(model, "Screw", "active_part", "passive_part", "direction")
    _action(model, "Reorient", "part", "description")

    raw = ET.tostring(root, encoding="unicode")
    reparsed = minidom.parseString(raw)
    return reparsed.toprettyxml(indent="  ")


# ---------------------------------------------------------------------------
# ASCII tree renderer
# ---------------------------------------------------------------------------

def _render_conn_ascii(conn: dict, lines: list[str], indent: str, is_last: bool) -> None:
    """Append ASCII lines for a single backchaining Fallback subtree."""
    predicate = conn["predicate"]
    p1 = conn["part1"]
    p2 = conn["part2"]
    active = conn.get("active_part", p1)
    passive = conn.get("passive_part", p2)
    direction = conn.get("direction", "downward")
    needs_reorient = conn.get("needs_reorient", False)
    reorient_desc = conn.get("reorient_description", "reorient before connecting")
    goal_key = f"{predicate}_{p1}_{p2}"

    pfx = indent + ("└── " if is_last else "├── ")
    bar = indent + ("    " if is_last else "│   ")

    lines.append(pfx + f"Fallback: achieve_{goal_key}")
    lines.append(bar + f"├── Condition: IsConnected(part1={p1}, part2={p2}, predicate={predicate})")
    lines.append(bar + f"└── Sequence: do_{goal_key}")

    do_bar = bar + "    "
    lines.append(do_bar + f"├── Fallback: achieve_held_{active}")
    held_bar = do_bar + "│   "
    lines.append(held_bar + f"├── Condition: IsHeld(part={active})")
    lines.append(held_bar + f"└── Action: Grasp(part={active})")

    if needs_reorient:
        lines.append(do_bar + f"├── Action: Reorient(part={active}, \"{reorient_desc}\")")

    action_name = {"inserted": "Insert", "pressed": "Press", "screwed": "Screw"}.get(predicate, predicate.capitalize())
    lines.append(do_bar + f"└── Action: {action_name}(active_part={active}, passive_part={passive}, direction={direction})")


def _render_ascii(actions_data: dict) -> str:
    """Render the backchaining BT as a human-readable ASCII tree.

    Uses the same causal graph partition as the XML builder, so Parallel nodes
    appear in the ASCII output when independent connection groups exist.
    """
    furniture = actions_data["furniture"]
    all_connections = actions_data["all_connections"]
    lines: list[str] = []

    lines.append(f"AssembleFurniture [{furniture}]")
    lines.append(f"└── Sequence: assemble_{furniture}")

    graph = _build_causal_graph(all_connections)
    groups = _partition_parallel(all_connections, graph)

    base = "    "  # indent under top-level Sequence
    if len(groups) == 1:
        # Flat sequential — iterate connections in topological order
        chain = groups[0]
        for ci, idx in enumerate(chain):
            _render_conn_ascii(all_connections[idx], lines,
                               indent=base, is_last=(ci == len(chain) - 1))
    else:
        # Parallel node wrapping independent chains
        lines.append(base + f"└── Parallel[{len(groups)}]: parallel_assembly_{furniture}")
        par_base = base + "    "
        for gi, group in enumerate(groups):
            is_last_group = gi == len(groups) - 1
            grp_pfx = par_base + ("└── " if is_last_group else "├── ")
            grp_bar = par_base + ("    " if is_last_group else "│   ")
            lines.append(grp_pfx + f"Sequence: chain_{group[0]}")
            for ci, idx in enumerate(group):
                _render_conn_ascii(all_connections[idx], lines,
                                   indent=grp_bar, is_last=(ci == len(group) - 1))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDDL output (domain + problem files)
# ---------------------------------------------------------------------------

def to_pddl(actions_data: dict, output_path: str) -> tuple[str, str]:
    """Write domain.pddl and problem.pddl for the given furniture.

    Delegates PDDL generation to stage3_5_pddl_validate which maintains the
    canonical SKILL_LIBRARY and PDDL domain/problem templates.

    Returns:
        (domain_path, problem_path) — absolute paths to the written files.
    """
    from stage3_5_pddl_validate import generate_pddl_domain, generate_pddl_problem

    furniture      = actions_data["furniture"]
    furniture_type = actions_data["category"]
    parts          = actions_data.get("parts", [])
    connections    = actions_data.get("all_connections", [])

    item_dir = os.path.join(output_path, furniture_type, furniture)
    ensure_dir(item_dir)

    domain_path  = os.path.join(item_dir, "domain.pddl")
    problem_path = os.path.join(item_dir, "problem.pddl")

    with open(domain_path,  "w", encoding="utf-8") as f:
        f.write(generate_pddl_domain())
    with open(problem_path, "w", encoding="utf-8") as f:
        f.write(generate_pddl_problem(parts, connections))

    return domain_path, problem_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def to_behavior_tree(actions_data: dict, output_path: str) -> str:
    """Build and write a backchaining BT from actions_data.

    Args:
        actions_data: dict returned by stage3's extract_actions()
        output_path:  base output directory (outputs/{timestamp}/)

    Returns:
        Absolute path to behavior_tree.xml
    """
    _validate_actions_data(actions_data)

    furniture_name = actions_data["furniture"]
    furniture_type = actions_data["category"]
    item_dir = os.path.join(output_path, furniture_type, furniture_name)
    ensure_dir(item_dir)

    xml_str = _build_backchain_bt_xml(actions_data)
    xml_path = os.path.join(item_dir, "behavior_tree.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    txt_str = _render_ascii(actions_data)
    txt_path = os.path.join(item_dir, "behavior_tree.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_str)

    # Print ASCII to terminal for immediate inspection
    try:
        print(txt_str)
    except UnicodeEncodeError:
        # Windows console fallback: replace box-drawing chars
        print(txt_str.encode("ascii", errors="replace").decode("ascii"))

    return xml_path
