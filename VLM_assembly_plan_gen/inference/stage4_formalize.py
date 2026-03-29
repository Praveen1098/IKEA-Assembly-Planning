"""Stage 4: Formalize actions.json into PDDL or Behavior Tree (XML + visualization).

Two output modes:
  pddl — generates domain.pddl + problem.pddl per furniture item.
          domain.pddl is built dynamically from `primitives_used` in actions.json
          using ACTION_TEMPLATES (template-based composition per NL2Plan / SPAR).
          problem.pddl is built from `parts` + `all_connections` — no ground truth.

  bt   — generates three files per furniture item:
          behavior_tree.xml       BehaviorTree.CPP-compatible XML (robot execution)
          behavior_tree_ascii.txt ASCII tree representation (inspection)
          behavior_tree.png       Rendered visualization via py_trees + pydot

Neither output mode calls the LLM — both are deterministic transformations.

Design principles (literature-grounded):
  - VLM-as-formalizer: VLM fills action parameters; code composes structure.
    Source: "Vision Language Models Cannot Plan, but Can They Formalize?" (arXiv 2509.21576)
  - Template-based PDDL composition via ACTION_TEMPLATES dict.
    Source: SPAR (2509.13691), NL2Plan (2405.04215), LLM+P (2304.11477)
  - Sequence-of-Sequences BT structure for sequential assembly tasks.
    Source: LLM-as-BT-Planner (arXiv 2409.10444, ICRA 2025)
  - BehaviorTree.CPP XML output format for robot execution.
    Source: Multimodal BT Generation (arXiv 2603.06084)
  - py_trees for Python-side tree construction and visualization.
    Source: py_trees 2.4.0 docs; standard robotics stack
  - Strict per-primitive required-field validation — no fuzzy fallbacks.
    Fail immediately with precise location and type mismatch if schema violated.
"""

import os
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

import py_trees
import py_trees.composites
import py_trees.display

from utils import ensure_dir


# ---------------------------------------------------------------------------
# PDDL action and predicate templates (template-based composition)
# ---------------------------------------------------------------------------

ACTION_TEMPLATES = {
    "GRASP": """\
  (:action grasp
    ; Pick up a part from the table.
    :parameters (?p - part)
    :precondition (and (on-table ?p) (gripper-empty))
    :effect (and (held ?p)
                 (not (on-table ?p))
                 (not (gripper-empty))))""",

    "PLACE": """\
  (:action place
    ; Set a held part back on the table.
    :parameters (?p - part)
    :precondition (held ?p)
    :effect (and (on-table ?p)
                 (not (held ?p))
                 (gripper-empty)))""",

    "REORIENT": """\
  (:action reorient
    ; Change part orientation while holding it.
    :parameters (?p - part)
    :precondition (held ?p)
    :effect (reoriented ?p))""",

    "INSERT": """\
  (:action insert
    ; Peg-in-hole / translational insertion.
    ; Covers: dowels, cam-lock pins, shelf pins, wooden pegs.
    :parameters (?peg - part ?socket - part)
    :precondition (and (held ?peg) (on-table ?socket))
    :effect (and (inserted ?peg ?socket)
                 (not (held ?peg))
                 (gripper-empty)))""",

    "FASTEN": """\
  (:action fasten
    ; Tool-mediated tightening: Allen/hex key, screwdriver, cam tool.
    ; Covers: cam locks, wood screws, bolts, hex bolts.
    :parameters (?fastener - part ?hole - part)
    :precondition (and (held ?fastener) (on-table ?hole))
    :effect (and (fastened ?fastener ?hole)
                 (not (held ?fastener))
                 (gripper-empty)))""",

    "PRESS": """\
  (:action press
    ; Press-fit or snap-fit coupling.
    :parameters (?p1 - part ?p2 - part)
    :precondition (and (held ?p1) (on-table ?p2))
    :effect (and (pressed ?p1 ?p2)
                 (not (held ?p1))
                 (gripper-empty)))""",
}

PREDICATE_TEMPLATES = {
    "inserted":   "    (inserted ?peg - part ?socket - part)",
    "fastened":   "    (fastened ?f - part ?h - part)",
    "pressed":    "    (pressed ?p1 - part ?p2 - part)",
    "reoriented": "    (reoriented ?p - part)",
}


BASE_PREDICATES = [
    "    (on-table ?p - part)       ; part is resting on the assembly table",
    "    (held ?p - part)           ; part is grasped by the robot gripper",
    "    (gripper-empty)            ; gripper holds nothing",
]


# ---------------------------------------------------------------------------
# Domain builder
# ---------------------------------------------------------------------------

def _build_domain_pddl(furniture_name: str, primitives_used: list[str]) -> str:
    """Build domain.pddl from primitives_used (only actions actually needed).

    Follows template-based PDDL composition recommended in SPAR (2509.13691)
    and NL2Plan (2405.04215): maintain a library of action templates and
    select only those needed for the current task.
    """
    primitive_to_pred = {
        "INSERT": "inserted",
        "FASTEN": "fastened",
        "PRESS": "pressed",
        "REORIENT": "reoriented",
    }
    action_order = ["GRASP", "PLACE", "REORIENT", "INSERT", "FASTEN", "PRESS"]
    used_set = set(primitives_used)

    predicates_lines = list(BASE_PREDICATES)
    for prim in action_order:
        pred = primitive_to_pred.get(prim)
        if pred and prim in used_set and pred in PREDICATE_TEMPLATES:
            predicates_lines.append(PREDICATE_TEMPLATES[pred])

    predicates_block = "\n".join(predicates_lines)

    action_blocks = []
    for prim in action_order:
        if prim in used_set and prim in ACTION_TEMPLATES:
            action_blocks.append(ACTION_TEMPLATES[prim])

    actions_block = "\n\n".join(action_blocks)

    header = (
        f"; IKEA Assembly PDDL Domain — {furniture_name}\n"
        "; Auto-generated by stage4_formalize.py from actions.json\n"
    )

    return (
        f"{header}\n"
        f"(define (domain ikea-assembly-{furniture_name})\n"
        "  (:requirements :strips :typing :negative-preconditions)\n\n"
        "  (:types part - object)\n\n"
        "  (:predicates\n"
        f"{predicates_block}\n"
        "  )\n\n"
        f"{actions_block}\n"
        ")\n"
    )


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

def _build_problem_pddl(
    furniture_name: str,
    furniture_type: str,
    parts: list[int],
    all_connections: list[dict],
) -> str:
    """Build problem.pddl from parts list and all_connections.

    No ground-truth data (main_data.json) used — everything comes from
    the tree-driven actions.json generated by stage3_action_extraction.py.
    """
    part_names = [f"part{i}" for i in parts]
    objects_str = " ".join(part_names) + " - part"

    init_lines = ["(gripper-empty)"] + [f"(on-table part{i})" for i in parts]
    init_str = "\n    ".join(init_lines)

    goal_lines = []
    for conn in all_connections:
        pred = conn.get("predicate", "inserted")
        p1 = conn.get("part1")
        p2 = conn.get("part2")
        if p1 is not None and p2 is not None:
            goal_lines.append(f"({pred} part{p1} part{p2})")

    goal_str = (
        "\n      ".join(goal_lines)
        if goal_lines
        else "gripper-empty"
    )

    return (
        f"; IKEA Assembly Problem: {furniture_type}/{furniture_name}\n"
        "; Auto-generated by stage4_formalize.py from actions.json\n\n"
        f"(define (problem assemble-{furniture_name})\n"
        f"  (:domain ikea-assembly-{furniture_name})\n\n"
        "  (:objects\n"
        f"    {objects_str}\n"
        "  )\n\n"
        "  (:init\n"
        f"    {init_str}\n"
        "  )\n\n"
        "  (:goal\n"
        "    (and\n"
        f"      {goal_str}\n"
        "    )\n"
        "  )\n"
        ")\n"
    )


# ---------------------------------------------------------------------------
# Behavior Tree — validation schema
# ---------------------------------------------------------------------------

class BehaviorTreeBuildError(Exception):
    """Raised when actions_data fails schema validation or BT construction fails.

    Message format: "step {idx}, action {j}: {PRIMITIVE} missing required field
    '{field}' (expected {type.__name__})" — precise enough to locate the offending
    VLM output without any fuzzy fallback.
    """


# Per-primitive required fields and their expected Python types.
# Any field absent or of wrong type raises BehaviorTreeBuildError immediately.
REQUIRED_ACTION_FIELDS: dict[str, dict[str, type]] = {
    "GRASP":    {"part_id": int},
    "PLACE":    {"part_id": int},
    "INSERT":   {
        "peg_part_id": int, "socket_part_id": int,
        "insertion_dir": str, "fit_type": str,
    },
    "FASTEN":   {
        "fastener_part_id": int, "hole_part_id": int,
        "tool": str, "connection_type": str,
    },
    "PRESS":    {
        "part1_id": int, "part2_id": int,
        "press_dir": str, "fit_type": str,
    },
    "REORIENT": {"part_id": int, "description": str},
}

VALID_PRIMITIVES = set(REQUIRED_ACTION_FIELDS.keys())

# Top-level fields that must be present in actions_data
_REQUIRED_TOP_LEVEL = {"furniture", "category", "steps", "parts", "primitives_used"}


def _validate_actions_data(actions_data: dict) -> None:
    """Validate actions_data against the expected schema.

    Raises BehaviorTreeBuildError with a precise location message on the first
    violation found.  No fuzzy fallbacks — every required field must be present
    with the correct type.
    """
    missing_top = _REQUIRED_TOP_LEVEL - set(actions_data.keys())
    if missing_top:
        raise BehaviorTreeBuildError(
            f"actions_data missing required top-level field(s): {sorted(missing_top)}"
        )

    for step_idx, step in enumerate(actions_data["steps"]):
        if not isinstance(step.get("step_idx"), int):
            raise BehaviorTreeBuildError(
                f"step[{step_idx}] missing or non-integer 'step_idx'"
            )
        if not isinstance(step.get("actions"), list):
            raise BehaviorTreeBuildError(
                f"step[{step_idx}] missing or non-list 'actions'"
            )

        for j, action in enumerate(step["actions"]):
            prim = action.get("primitive")
            if prim not in VALID_PRIMITIVES:
                raise BehaviorTreeBuildError(
                    f"step {step_idx}, action {j}: unknown primitive '{prim}'; "
                    f"expected one of {sorted(VALID_PRIMITIVES)}"
                )
            for field, expected_type in REQUIRED_ACTION_FIELDS[prim].items():
                if field not in action:
                    raise BehaviorTreeBuildError(
                        f"step {step_idx}, action {j}: {prim} missing required "
                        f"field '{field}' (expected {expected_type.__name__})"
                    )
                if not isinstance(action[field], expected_type):
                    raise BehaviorTreeBuildError(
                        f"step {step_idx}, action {j}: {prim}.{field} has type "
                        f"{type(action[field]).__name__}, expected {expected_type.__name__}"
                    )


# ---------------------------------------------------------------------------
# Behavior Tree — py_trees action node classes
# ---------------------------------------------------------------------------

class AssemblyAction(py_trees.behaviour.Behaviour):
    """Planning stub — always returns SUCCESS immediately.

    Used for offline task planning BTs (LLM-as-BT-Planner §3.2;
    VLM-driven BT, arXiv 2501.03968): the BT captures structure and
    parameters for downstream execution; it does not simulate robot physics.

    Subclasses store action-specific parameters as instance attributes so
    that both the ASCII visualization (via the node name) and the XML
    serializer can read them without re-parsing the raw actions_data.
    """

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class GraspAction(AssemblyAction):
    def __init__(self, part_id: int):
        super().__init__(name=f"Grasp(part={part_id})")
        self.part_id = part_id
        self.xml_attribs = {"ID": "Grasp", "part": str(part_id)}


class PlaceAction(AssemblyAction):
    def __init__(self, part_id: int):
        super().__init__(name=f"Place(part={part_id})")
        self.part_id = part_id
        self.xml_attribs = {"ID": "Place", "part": str(part_id)}


class InsertAction(AssemblyAction):
    def __init__(
        self,
        peg_part_id: int,
        socket_part_id: int,
        insertion_dir: str,
        fit_type: str,
    ):
        super().__init__(
            name=f"Insert(peg={peg_part_id}->socket={socket_part_id}, {insertion_dir}, {fit_type})"
        )
        self.peg_part_id = peg_part_id
        self.socket_part_id = socket_part_id
        self.insertion_dir = insertion_dir
        self.fit_type = fit_type
        self.xml_attribs = {
            "ID": "Insert",
            "peg": str(peg_part_id),
            "socket": str(socket_part_id),
            "direction": insertion_dir,
            "fit_type": fit_type,
        }


class FastenAction(AssemblyAction):
    def __init__(
        self,
        fastener_part_id: int,
        hole_part_id: int,
        tool: str,
        connection_type: str,
    ):
        super().__init__(
            name=f"Fasten(f={fastener_part_id}->hole={hole_part_id}, {tool}, {connection_type})"
        )
        self.fastener_part_id = fastener_part_id
        self.hole_part_id = hole_part_id
        self.tool = tool
        self.connection_type = connection_type
        self.xml_attribs = {
            "ID": "Fasten",
            "fastener": str(fastener_part_id),
            "hole": str(hole_part_id),
            "tool": tool,
            "connection": connection_type,
        }


class PressAction(AssemblyAction):
    def __init__(
        self,
        part1_id: int,
        part2_id: int,
        press_dir: str,
        fit_type: str,
    ):
        super().__init__(
            name=f"Press(p1={part1_id}<->p2={part2_id}, {press_dir}, {fit_type})"
        )
        self.part1_id = part1_id
        self.part2_id = part2_id
        self.press_dir = press_dir
        self.fit_type = fit_type
        self.xml_attribs = {
            "ID": "Press",
            "part1": str(part1_id),
            "part2": str(part2_id),
            "direction": press_dir,
            "fit_type": fit_type,
        }


class ReorientAction(AssemblyAction):
    def __init__(self, part_id: int, description: str):
        super().__init__(name=f"Reorient(part={part_id}): {description}")
        self.part_id = part_id
        self.description = description
        self.xml_attribs = {
            "ID": "Reorient",
            "part": str(part_id),
            "description": description,
        }


# Maps primitive name → constructor; called in _make_action_node.
# Fields are accessed by name from the validated action dict — no .get() fallbacks.
def _make_action_node(step_idx: int, j: int, action: dict) -> AssemblyAction:
    """Construct a typed py_trees action node from a validated action dict.

    Raises BehaviorTreeBuildError if a field value is of unexpected type
    (secondary guard; _validate_actions_data should have caught this first).
    """
    prim = action["primitive"]
    try:
        if prim == "GRASP":
            return GraspAction(part_id=action["part_id"])
        if prim == "PLACE":
            return PlaceAction(part_id=action["part_id"])
        if prim == "INSERT":
            return InsertAction(
                peg_part_id=action["peg_part_id"],
                socket_part_id=action["socket_part_id"],
                insertion_dir=action["insertion_dir"],
                fit_type=action["fit_type"],
            )
        if prim == "FASTEN":
            return FastenAction(
                fastener_part_id=action["fastener_part_id"],
                hole_part_id=action["hole_part_id"],
                tool=action["tool"],
                connection_type=action["connection_type"],
            )
        if prim == "PRESS":
            return PressAction(
                part1_id=action["part1_id"],
                part2_id=action["part2_id"],
                press_dir=action["press_dir"],
                fit_type=action["fit_type"],
            )
        if prim == "REORIENT":
            return ReorientAction(
                part_id=action["part_id"],
                description=action["description"],
            )
    except (KeyError, TypeError) as exc:
        raise BehaviorTreeBuildError(
            f"step {step_idx}, action {j}: failed to construct {prim} node — {exc}"
        ) from exc
    # Unreachable if _validate_actions_data ran correctly, but kept for safety.
    raise BehaviorTreeBuildError(
        f"step {step_idx}, action {j}: unhandled primitive '{prim}'"
    )


# ---------------------------------------------------------------------------
# Behavior Tree — tree builder
# ---------------------------------------------------------------------------

def _build_py_tree(actions_data: dict) -> py_trees.composites.Sequence:
    """Build a py_trees Sequence-of-Sequences tree from validated actions_data.

    BT structure (LLM-as-BT-Planner, arXiv 2409.10444):
      Sequence("assemble_{furniture}")     ← root, one per furniture item
        Sequence("step_{idx} new={...}")   ← one per assembly step (post-order)
          GraspAction / InsertAction / ... ← one per extracted robot action
    """
    furniture = actions_data["furniture"]

    root = py_trees.composites.Sequence(
        name=f"assemble_{furniture}",
        memory=True,  # resume from last RUNNING child on re-tick
    )

    for step in actions_data["steps"]:
        step_idx = step["step_idx"]
        new_parts = step.get("new_parts", [])
        sub_parts = step.get("subassembly_parts", [])
        label = f"step_{step_idx} new={new_parts}"
        if sub_parts:
            label += f" sub={sub_parts}"

        step_seq = py_trees.composites.Sequence(name=label, memory=True)

        for j, action in enumerate(step["actions"]):
            node = _make_action_node(step_idx, j, action)
            step_seq.add_child(node)

        root.add_child(step_seq)

    return root


# ---------------------------------------------------------------------------
# Behavior Tree — XML serializer (BehaviorTree.CPP-compatible)
# ---------------------------------------------------------------------------

def _serialize_bt_xml(root: py_trees.composites.Sequence, furniture_name: str) -> str:
    """Walk the py_trees tree and produce BehaviorTree.CPP-compatible XML.

    Grounded in Multimodal BT Generation (arXiv 2603.06084) which targets
    BehaviorTree.CPP format for downstream robot execution.

    XML attributes are read from each node's .xml_attribs dict (AssemblyAction
    subclasses) or synthesized from the node name (Sequence nodes) — never from
    the raw actions_data, ensuring XML always reflects the constructed tree.
    """
    xml_root = ET.Element("root", attrib={"main_tree_to_execute": "AssembleFurniture"})
    bt_elem = ET.SubElement(xml_root, "BehaviorTree", attrib={"ID": "AssembleFurniture"})

    def _add_node(parent_elem: ET.Element, node) -> None:
        if isinstance(node, py_trees.composites.Sequence):
            seq_elem = ET.SubElement(parent_elem, "Sequence", attrib={"name": node.name})
            for child in node.children:
                _add_node(seq_elem, child)
        elif isinstance(node, AssemblyAction):
            ET.SubElement(parent_elem, "Action", attrib=node.xml_attribs)
        else:
            raise BehaviorTreeBuildError(
                f"Unexpected py_trees node type '{type(node).__name__}' "
                f"(name='{node.name}') — only Sequence and AssemblyAction subclasses "
                "are supported in the BT serializer."
            )

    _add_node(bt_elem, root)

    raw_xml = ET.tostring(xml_root, encoding="unicode")
    reparsed = minidom.parseString(raw_xml)
    return reparsed.toprettyxml(indent="  ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_pddl(
    actions_data: dict,
    output_path: str,
) -> tuple[str, str]:
    """Write domain.pddl and problem.pddl; return (domain_path, problem_path).

    Both files are generated entirely from actions_data (actions.json).
    No ground-truth data (main_data.json) is used.

    domain.pddl: built from actions_data["primitives_used"] via ACTION_TEMPLATES.
    problem.pddl: built from actions_data["parts"] + actions_data["all_connections"].
    """
    furniture_name = actions_data["furniture"]
    furniture_type = actions_data["category"]
    primitives_used = actions_data.get("primitives_used", [])
    parts = actions_data.get("parts", [])
    all_connections = actions_data.get("all_connections", [])

    item_out_dir = os.path.join(output_path, furniture_type, furniture_name)
    ensure_dir(item_out_dir)

    domain_path = os.path.join(item_out_dir, "domain.pddl")
    problem_path = os.path.join(item_out_dir, "problem.pddl")

    with open(domain_path, "w", encoding="utf-8") as f:
        f.write(_build_domain_pddl(furniture_name, primitives_used))

    with open(problem_path, "w", encoding="utf-8") as f:
        f.write(_build_problem_pddl(furniture_name, furniture_type, parts, all_connections))

    return domain_path, problem_path


def to_behavior_tree(
    actions_data: dict,
    output_path: str,
) -> str:
    """Build, validate, and write a behavior tree; return path to behavior_tree.xml.

    Outputs three files per furniture item:
      behavior_tree.xml       — BehaviorTree.CPP-compatible XML (robot execution)
      behavior_tree_ascii.txt — ASCII tree representation (always available)
      behavior_tree.png       — Rendered visualization (requires pydot + graphviz)

    Raises:
      BehaviorTreeBuildError — if actions_data fails schema validation or tree
                               construction encounters an unexpected node type.
      ImportError            — if pydot is not installed (with install instructions).
      RuntimeError           — if graphviz executables are not found on PATH.
    """
    # Step 1: strict schema validation — fail fast before building anything
    _validate_actions_data(actions_data)

    # Step 2: build py_trees tree from validated data
    root = _build_py_tree(actions_data)

    furniture_name = actions_data["furniture"]
    furniture_type = actions_data["category"]
    item_out_dir = os.path.join(output_path, furniture_type, furniture_name)
    ensure_dir(item_out_dir)

    # Step 3: ASCII representation — always available, no external dependencies
    ascii_path = os.path.join(item_out_dir, "behavior_tree_ascii.txt")
    ascii_str = py_trees.display.ascii_tree(root)
    with open(ascii_path, "w", encoding="utf-8") as f:
        f.write(ascii_str)
    print(ascii_str)  # surface tree structure to stdout for immediate inspection

    # Step 4: BehaviorTree.CPP-compatible XML
    bt_path = os.path.join(item_out_dir, "behavior_tree.xml")
    with open(bt_path, "w", encoding="utf-8") as f:
        f.write(_serialize_bt_xml(root, furniture_name))

    # Step 5: DOT/PNG/SVG visualization via py_trees.display.dot_tree + pydot.
    # We use dot_tree() (returns a pydot graph object) rather than render_dot_tree()
    # because render_dot_tree() opens the .dot text file with the system default
    # encoding (cp1252 on Windows), which cannot encode the Unicode symbols that
    # py_trees injects into node labels (e.g. U+24C2).
    # Writing the .dot file ourselves with explicit UTF-8, and calling pydot's
    # write_png/write_svg methods (which pipe through graphviz as a binary subprocess),
    # bypasses Python's text codec entirely for the binary output formats.
    try:
        import pydot  # noqa: F401 — presence check
    except ImportError as exc:
        raise ImportError(
            "PNG visualization requires pydot. Install with:\n"
            "  uv add pydot\n"
            "and system graphviz:\n"
            "  Windows: winget install graphviz\n"
            "  macOS:   brew install graphviz\n"
            "  Ubuntu:  sudo apt install graphviz"
        ) from exc

    graph = py_trees.display.dot_tree(root)

    dot_file = os.path.join(item_out_dir, "behavior_tree.dot")
    png_file = os.path.join(item_out_dir, "behavior_tree.png")
    svg_file = os.path.join(item_out_dir, "behavior_tree.svg")

    # Write DOT text with explicit UTF-8 to avoid cp1252 UnicodeEncodeError
    with open(dot_file, "w", encoding="utf-8") as f:
        f.write(graph.to_string())

    # Call graphviz directly using the UTF-8 .dot file already written above.
    # graph.write_png/svg() internally re-writes the DOT source to a temp file
    # using the system default encoding (cp1252 on Windows), which fails on the
    # Unicode symbols py_trees injects into node labels (e.g. U+24C2).
    # Calling `dot` as a subprocess with the explicit file path bypasses that.
    try:
        subprocess.run(["dot", "-Tpng", "-o", png_file, dot_file], check=True)
        subprocess.run(["dot", "-Tsvg", "-o", svg_file, dot_file], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "graphviz executables not found on PATH. Install graphviz:\n"
            "  Windows: winget install graphviz  (then restart shell)\n"
            "  macOS:   brew install graphviz\n"
            "  Ubuntu:  sudo apt install graphviz"
        ) from exc

    return bt_path
