"""Convert our internal BTNode representation to a py_trees tree + renderers.

`py_trees.composites.Sequence(memory=False)` and `Selector(memory=False)` are
the right choice because our BTs are REACTIVE in the Biggar-Zamani 2020 sense:
decisions depend only on the current state, not on a memory of prior ticks.
`memory=False` preserves this property structurally.

Leaf nodes are `_IKEAStubBehaviour` subclasses whose `update()` always returns
SUCCESS — they exist for visualisation, not execution. Real-robot wiring
would replace these with trajectory controllers bound to the actual trained
policies.
"""

from __future__ import annotations

import os
from typing import Any

import py_trees

from .action_model import ActionModel, Literal
from .bt_expansion import BTNode


class _IKEAStubBehaviour(py_trees.behaviour.Behaviour):
    """Placeholder leaf for visualisation. Always returns SUCCESS.

    Real-robot usage would replace this with a behaviour that invokes the
    corresponding RL/IL-trained policy and reports its termination status.
    """

    def __init__(
        self,
        name: str,
        action_model: ActionModel | None = None,
        condition: frozenset[Literal] | None = None,
    ):
        super().__init__(name=name)
        self.action_model = action_model
        self.condition = condition

    def update(self) -> Any:
        return py_trees.common.Status.SUCCESS


def _grounded_args_of(action: ActionModel) -> tuple:
    """Recover the grounded argument tuple by inspecting the add-set."""
    for lit in action.add:
        if lit.name == "inserted" and len(lit.args) == 2:
            return lit.args
        if lit.name == "holding" and len(lit.args) == 1:
            return lit.args
    return ()


def _condition_name(condition: frozenset[Literal]) -> str:
    if not condition:
        return "Condition(∅)"
    items = sorted(str(l) for l in condition)
    if len(items) == 1:
        return f"Condition({items[0]})"
    return "Condition(" + ", ".join(items) + ")"


def _action_name(action: ActionModel) -> str:
    args = _grounded_args_of(action)
    arg_str = ", ".join(str(a) for a in args) if args else ""
    return f"{action.name.capitalize()}({arg_str})"


def to_pytrees(bt_node: BTNode) -> py_trees.behaviour.Behaviour:
    """Recursively build a py_trees.behaviour.Behaviour mirroring bt_node.

    Mapping:
      condition → _IKEAStubBehaviour (leaf)
      action    → _IKEAStubBehaviour (leaf, carries the ActionModel)
      sequence  → py_trees.composites.Sequence(memory=False)
      fallback  → py_trees.composites.Selector(memory=False)
    """
    if bt_node.kind == "condition":
        assert bt_node.condition is not None
        return _IKEAStubBehaviour(
            name=_condition_name(bt_node.condition),
            condition=bt_node.condition,
        )
    if bt_node.kind == "action":
        assert bt_node.action is not None
        return _IKEAStubBehaviour(
            name=_action_name(bt_node.action),
            action_model=bt_node.action,
        )
    if bt_node.kind == "sequence":
        seq = py_trees.composites.Sequence(name="Sequence", memory=False)
        for c in bt_node.children:
            seq.add_child(to_pytrees(c))
        return seq
    if bt_node.kind == "fallback":
        sel = py_trees.composites.Selector(name="Fallback", memory=False)
        for c in bt_node.children:
            sel.add_child(to_pytrees(c))
        return sel
    raise ValueError(f"unknown BTNode kind {bt_node.kind!r}")


def render_unicode(bt_node: BTNode) -> str:
    """Unicode tree string via py_trees.display.unicode_tree."""
    root = to_pytrees(bt_node)
    return py_trees.display.unicode_tree(root=root)


def render_dot_png(bt_node: BTNode, output_path: str) -> str:
    """Render the BT as PNG (and .dot + .svg) via py_trees.display.render_dot_tree.

    `output_path` is interpreted as <dir>/<name>.png; py_trees writes
    <dir>/<name>.dot + .svg + .png alongside it.

    Returns the absolute PNG path. Requires graphviz on PATH for the .png to
    be generated (the .dot file is always produced).
    """
    output_path = os.path.abspath(output_path)
    target_dir = os.path.dirname(output_path)
    os.makedirs(target_dir, exist_ok=True)
    name_root = os.path.splitext(os.path.basename(output_path))[0]

    root = to_pytrees(bt_node)
    py_trees.display.render_dot_tree(
        root, name=name_root, target_directory=target_dir
    )
    return os.path.join(target_dir, f"{name_root}.png")


def to_btcpp_xml(bt_node: BTNode, tree_id: str = "AssembleFurniture") -> str:
    """Emit BehaviorTree.CPP v4 XML (Groot2-compatible).

    Format mirrors `stage4_formalize.py::_build_backchain_bt_xml` so the XML
    is drop-in for the existing Groot2 viewing workflow.
    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    root_xml = ET.Element("root", attrib={"BTCPP_format": "4"})
    bt_elem = ET.SubElement(root_xml, "BehaviorTree", attrib={"ID": tree_id})
    _emit_xml(bt_node, bt_elem)

    # TreeNodesModel — minimal declaration for Groot2 schema
    model = ET.SubElement(root_xml, "TreeNodesModel")
    ET.SubElement(model, "Action", attrib={"ID": "Lift"})
    ET.SubElement(model, "Action", attrib={"ID": "Insert"})
    ET.SubElement(model, "Condition", attrib={"ID": "Check"})

    raw = ET.tostring(root_xml, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def _emit_xml(bt_node: BTNode, parent):
    import xml.etree.ElementTree as ET

    if bt_node.kind == "condition":
        assert bt_node.condition is not None
        lits = ", ".join(sorted(str(l) for l in bt_node.condition))
        ET.SubElement(
            parent,
            "Condition",
            attrib={"ID": "Check", "predicate": lits},
        )
        return
    if bt_node.kind == "action":
        assert bt_node.action is not None
        args = _grounded_args_of(bt_node.action)
        attr = {"ID": bt_node.action.name.capitalize()}
        if bt_node.action.name == "insert" and len(args) == 2:
            attr["peg"] = str(args[0])
            attr["socket"] = str(args[1])
        elif bt_node.action.name == "lift" and len(args) == 1:
            attr["part"] = str(args[0])
        ET.SubElement(parent, "Action", attrib=attr)
        return
    if bt_node.kind == "sequence":
        seq_el = ET.SubElement(parent, "Sequence", attrib={"name": "Sequence"})
        for c in bt_node.children:
            _emit_xml(c, seq_el)
        return
    if bt_node.kind == "fallback":
        fb_el = ET.SubElement(parent, "Fallback", attrib={"name": "Fallback"})
        for c in bt_node.children:
            _emit_xml(c, fb_el)
        return
    raise ValueError(f"unknown BTNode kind {bt_node.kind!r}")
