"""Standards-compliant BT visualisation (Groot2 / BehaviorTree.CPP convention).

Node styling follows the published BehaviorTree.CPP v4 + Groot2 colour/shape
conventions so the rendered output is immediately recognisable to anyone
familiar with those tools:

  Sequence   → blue rounded rectangle, label '→'
  Fallback   → orange rounded rectangle, label '?' (Groot2 calls it Selector)
  Action     → green rectangle with solid border, grounded args in label
  Condition  → light-gray ellipse with the literal conjunction as label

All nodes get a small monospace body underneath showing their role
(e.g. grounded parameters, condition literals). Edges are directed, top-down.
A compact legend sits below the graph.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .action_model import ActionModel, Literal
from .bt_expansion import BTNode


# ---------- Groot2 palette ----------

_COLORS = {
    "sequence":  {"fill": "#DAE8FC", "border": "#1A73E8"},  # blue
    "fallback":  {"fill": "#FFE6CC", "border": "#E8A037"},  # orange
    "action":    {"fill": "#D5E8D4", "border": "#4E8F3E"},  # green
    "condition": {"fill": "#F5F5F5", "border": "#808080"},  # gray
    "goal":      {"fill": "#F8CECC", "border": "#B85450"},  # red (root goal)
}

_GLYPH = {
    "sequence": "&#8594;",  # →
    "fallback": "?",
}


# ---------- Label helpers ----------

def _condition_label(condition: frozenset[Literal]) -> str:
    if not condition:
        return "⊤"
    items = sorted(str(l) for l in condition)
    # Break long conjunctions across lines for readability
    if len(items) <= 2:
        return " ∧ ".join(items)
    lines = []
    buf = []
    for lit in items:
        buf.append(lit)
        if len(buf) >= 2:
            lines.append(" ∧ ".join(buf))
            buf = []
    if buf:
        lines.append(" ∧ ".join(buf))
    return "\\n".join(lines)


def _action_label(action: ActionModel) -> str:
    """Grounded action, e.g. 'Insert(peg=0, socket=2)'."""
    args = _grounded_args(action)
    if action.name == "lift" and len(args) == 1:
        return f"Lift(part={args[0]})"
    if action.name == "insert" and len(args) == 2:
        return f"Insert(peg={args[0]}, socket={args[1]})"
    return f"{action.name.capitalize()}({', '.join(str(a) for a in args)})"


def _grounded_args(action: ActionModel) -> tuple:
    for lit in action.add:
        if lit.name == "inserted" and len(lit.args) == 2:
            return lit.args
        if lit.name == "holding" and len(lit.args) == 1:
            return lit.args
    return ()


# ---------- DOT emission ----------

def _escape(s: str) -> str:
    return s.replace('"', r"\"")


def _emit_node(
    node: BTNode,
    node_id: str,
    is_root: bool,
    lines: list[str],
    counter: list[int],
) -> None:
    """Emit a styled node + recurse on children."""
    if node.kind == "condition":
        assert node.condition is not None
        fill = _COLORS["goal"]["fill"] if is_root else _COLORS["condition"]["fill"]
        border = _COLORS["goal"]["border"] if is_root else _COLORS["condition"]["border"]
        label = _condition_label(node.condition)
        header = "GOAL" if is_root else "Condition"
        lines.append(
            f'  {node_id} ['
            f'shape=ellipse, style="filled,rounded", '
            f'fillcolor="{fill}", color="{border}", '
            f'label=<<b>{header}</b><br/><font point-size="10">'
            f'{_escape(label).replace(chr(92)+"n","<br/>")}</font>>];'
        )
    elif node.kind == "action":
        assert node.action is not None
        color = _COLORS["action"]
        lines.append(
            f'  {node_id} ['
            f'shape=box, style="filled,rounded,bold", '
            f'fillcolor="{color["fill"]}", color="{color["border"]}", '
            f'label=<<b>{_escape(_action_label(node.action))}</b>>];'
        )
    elif node.kind == "sequence":
        color = _COLORS["sequence"]
        lines.append(
            f'  {node_id} ['
            f'shape=box, style="filled,rounded", '
            f'fillcolor="{color["fill"]}", color="{color["border"]}", '
            f'label=<<b><font point-size="18">{_GLYPH["sequence"]}</font></b><br/>'
            f'<font point-size="10">Sequence</font>>];'
        )
    elif node.kind == "fallback":
        color = _COLORS["fallback"]
        lines.append(
            f'  {node_id} ['
            f'shape=box, style="filled,rounded", '
            f'fillcolor="{color["fill"]}", color="{color["border"]}", '
            f'label=<<b><font point-size="18">{_GLYPH["fallback"]}</font></b><br/>'
            f'<font point-size="10">Fallback</font>>];'
        )
    else:
        raise ValueError(f"unknown kind {node.kind}")

    # Children
    for child in node.children:
        counter[0] += 1
        child_id = f"n{counter[0]}"
        _emit_node(child, child_id, False, lines, counter)
        lines.append(f"  {node_id} -> {child_id};")


def _emit_legend(lines: list[str]) -> None:
    """Compact legend subgraph — renders as a separate cluster at the bottom."""
    legend = [
        ('legend_seq', "sequence", f'<<b><font point-size="16">{_GLYPH["sequence"]}</font></b> Sequence>'),
        ('legend_fb', "fallback", f'<<b><font point-size="16">{_GLYPH["fallback"]}</font></b> Fallback>'),
        ('legend_act', "action", '<<b>Action</b>>'),
        ('legend_cond', "condition", '<<b>Condition</b>>'),
        ('legend_goal', "goal", '<<b>GOAL</b>>'),
    ]
    lines.append("  subgraph cluster_legend {")
    lines.append('    label="Legend"; style="dashed"; fontsize=10;')
    for nid, kind, label in legend:
        c = _COLORS[kind]
        shape = "ellipse" if kind in ("condition", "goal") else "box"
        lines.append(
            f'    {nid} [shape={shape}, style="filled,rounded", '
            f'fillcolor="{c["fill"]}", color="{c["border"]}", '
            f'label={label}];'
        )
    # Force legend nodes to line up horizontally
    for a, b in zip(legend, legend[1:]):
        lines.append(f"    {a[0]} -> {b[0]} [style=invis];")
    lines.append("  }")


def to_dot(root: BTNode, title: str | None = None) -> str:
    """Return the full .dot source for a standards-compliant BT diagram."""
    header = ["digraph BT {"]
    if title:
        header.append(f'  labelloc="t"; label=<<b>{_escape(title)}</b>>; fontsize=14;')
    header.extend([
        '  graph [rankdir=TB, splines=ortho, nodesep=0.35, ranksep=0.45];',
        '  node [fontname="Helvetica"];',
        '  edge [arrowsize=0.7];',
    ])
    body: list[str] = []
    counter = [0]
    _emit_node(root, "n0", True, body, counter)
    _emit_legend(body)
    return "\n".join(header + body + ["}"])


# ---------- PNG + DOT writers ----------

def write_bt_dot(root: BTNode, output_path: str | os.PathLike, title: str | None = None) -> Path:
    """Write a .dot file to `output_path` and return the Path."""
    output_path = Path(output_path).with_suffix(".dot")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(to_dot(root, title=title), encoding="utf-8")
    return output_path


def write_bt_png(
    root: BTNode,
    output_path: str | os.PathLike,
    title: str | None = None,
    dpi: int = 120,
) -> Path | None:
    """Write a .dot + .png via Graphviz `dot`. Returns the PNG path, or None
    if Graphviz isn't on PATH (the .dot still gets written)."""
    output_path = Path(output_path).with_suffix(".png")
    dot_path = output_path.with_suffix(".dot")
    write_bt_dot(root, dot_path, title=title)

    if shutil.which("dot") is None:
        return None

    try:
        subprocess.run(
            ["dot", "-Tpng", f"-Gdpi={dpi}", "-o", str(output_path), str(dot_path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return output_path


def write_bt_svg(
    root: BTNode,
    output_path: str | os.PathLike,
    title: str | None = None,
) -> Path | None:
    """Vector equivalent of write_bt_png. SVG scales without pixelation."""
    output_path = Path(output_path).with_suffix(".svg")
    dot_path = output_path.with_suffix(".dot")
    write_bt_dot(root, dot_path, title=title)

    if shutil.which("dot") is None:
        return None

    try:
        subprocess.run(
            ["dot", "-Tsvg", "-o", str(output_path), str(dot_path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return output_path
