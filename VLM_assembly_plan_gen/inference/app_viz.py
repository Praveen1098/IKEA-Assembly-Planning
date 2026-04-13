"""app_viz.py — Visualization helpers for the IKEA Assembly Planning Streamlit app.

All functions are pure Python (no Streamlit calls, no LLM calls) so they can be
tested independently.  Heavy imports (plotly, xml, json) are deferred to the
function body to keep module-load time fast.

Exposed API:
    bt_xml_to_visjs_html(xml_str)          → HTML string with vis.js interactive BT tree
    make_plotly_assembly_tree(root, title) → Plotly Figure (interactive network graph)
    make_plotly_plan_sequence(plan_steps)  → Plotly Figure (horizontal bar chart)
    make_pddl_skill_graph()                → Plotly Figure (bipartite predicate-action graph)
    make_causal_graph_figure(actions_data) → Plotly Figure (causal dependency graph for BT)
    bt_to_nuxmv_model(bt_xml_path)         → nuXmv SMV string | None (via BehaVerify)
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# vis.js node style per BT.CPP XML tag
_VIS_STYLE: dict[str, dict] = {
    "BehaviorTree": {"color": "#37474F", "font_color": "#FFFFFF", "shape": "box"},
    "Sequence":     {"color": "#1565C0", "font_color": "#FFFFFF", "shape": "box"},
    "Fallback":     {"color": "#B71C1C", "font_color": "#FFFFFF", "shape": "diamond"},
    "Parallel":     {"color": "#6A1B9A", "font_color": "#FFFFFF", "shape": "hexagon"},
    "Condition":    {"color": "#2E7D32", "font_color": "#FFFFFF", "shape": "ellipse"},
    "Action":       {"color": "#E65100", "font_color": "#FFFFFF", "shape": "box"},
    # Fallback for unknown tags
    "_default":     {"color": "#546E7A", "font_color": "#FFFFFF", "shape": "box"},
}

# Plan-sequence bar colours per PDDL action type
_ACTION_COLORS: dict[str, str] = {
    "grasp":    "#4CAF50",
    "release":  "#8BC34A",
    "insert":   "#2196F3",
    "press":    "#03A9F4",
    "screw":    "#673AB7",
    "reorient": "#FFC107",
}
_DEFAULT_ACTION_COLOR = "#90A4AE"


# ─────────────────────────────────────────────────────────────────────────────
# 1. BT XML → vis.js HTML
# ─────────────────────────────────────────────────────────────────────────────

def bt_xml_to_visjs_html(xml_str: str) -> str:
    """Convert a BehaviorTree.CPP v4 XML string into a self-contained HTML page.

    The page loads vis.js from CDN and renders the BT as an interactive
    hierarchical network:
      - Color-coded by node type (Sequence, Fallback, Condition, Action, Parallel)
      - Hover tooltips show all XML attributes (description, direction, part, etc.)
      - Top-down hierarchical layout matching standard BT diagrams
      - Pan, zoom, drag via vis.js built-ins

    No graphviz or external binary required — CDN-only.
    """
    try:
        root = ET.fromstring(xml_str.strip())
    except ET.ParseError as exc:
        return f"<p>XML parse error: {exc}</p>"

    nodes: list[dict] = []
    edges: list[dict] = []
    counter = [0]

    def _node_id() -> int:
        counter[0] += 1
        return counter[0]

    def _walk(elem: ET.Element, parent_id: int | None) -> None:
        # Skip <TreeNodesModel> and its children — declaration only, not runtime
        if elem.tag == "TreeNodesModel":
            return

        style = _VIS_STYLE.get(elem.tag, _VIS_STYLE["_default"])
        nid = _node_id()

        # Label: tag name + "name" attribute if present
        label = elem.tag
        node_name = elem.get("name") or elem.get("ID") or ""
        if node_name and node_name != label:
            label = f"{label}\n{node_name}"

        # Tooltip: all attributes except "name" (already in label)
        tooltip_lines = [f"<b>{elem.tag}</b>"]
        for k, v in elem.attrib.items():
            if k != "name":
                # Truncate long description strings
                display_v = (v[:80] + "…") if len(v) > 80 else v
                tooltip_lines.append(f"<i>{k}</i>: {display_v}")
        tooltip = "<br>".join(tooltip_lines)

        nodes.append({
            "id":    nid,
            "label": label,
            "title": tooltip,
            "color": {
                "background": style["color"],
                "border":     style["color"],
                "highlight":  {"background": "#FFD54F", "border": "#F57F17"},
            },
            "font":  {"color": style["font_color"], "size": 13},
            "shape": style["shape"],
        })

        if parent_id is not None:
            edges.append({"from": parent_id, "to": nid, "arrows": "to"})

        for child in elem:
            _walk(child, nid)

    _walk(root, None)

    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin: 0; padding: 0; background: #FAFAFA; }}
  #bt-container {{
    width: 100%;
    height: 600px;
    border: 1px solid #BDBDBD;
    border-radius: 4px;
    background: #FFFFFF;
  }}
  #legend {{
    display: flex; flex-wrap: wrap; gap: 8px;
    padding: 6px 12px; font-family: sans-serif; font-size: 12px;
    background: #F5F5F5; border-bottom: 1px solid #E0E0E0;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 4px;
  }}
  .legend-dot {{
    width: 14px; height: 14px; border-radius: 2px;
  }}
</style>
</head>
<body>
<div id="legend">
  <span style="font-weight:bold;margin-right:4px">Node types:</span>
  <div class="legend-item"><div class="legend-dot" style="background:#1565C0"></div>Sequence</div>
  <div class="legend-item"><div class="legend-dot" style="background:#B71C1C;transform:rotate(45deg)"></div>Fallback</div>
  <div class="legend-item"><div class="legend-dot" style="background:#6A1B9A;border-radius:50%"></div>Parallel</div>
  <div class="legend-item"><div class="legend-dot" style="background:#2E7D32;border-radius:50%"></div>Condition</div>
  <div class="legend-item"><div class="legend-dot" style="background:#E65100"></div>Action</div>
  <div class="legend-item"><div class="legend-dot" style="background:#37474F"></div>Root</div>
</div>
<div id="bt-container"></div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});
const container = document.getElementById("bt-container");
const options = {{
  layout: {{
    hierarchical: {{
      enabled: true,
      direction: "UD",
      sortMethod: "directed",
      levelSeparation: 90,
      nodeSpacing: 160,
    }}
  }},
  physics: {{ enabled: false }},
  interaction: {{
    hover: true,
    tooltipDelay: 150,
    navigationButtons: true,
    keyboard: true,
    zoomView: true,
  }},
  edges: {{
    smooth: {{ type: "cubicBezier", forceDirection: "vertical" }},
    color: {{ color: "#90A4AE", highlight: "#F57F17" }},
    width: 1.5,
  }},
  nodes: {{ margin: 8 }},
}};
const network = new vis.Network(container, {{ nodes, edges }}, options);
network.fit({{ animation: {{ duration: 400, easingFunction: "easeInOutQuad" }} }});
</script>
</body>
</html>"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# 2. Plotly interactive assembly tree
# ─────────────────────────────────────────────────────────────────────────────

def make_plotly_assembly_tree(root, title: str):
    """Build an interactive Plotly scatter-graph for an assembly tree.

    Args:
        root: _TreeNode instance (from app.py's _build_assembly_tree())
        title: chart title string

    Returns:
        plotly.graph_objects.Figure
    """
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.DiGraph()
    pos: dict[int, tuple[float, float]] = {}

    def _collect(node, parent=None, level: int = 0,
                 x: float = 0.0, width: float = 4.0) -> None:
        pos[node.id] = (x, -level)
        if parent is not None:
            G.add_edge(parent.id, node.id)
        n = len(node.children)
        for i, child in enumerate(node.children):
            child_x = x + (i - (n - 1) / 2) * (width / max(n, 1))
            _collect(child, node, level + 1, child_x, width / max(n, 1))

    _collect(root)

    # Edge trace
    ex, ey = [], []
    for (u, v) in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ex += [x0, x1, None]
        ey += [y0, y1, None]

    edge_trace = go.Scatter(
        x=ex, y=ey,
        mode="lines",
        line=dict(width=1.5, color="#9E9E9E"),
        hoverinfo="none",
    )

    # Node trace — two colors: leaf (part) vs internal (step)
    def _iter_nodes(node):
        yield node
        for c in node.children:
            yield from _iter_nodes(c)

    all_nodes = list(_iter_nodes(root))
    nx_vals = [pos[n.id][0] for n in all_nodes]
    ny_vals = [pos[n.id][1] for n in all_nodes]
    labels  = [n.name for n in all_nodes]
    colors  = ["#4CAF50" if not n.children else "#1976D2" for n in all_nodes]
    hover   = [
        f"<b>{n.name}</b><br>{'Leaf part' if not n.children else 'Assembly step'}"
        for n in all_nodes
    ]

    node_trace = go.Scatter(
        x=nx_vals, y=ny_vals,
        mode="markers+text",
        marker=dict(size=28, color=colors, line=dict(width=1, color="#FFFFFF")),
        text=labels,
        textposition="middle center",
        textfont=dict(color="#FFFFFF", size=11),
        hovertext=hover,
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=14)),
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="#FFFFFF",
            # Legend annotation
            annotations=[
                dict(x=0, y=1.05, xref="paper", yref="paper", showarrow=False,
                     text="● Part (leaf)  ● Assembly step",
                     font=dict(size=11, color="#555")),
            ],
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. PDDL plan sequence chart
# ─────────────────────────────────────────────────────────────────────────────

def make_plotly_plan_sequence(plan_steps: list[str]):
    """Render a PDDL plan as a horizontal ordered bar chart.

    Args:
        plan_steps: list of grounded action strings, e.g.
                    ["grasp part0", "insert part0 part2", ...]

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    if not plan_steps:
        fig = go.Figure()
        fig.add_annotation(text="No plan steps", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    # Normalise step strings (str(Operator) may include angle brackets)
    def _clean(s: str) -> str:
        return str(s).strip("<>").replace("  ", " ")

    steps = [_clean(s) for s in plan_steps]
    n = len(steps)

    def _color(step: str) -> str:
        key = step.split()[0].lower() if step else ""
        return _ACTION_COLORS.get(key, _DEFAULT_ACTION_COLOR)

    bar_colors = [_color(s) for s in steps]

    fig = go.Figure(go.Bar(
        y=list(range(n)),
        x=[1] * n,
        orientation="h",
        text=steps,
        textposition="inside",
        insidetextanchor="middle",
        marker_color=bar_colors,
        hovertext=[f"Step {i + 1}: {s}" for i, s in enumerate(steps)],
        hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(text=f"PDDL Plan — {n} actions", font=dict(size=14)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n)),
            ticktext=[f"{i + 1}" for i in range(n)],
            autorange="reversed",
            title="Step",
        ),
        margin=dict(l=50, r=20, t=40, b=20),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        height=max(300, 40 * n),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDDL skill library graph
# ─────────────────────────────────────────────────────────────────────────────

def make_pddl_skill_graph():
    """Build a Plotly bipartite graph: predicate → action → predicate.

    Left column:  precondition predicates
    Centre column: action primitives
    Right column: postcondition predicates (effects)

    Edges:
        precond predicate → action  (dashed, grey)
        action → effect predicate   (solid, colored)

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from stage3_5_pddl_validate import SKILL_LIBRARY

    # Collect unique predicates
    all_pre: set[str] = set()
    all_post: set[str] = set()
    for v in SKILL_LIBRARY.values():
        for p in v["pre"]:
            all_pre.add(_strip_pred(p))
        for p in v["post"]:
            cleaned = _strip_pred(p)
            if cleaned:
                all_post.add(cleaned)

    predicates_left  = sorted(all_pre)
    actions_mid      = list(SKILL_LIBRARY.keys())
    predicates_right = sorted(all_post)

    # Layout positions
    def _col_positions(items: list[str], x: float) -> dict[str, tuple[float, float]]:
        n = len(items)
        return {item: (x, (i - (n - 1) / 2) * 1.4) for i, item in enumerate(items)}

    lpos = _col_positions(predicates_left,  0.0)
    mpos = _col_positions(actions_mid,       1.5)
    rpos = _col_positions(predicates_right,  3.0)

    node_x, node_y, node_text, node_colors, node_hover = [], [], [], [], []

    def _add_nodes(pos_dict: dict, color: str, prefix: str = "") -> None:
        for label, (x, y) in pos_dict.items():
            node_x.append(x)
            node_y.append(y)
            node_text.append(label)
            node_colors.append(color)
            node_hover.append(f"<b>{prefix}{label}</b>")

    _add_nodes(lpos, "#66BB6A", "Pre: ")     # green — precond predicates
    _add_nodes(mpos, "#E65100", "Action: ")  # orange — actions
    _add_nodes(rpos, "#42A5F5", "Post: ")    # blue — effect predicates

    edge_x, edge_y = [], []

    def _add_edge(p1: dict, name1: str, p2: dict, name2: str) -> None:
        if name1 not in p1 or name2 not in p2:
            return
        x0, y0 = p1[name1]
        x1, y1 = p2[name2]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # precond → action edges
    for action, spec in SKILL_LIBRARY.items():
        for p in spec["pre"]:
            pred = _strip_pred(p)
            if pred and pred in lpos:
                _add_edge(lpos, pred, mpos, action)

    # action → effect edges
    for action, spec in SKILL_LIBRARY.items():
        for p in spec["post"]:
            pred = _strip_pred(p)
            if pred and pred in rpos:
                _add_edge(mpos, action, rpos, pred)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.2, color="#BDBDBD"),
        hoverinfo="none",
    )
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=22, color=node_colors,
                    line=dict(width=1, color="#FFFFFF")),
        text=node_text,
        textposition="middle center",
        textfont=dict(color="#FFFFFF", size=10),
        hovertext=node_hover,
        hoverinfo="text",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text="Skill Library — Precondition / Effect Graph",
                       font=dict(size=14)),
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.4, 3.4]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="#FFFFFF",
            annotations=[
                dict(x=0.0, y=1.05, xref="paper", yref="paper",
                     showarrow=False, text="● Preconditions",
                     font=dict(color="#66BB6A", size=11)),
                dict(x=0.5, y=1.05, xref="paper", yref="paper",
                     showarrow=False, text="● Actions",
                     font=dict(color="#E65100", size=11)),
                dict(x=1.0, y=1.05, xref="paper", yref="paper",
                     showarrow=False, text="● Postconditions",
                     font=dict(color="#42A5F5", size=11)),
            ],
            height=420,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Causal graph for BT generation (Martín et al., AAMAS 2021)
# ─────────────────────────────────────────────────────────────────────────────

# Distinct palette for parallel groups
_GROUP_PALETTE = [
    "#1976D2", "#388E3C", "#F57C00", "#7B1FA2",
    "#00838F", "#C62828", "#558B2F", "#AD1457",
]


def make_causal_graph_figure(actions_data: dict):
    """Build an interactive Plotly figure of the causal dependency graph.

    Nodes  = connection goals (predicate + parts).
    Edges  = causal dependencies between connections (directed).
    Colors = parallel group — connections sharing a color are causally
             independent and will be wrapped in a BT Parallel node.

    Edge dependency rules (hover to see which applies):
      Rule 1 — active_part of C_i becomes passive_part of C_j
               (the part joined in step i is the mounting base for step j).
      Rule 2 — C_i and C_j share the same active_part (same gripper target;
               robot must release and re-grasp, so the steps are serialized).

    Args:
        actions_data: dict from stage3's extract_actions() / session_state results.

    Returns:
        plotly.graph_objects.Figure
    """
    import sys
    import os
    import plotly.graph_objects as go

    sys.path.insert(0, os.path.dirname(__file__))
    from stage4_formalize import _build_causal_graph, _partition_parallel

    connections = actions_data.get("all_connections", [])
    if not connections:
        fig = go.Figure()
        fig.add_annotation(text="No connections available", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    graph = _build_causal_graph(connections)
    groups = _partition_parallel(connections, graph)

    # Map each connection index to its group index and color
    conn_group: dict[int, int] = {}
    conn_color: dict[int, str] = {}
    for gi, group in enumerate(groups):
        color = _GROUP_PALETTE[gi % len(_GROUP_PALETTE)]
        for idx in group:
            conn_group[idx] = gi
            conn_color[idx] = color

    # Layout: groups as columns, connections within group stacked vertically
    pos: dict[int, tuple[float, float]] = {}
    for gi, group in enumerate(groups):
        x = gi * 2.5
        n = len(group)
        for rank, idx in enumerate(group):
            y = -rank * 1.8 + (n - 1) * 0.9  # centre the column vertically
            pos[idx] = (x, y)

    # Determine which dependency rule(s) caused an edge
    def _edge_rule(i: int, j: int) -> str:
        ci, cj = connections[i], connections[j]
        ai = ci.get("active_part")
        pj = cj.get("passive_part")
        aj = cj.get("active_part")
        reasons = []
        if ai is not None and ai == pj:
            reasons.append(
                f"Rule 1: Part {ai} (active in C{i}) becomes mounting base for C{j}"
            )
        if ai is not None and ai == aj and i < j:
            reasons.append(
                f"Rule 2: Part {ai} is the gripper target for both C{i} and C{j}"
            )
        return "<br>".join(reasons) if reasons else "dependency"

    # Edge traces — one per edge so hover text works per edge
    edge_traces = []
    for i in range(len(connections)):
        for j in sorted(graph[i]):
            x0, y0 = pos[i]
            x1, y1 = pos[j]
            # Mid-point marker carries the hover tooltip
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=2, color="#BDBDBD"),
                hoverinfo="none",
                showlegend=False,
            ))
            edge_traces.append(go.Scatter(
                x=[mx], y=[my],
                mode="markers",
                marker=dict(size=8, color="#BDBDBD", symbol="arrow",
                            angleref="previous",
                            line=dict(width=1, color="#757575")),
                hovertext=_edge_rule(i, j),
                hoverinfo="text",
                showlegend=False,
            ))

    # Node traces — one per parallel group for legend entries
    node_traces = []
    for gi, group in enumerate(groups):
        color = _GROUP_PALETTE[gi % len(_GROUP_PALETTE)]
        xs = [pos[idx][0] for idx in group]
        ys = [pos[idx][1] for idx in group]

        def _short_label(conn: dict) -> str:
            pred = conn.get("predicate", "?")
            p1 = conn.get("part1", "?")
            p2 = conn.get("part2", "?")
            return f"{pred}<br>P{p1}→P{p2}"

        def _hover(idx: int, conn: dict) -> str:
            return (
                f"<b>Connection {idx}</b><br>"
                f"Predicate: {conn.get('predicate', '?')}<br>"
                f"Part {conn.get('part1', '?')} → Part {conn.get('part2', '?')}<br>"
                f"Active part: {conn.get('active_part', '?')}<br>"
                f"Passive part: {conn.get('passive_part', '?')}<br>"
                f"Direction: {conn.get('direction', '?')}<br>"
                f"Parallel group: {gi + 1} of {len(groups)}"
            )

        texts = [_short_label(connections[idx]) for idx in group]
        hovers = [_hover(idx, connections[idx]) for idx in group]

        group_label = (
            f"Group {gi + 1} — parallel"
            if len(groups) > 1
            else "Group 1 — sequential"
        )
        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(size=44, color=color, line=dict(width=2, color="#FFFFFF")),
            text=texts,
            textposition="middle center",
            textfont=dict(color="#FFFFFF", size=10),
            hovertext=hovers,
            hoverinfo="text",
            name=group_label,
        ))

    # Column header annotations (Group labels above each column)
    col_annotations = []
    for gi, group in enumerate(groups):
        x_center = gi * 2.5
        top_y = max(pos[idx][1] for idx in group) + 1.2
        col_annotations.append(dict(
            x=x_center, y=top_y, xref="x", yref="y",
            text=f"<b>Group {gi + 1}</b>",
            showarrow=False,
            font=dict(size=12, color=_GROUP_PALETTE[gi % len(_GROUP_PALETTE)]),
            bgcolor="rgba(255,255,255,0.8)",
        ))

    max_group_size = max(len(g) for g in groups)
    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=dict(
                text=(
                    f"Causal Dependency Graph — {actions_data.get('furniture', '')}  "
                    f"({len(connections)} connections · "
                    f"{len(groups)} parallel group{'s' if len(groups) != 1 else ''})"
                ),
                font=dict(size=14),
            ),
            showlegend=True,
            legend=dict(
                title=dict(text="Parallel groups", font=dict(size=11)),
                orientation="h",
                yanchor="top",
                y=-0.08,
            ),
            hovermode="closest",
            margin=dict(l=20, r=20, t=80, b=80),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="#FFFFFF",
            annotations=col_annotations + [
                dict(
                    x=0.5, y=1.06, xref="paper", yref="paper", showarrow=False,
                    text=(
                        "Nodes = connection goals · Edges = causal dependencies (hover for rule) · "
                        "Same color = parallel group → emitted as BT Parallel node"
                    ),
                    font=dict(size=11, color="#555"),
                ),
            ],
            height=max(420, 140 * max_group_size + 160),
        ),
    )
    return fig


def _strip_pred(raw: str) -> str:
    """Extract base predicate name from a SKILL_LIBRARY pre/post string.

    E.g. "accessible(?p)" → "accessible"
         "not(holding(?active))" → ""   (negated — exclude from graph)
         "gripper-empty" → "gripper-empty"
    """
    s = raw.strip()
    if s.startswith("not(") or s.startswith("~"):
        return ""
    # Remove parameter list
    return s.split("(")[0].strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5. BT → nuXmv model (via BehaVerify)
# ─────────────────────────────────────────────────────────────────────────────

_LTL_PROPERTIES = {
    "Safety — grasp before connect":
        "G(insert(A,B) -> F(holding(A)))",
    "Liveness — goal reachability":
        "G(F(all_connected))",
    "Mutual exclusion — no simultaneous hold":
        "G(!(holding(A) & holding(B) & A!=B))",
    "Progress — gripper eventually released":
        "G(holding(A) -> F(gripper_empty))",
}


def get_ltl_properties() -> dict[str, str]:
    """Return the predefined LTL properties for the IKEA assembly domain."""
    return dict(_LTL_PROPERTIES)


def bt_to_nuxmv_model(bt_xml_str: str) -> str | None:
    """Convert a BT.CPP v4 XML string to a nuXmv SMV model.

    Tries the behaverify CLI first (if installed); falls back to a direct
    BT→SMV encoder that requires no external tools.

    Returns the SMV model string, or None if conversion fails entirely.
    """
    # ── Upgrade path: behaverify CLI ────────────────────────────────────────
    try:
        import behaverify as _bv  # noqa: F401 — confirm package present
        import tempfile, os, subprocess
        from pathlib import Path

        tree_dsl = _bt_xml_to_behaverify_dsl(bt_xml_str)
        if tree_dsl is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                tree_file = os.path.join(tmpdir, "assembly.tree")
                with open(tree_file, "w") as fh:
                    fh.write(tree_dsl)

                result = subprocess.run(
                    ["behaverify", "nuxmv", tree_file, tmpdir, "--generate"],
                    capture_output=True, text=True, timeout=30,
                )
                smv_files = list(Path(tmpdir).glob("*.smv"))
                if result.returncode == 0 and smv_files:
                    return smv_files[0].read_text()
    except Exception:
        pass  # fall through to direct encoder

    # ── Primary path: direct BT → SMV encoder (always available) ────────────
    return _bt_xml_to_smv_direct(bt_xml_str)


def _bt_xml_to_behaverify_dsl(xml_str: str) -> str | None:
    """Convert BT.CPP XML to a minimal BehaVerify .tree DSL string.

    Produces a simplified DSL with only the structural information needed for
    LTL property checking (composite nodes + leaf type annotations).
    Returns None if conversion fails.
    """
    try:
        root = ET.fromstring(xml_str.strip())
    except ET.ParseError:
        return None

    lines: list[str] = []
    # Find the BehaviorTree element
    bt_elem = root.find("BehaviorTree")
    if bt_elem is None:
        return None

    bt_id = bt_elem.get("ID", "AssembleFurniture")
    lines.append(f"configuration {{")
    lines.append(f"  {bt_id}")
    lines.append(f"}}")
    lines.append("")

    def _emit(elem: ET.Element, depth: int = 0) -> None:
        indent = "  " * depth
        tag = elem.tag
        name = elem.get("name") or elem.get("ID") or tag
        # Sanitise name for identifiers
        safe = name.replace(" ", "_").replace("-", "_").replace(".", "_")

        if tag in ("Sequence", "sequence"):
            lines.append(f"{indent}sequence {safe} {{")
            for child in elem:
                _emit(child, depth + 1)
            lines.append(f"{indent}}}")
        elif tag in ("Fallback", "fallback", "Selector"):
            lines.append(f"{indent}fallback {safe} {{")
            for child in elem:
                _emit(child, depth + 1)
            lines.append(f"{indent}}}")
        elif tag in ("Parallel", "parallel"):
            lines.append(f"{indent}parallel {safe} {{")
            for child in elem:
                _emit(child, depth + 1)
            lines.append(f"{indent}}}")
        elif tag == "Condition":
            lines.append(f"{indent}leaf {safe} {{")
            lines.append(f"{indent}  success_on_true")
            lines.append(f"{indent}}}")
        elif tag == "Action":
            lines.append(f"{indent}leaf {safe} {{")
            lines.append(f"{indent}  running_until_success")
            lines.append(f"{indent}}}")
        # Skip root/TreeNodesModel/BehaviorTree wrapper tags
        elif tag in ("root", "BehaviorTree", "TreeNodesModel"):
            for child in elem:
                _emit(child, depth)

    _emit(bt_elem, 0)
    return "\n".join(lines)


def _bt_xml_to_smv_direct(xml_str: str) -> str | None:
    """Convert BT.CPP v4 XML directly to a nuXmv SMV model string.

    No external packages required. Implements standard BT semantics:
      Sequence : returns SUCCESS iff all children succeed (left-to-right)
      Fallback : returns SUCCESS on first succeeding child
      Parallel : simplified — treated as Sequence for model checking
      Condition : boolean env var → SUCCESS if True, FAILURE if False
      Action    : ternary env var in {s, r, f} (nondeterministic)

    Based on: Colledanchise & Ögren, "Behavior Trees in Robotics and AI"
    (Cambridge University Press, 2018) — Chapter 2 formal semantics.
    """
    try:
        doc = ET.fromstring(xml_str.strip())
        _found = doc.find("BehaviorTree")
        bt_elem = _found if _found is not None else doc
        bt_id = bt_elem.get("ID", "AssembleFurniture")

        counter = [0]
        nodes: dict[int, dict] = {}

        def _walk(elem) -> int | None:
            if elem.tag == "TreeNodesModel":
                return None
            nid = counter[0]
            counter[0] += 1
            tag = elem.tag
            raw = elem.get("name") or elem.get("ID") or tag
            safe = "".join(c if c.isalnum() else "_" for c in raw)[:20]
            if safe[:1].isdigit():
                safe = "n" + safe
            var = f"{safe}_{nid}"
            children = [_walk(c) for c in elem]
            children = [c for c in children if c is not None]
            nodes[nid] = dict(tag=tag, var=var, raw=raw, children=children)
            return nid

        top = [e for e in bt_elem if e.tag != "TreeNodesModel"]
        if not top:
            return None
        root_id = _walk(top[0])
        if root_id is None:
            return None

        leaves = {nid: n for nid, n in nodes.items() if not n["children"]}

        out: list[str] = [
            f"-- nuXmv SMV model: {bt_id}",
            "-- Encoding: Colledanchise & Ogren BT formal semantics (2018)",
            "-- Return values: s = SUCCESS, r = RUNNING, f = FAILURE",
            "",
            "MODULE main",
            "",
            "  VAR",
        ]

        for nid, n in sorted(leaves.items()):
            if n["tag"] == "Condition":
                out.append(f"    {n['var']} : boolean;  -- Condition: {n['raw']}")
            else:
                out.append(f"    {n['var']} : {{s, r, f}};  -- Action: {n['raw']}")

        out.append("")
        out.append("  DEFINE")

        emitted: set[int] = set()

        def _ret(nid: int) -> str:
            return f"{nodes[nid]['var']}_ret"

        def _emit_smv(nid: int) -> None:
            if nid in emitted:
                return
            n = nodes[nid]
            for cid in n["children"]:
                _emit_smv(cid)

            tag, var = n["tag"], n["var"]
            children = n["children"]

            if not children:
                if tag == "Condition":
                    out.append(f"    {var}_ret := case {var} : s; TRUE : f; esac;")
                else:
                    out.append(f"    {var}_ret := {var};")
            elif tag in ("Sequence", "sequence"):
                out.append(f"    -- Sequence: {n['raw']}")
                cases: list[str] = []
                for cid in children:
                    cr = _ret(cid)
                    cases += [f"      {cr} = f : f", f"      {cr} = r : r"]
                cases += [f"      {_ret(children[-1])} = s : s", "      TRUE : r"]
                out.append(f"    {var}_ret := case")
                out.extend(c + ";" for c in cases)
                out.append("    esac;")
            elif tag in ("Fallback", "fallback", "Selector"):
                out.append(f"    -- Fallback: {n['raw']}")
                cases = []
                for cid in children:
                    cr = _ret(cid)
                    cases += [f"      {cr} = s : s", f"      {cr} = r : r"]
                cases.append("      TRUE : f")
                out.append(f"    {var}_ret := case")
                out.extend(c + ";" for c in cases)
                out.append("    esac;")
            else:
                # Parallel and unknown composites — conservative Sequence encoding
                out.append(f"    -- Parallel (M-of-N, simplified): {n['raw']}")
                cases = []
                for cid in children:
                    cr = _ret(cid)
                    cases += [f"      {cr} = f : f", f"      {cr} = r : r"]
                cases += [f"      {_ret(children[-1])} = s : s", "      TRUE : r"]
                out.append(f"    {var}_ret := case")
                out.extend(c + ";" for c in cases)
                out.append("    esac;")
            emitted.add(nid)

        _emit_smv(root_id)

        root_var = nodes[root_id]["var"]
        action_vars = [
            n["var"] for n in nodes.values()
            if not n["children"] and n["tag"] != "Condition"
        ]

        out.extend([
            "",
            "  -- --- LTL Specification Properties ---",
            f"  LTLSPEC G(F({root_var}_ret = s));  -- Goal reachability",
            f"  LTLSPEC G({root_var}_ret != f -> F({root_var}_ret = s));  -- No permanent failure",
        ])
        if len(action_vars) >= 2:
            out.append(
                f"  LTLSPEC G(!({action_vars[0]} = r & {action_vars[1]} = r));  "
                "-- No two actions running simultaneously"
            )

        return "\n".join(out)

    except Exception:
        return None
