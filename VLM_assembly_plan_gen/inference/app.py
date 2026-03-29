"""
IKEA Assembly Planning — Streamlit Visualization App

Run from VLM_assembly_plan_gen/:
    streamlit run inference/app.py
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Path setup (mirrors run.py line 9) ───────────────────────────────────────
# inference/ must be at the front so inference/utils.py takes priority over
# the empty VLM_assembly_plan_gen/utils/ package.  Use append (not insert) for
# the parent so it sits behind inference/ in the search order.
sys.path.insert(0, str(Path(__file__).parent))         # inference/ → config, utils, stage*
sys.path.append(str(Path(__file__).parent.parent))     # VLM_assembly_plan_gen/ → llm.*

import streamlit as st
import yaml

from config import DATA_DIR, MANUAL_DATA_PATH, OUTPUT_DIR, RECIPE_PATH, SCENE_DIR
from utils import alphanumeric_sort_key, load_json


# ─────────────────────────────────────────────────────────────────────────────
# Cached loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_catalog():
    """Return (catalog, data) where catalog = {category: [(idx, name), ...]}."""
    data = load_json(MANUAL_DATA_PATH)
    catalog: dict[str, list[tuple[int, str]]] = {}
    for i, item in enumerate(data):
        catalog.setdefault(item["category"], []).append((i, item["name"]))
    return catalog, data


@st.cache_data(show_spinner=False)
def load_model_ids():
    """Return (list[model_id], default_model_id) from recipe.yaml."""
    with open(RECIPE_PATH, encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    ids = [m["id"] for m in recipe["models"]]
    default = recipe.get("default_model", ids[0])
    return ids, default


@st.cache_resource(show_spinner="Loading LLM...")
def get_llm(model_id: str):
    from llm.model import load_llm_from_recipe
    return load_llm_from_recipe(RECIPE_PATH, model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_tree(node, indent: int = 0) -> str:
    """Recursively render a nested list / int tree as indented text."""
    prefix = "  " * indent
    if isinstance(node, (int, str)):
        return f"{prefix}Part {node}\n"
    lines = [f"{prefix}[\n"]
    for child in node:
        lines.append(render_tree(child, indent + 1))
    lines.append(f"{prefix}]\n")
    return "".join(lines)


def list_manual_pages(cat: str, name: str) -> list[str]:
    pages_dir = os.path.join(DATA_DIR, "pdfs", cat, name)
    if not os.path.isdir(pages_dir):
        return []
    files = [
        os.path.join(pages_dir, f)
        for f in os.listdir(pages_dir)
        if f.lower().endswith(".png")
    ]
    return sorted(files, key=lambda p: alphanumeric_sort_key(os.path.basename(p)))


def scene_image_path(cat: str, name: str, scene_type: str) -> str:
    fname = "scene_annotated.png" if scene_type == "original" else "scene_rot_annotated.png"
    return os.path.join(SCENE_DIR, cat, name, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Assembly graph visualization (ported from manual2skill_demo.py)
# ─────────────────────────────────────────────────────────────────────────────

class _TreeNode:
    """Node for assembly graph visualization."""
    def __init__(self, name: str, node_id: int, image=None):
        self.name = name
        self.id = node_id
        self.children: list = []
        self.image = image


def _build_assembly_tree(nested_list, png_counter: list, node_id: list,
                         cat: str, name: str) -> _TreeNode:
    """Build a _TreeNode tree from a nested-list assembly tree.

    Leaf nodes use part PNG images from data/parts/{cat}/{name}/{id:02d}.obj.png.
    Non-leaf nodes use step mask images from data/mask/{cat}/{name}/step_{idx}_no_seg_numbered.png.
    Missing images are stored as None (node drawn without an overlay image).
    """
    if isinstance(nested_list, (int, str)):
        current_id = node_id[0]
        node_id[0] += 1
        part_idx = int(nested_list)
        png_path = os.path.join(
            DATA_DIR, "parts", cat, name, f"{part_idx:02d}.obj.png"
        )
        if not os.path.exists(png_path):
            png_path = None
        return _TreeNode(name=f"Part {part_idx}", node_id=current_id, image=png_path)

    children = [
        _build_assembly_tree(child, png_counter, node_id, cat, name)
        for child in nested_list
    ]
    current_id = node_id[0]
    node_id[0] += 1
    mask_path = os.path.join(
        DATA_DIR, "mask", cat, name,
        f"step_{png_counter[0]}_no_seg_numbered.png",
    )
    png_counter[0] += 1
    if not os.path.exists(mask_path):
        mask_path = None
    node = _TreeNode(name=f"Step {png_counter[0] - 1}", node_id=current_id, image=mask_path)
    node.children = children
    return node


def _add_graph_edges(graph, node: _TreeNode, parent=None,
                     pos: dict | None = None, level: int = 0,
                     x: float = 0.0, width: float = 4.0) -> dict:
    if pos is None:
        pos = {}
    pos[node.id] = (x, -level - 2)
    if parent is not None:
        graph.add_edge(parent.id, node.id)
    n = len(node.children)
    for i, child in enumerate(node.children):
        _add_graph_edges(graph, child, node, pos, level + 1,
                         x + (i - n / 2) * width, width)
    return pos


def _find_node_by_id(node: _TreeNode, target_id: int) -> _TreeNode | None:
    if node.id == target_id:
        return node
    for child in node.children:
        found = _find_node_by_id(child, target_id)
        if found:
            return found
    return None


def _draw_assembly_graph(root: _TreeNode, title: str, ax) -> None:
    """Draw the assembly tree with part/step images at nodes onto ax."""
    import networkx as nx
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    G = nx.DiGraph()
    pos = _add_graph_edges(G, root)

    nx.draw(G, pos, with_labels=True, node_size=6000, ax=ax, arrows=True)

    for node_id, (x, y) in pos.items():
        node = _find_node_by_id(root, node_id)
        if node and node.image:
            try:
                img = Image.open(node.image).resize((50, 50))
                imagebox = OffsetImage(img, zoom=1)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False,
                                    box_alignment=(0.5, 0.5))
                ax.add_artist(ab)
            except Exception:
                pass

    ax.set_title(title)


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IKEA Assembly Planning",
    page_icon="🪑",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("IKEA Assembly Planning Visualizer")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls
# ─────────────────────────────────────────────────────────────────────────────

catalog, data = load_catalog()
model_ids, default_model = load_model_ids()

with st.sidebar:
    st.header("Furniture")
    category = st.selectbox("Category", sorted(catalog.keys()), key="sel_cat")
    items = catalog[category]
    name_to_idx = {name: idx for idx, name in items}
    furniture_name = st.selectbox("Name", [n for _, n in items], key="sel_name")
    furniture_idx = name_to_idx[furniture_name]
    furniture_item = data[furniture_idx]

    st.divider()
    st.header("Pipeline")
    model_id = st.selectbox(
        "Model",
        model_ids,
        index=model_ids.index(default_model) if default_model in model_ids else 0,
        key="sel_model",
    )
    prompt_type = st.selectbox("Prompt Type", ["numbered", "unnumbered"], key="sel_prompt")
    scene_type = st.selectbox("Scene Type", ["original", "rotated"], key="sel_scene")

    st.divider()
    st.header("Stages to Run")
    run_stage3 = st.checkbox("Stage 3 — Actions", value=True)
    run_pddl   = st.checkbox("Stage 4 — PDDL",    value=True)
    run_bt     = st.checkbox("Stage 4 — Behavior Tree", value=True)
    debug_mode = st.checkbox("Debug mode", value=False)

    st.divider()
    run_clicked = st.button("▶ Run Pipeline", type="primary", width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ─────────────────────────────────────────────────────────────────────────────

if run_clicked:
    from stage1_associate import select_materials_for_planning
    from stage2_planning import create_plan
    from convert import convert_to_tree
    from stage3_action_extraction import extract_actions
    from stage4_formalize import to_behavior_tree, to_pddl

    args = argparse.Namespace(
        start=furniture_idx,
        end=furniture_idx + 1,
        model=model_id,
        debug=debug_mode,
        prompt_type=prompt_type,
        scene_type="original" if scene_type == "original" else "rotated",
        output_format="bt",
    )
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, timestamp)

    llm = get_llm(model_id)
    results: dict = {"output_path": output_path, "timestamp": timestamp}

    run_s3 = run_stage3 or run_pddl or run_bt

    with st.status("Running pipeline…", expanded=True) as pipeline_status:
        # Stage 1
        st.write("Stage 1 — Part Association…")
        try:
            results["stage1"] = select_materials_for_planning(
                furniture_name, category, output_path, args, llm
            )
            st.write("Stage 1 complete")
        except Exception as exc:
            results["stage1_error"] = str(exc)
            st.write(f"Stage 1 failed: {exc}")

        # Stage 2
        if "stage1" in results:
            st.write("Stage 2 — Assembly Planning…")
            try:
                results["stage2"] = create_plan(
                    furniture_name, category, output_path,
                    results["stage1"], args, llm=llm
                )
                st.write("Stage 2 complete")
            except Exception as exc:
                results["stage2_error"] = str(exc)
                st.write(f"Stage 2 failed: {exc}")

        # Convert → tree.json
        if "stage2" in results:
            st.write("Convert — Assembly Tree…")
            try:
                convert_to_tree(
                    furniture_name, category, output_path,
                    results["stage2"], args, llm
                )
                tree_path = os.path.join(output_path, category, furniture_name, "tree.json")
                raw_str = json.load(open(tree_path, encoding="utf-8"))
                results["tree"] = json.loads(raw_str)
                st.write("Tree conversion complete")
            except Exception as exc:
                results["tree_error"] = str(exc)
                st.write(f"Tree conversion failed: {exc}")

        # Stage 3 — Actions
        if run_s3 and "stage2" in results:
            st.write("Stage 3 — Action Extraction…")
            try:
                results["actions"] = extract_actions(
                    furniture_name, category, output_path,
                    results["stage2"], results["stage1"], args, llm
                )
                st.write("Stage 3 complete")
            except Exception as exc:
                results["actions_error"] = str(exc)
                st.write(f"Stage 3 failed: {exc}")

        # Stage 4 — PDDL
        if run_pddl and "actions" in results:
            st.write("Stage 4 — PDDL generation…")
            try:
                domain_p, problem_p = to_pddl(results["actions"], output_path)
                results["domain_pddl"]  = open(domain_p,  encoding="utf-8").read()
                results["problem_pddl"] = open(problem_p, encoding="utf-8").read()
                st.write("PDDL generated")
            except Exception as exc:
                results["pddl_error"] = str(exc)
                st.write(f"PDDL failed: {exc}")

        # Stage 4 — Behavior Tree
        if run_bt and "actions" in results:
            st.write("Stage 4 — Behavior Tree generation…")
            try:
                to_behavior_tree(results["actions"], output_path)
                item_dir = os.path.join(output_path, category, furniture_name)
                results["bt_ascii"] = open(
                    os.path.join(item_dir, "behavior_tree_ascii.txt"), encoding="utf-8"
                ).read()
                results["bt_xml"] = open(
                    os.path.join(item_dir, "behavior_tree.xml"), encoding="utf-8"
                ).read()
                png_path = os.path.join(item_dir, "behavior_tree.png")
                if os.path.exists(png_path):
                    results["bt_png"] = png_path
                svg_path = os.path.join(item_dir, "behavior_tree.svg")
                if os.path.exists(svg_path):
                    results["bt_svg"] = svg_path
                st.write("Behavior Tree generated")
            except Exception as exc:
                results["bt_error"] = str(exc)
                st.write(f"Behavior Tree failed: {exc}")

        pipeline_status.update(label="Pipeline complete!", state="complete")

    st.session_state["results"] = results
    st.session_state["last_furniture"] = (category, furniture_name, furniture_idx)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs — always rendered; post-run tabs show session_state results
# ─────────────────────────────────────────────────────────────────────────────

results = st.session_state.get("results", {})
last = st.session_state.get("last_furniture", (category, furniture_name, furniture_idx))
# For Inputs tab always use current sidebar selection
disp_cat, disp_name, disp_idx = category, furniture_name, furniture_idx

tab_inputs, tab_s1, tab_s2, tab_tree, tab_s3, tab_pddl, tab_bt = st.tabs([
    "Inputs",
    "Stage 1 — Parts Identification",
    "Stage 2 — Plan Generation",
    "Assembly Tree Generation",
    "Stage 3 — Action Extraction",
    "PDDL Generation",
    "Behavior Tree Generation",
])

# ── Tab: Inputs ───────────────────────────────────────────────────────────────
with tab_inputs:
    col_scene, col_gt = st.columns([1, 1])

    with col_scene:
        st.subheader("Scene Image")
        scene_p = scene_image_path(disp_cat, disp_name, scene_type)
        if os.path.exists(scene_p):
            st.image(scene_p, width='stretch')
        else:
            st.warning(f"Scene image not found: `{scene_p}`")

        st.subheader("Manual Pages")
        pages = list_manual_pages(disp_cat, disp_name)
        if pages:
            cols = st.columns(min(len(pages), 4))
            for i, pg in enumerate(pages):
                with cols[i % 4]:
                    st.image(pg, caption=os.path.basename(pg), width='stretch')
        else:
            st.info("No manual page images found.")

    with col_gt:
        st.subheader("Ground-Truth Assembly Plan")
        item = data[disp_idx]

        st.metric("Total Parts", item.get("parts_ct", "—"))

        st.markdown("**Assembly Tree** (ground truth)")
        gt_tree = item.get("assembly_tree")
        if gt_tree is not None:
            st.code(render_tree(gt_tree).strip(), language=None)

        st.markdown("**Connection Relations**")
        conn_rel = item.get("connection_relation", [])
        if conn_rel:
            import pandas as pd
            df_conn = pd.DataFrame(conn_rel, columns=["Part A", "Part B"])
            st.dataframe(df_conn, width='stretch', hide_index=True)
        else:
            st.info("No connection data.")

        st.markdown("**Ground-Truth Steps**")
        gt_steps = item.get("steps", [])
        if gt_steps:
            for step in gt_steps:
                label = f"Step {step.get('step_id', '?')}  (page {step.get('page_id', '?')})"
                with st.expander(label):
                    st.write("**Parts:**", step.get("parts", []))
                    st.write("**Connections:**", step.get("connections", []))
        else:
            st.info("No step data.")

# ── Tab: Stage 1 ─────────────────────────────────────────────────────────────
with tab_s1:
    if "stage1_error" in results:
        st.error(f"Stage 1 failed: {results['stage1_error']}")
    elif "stage1" in results:
        st.subheader("Parts Table (Stage 1 Output)")
        raw = results["stage1"]
        # Try to parse and pretty-print as JSON
        try:
            parsed = json.loads(raw)
            st.json(parsed)
        except json.JSONDecodeError:
            st.text(raw)
    else:
        st.info("Run the pipeline to see Stage 1 output.")

# ── Tab: Stage 2 ─────────────────────────────────────────────────────────────
with tab_s2:
    if "stage2_error" in results:
        st.error(f"Stage 2 failed: {results['stage2_error']}")
    elif "stage2" in results:
        st.subheader("Assembly Plan (Stage 2 Output)")
        st.markdown(results["stage2"])
    else:
        st.info("Run the pipeline to see Stage 2 output.")

# ── Tab: Assembly Tree ────────────────────────────────────────────────────────
with tab_tree:
    if "tree_error" in results:
        st.error(f"Tree conversion failed: {results['tree_error']}")

    gt_tree = data[disp_idx].get("assembly_tree")
    predicted_tree = results.get("tree")

    # Determine how many columns to draw
    _has_gt = gt_tree is not None
    _has_pred = predicted_tree is not None

    if _has_gt or _has_pred:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            num_cols = (1 if _has_gt else 0) + (1 if _has_pred else 0)
            fig, axes = plt.subplots(1, num_cols, figsize=(10 * num_cols, 10))
            if num_cols == 1:
                axes = [axes]

            _ax_idx = 0
            if _has_pred:
                try:
                    pred_root = _build_assembly_tree(
                        predicted_tree, [0], [0], disp_cat, disp_name
                    )
                    _draw_assembly_graph(pred_root, "VLM Predicted Assembly Graph", axes[_ax_idx])
                except Exception as _e:
                    axes[_ax_idx].set_title(f"Predicted graph unavailable: {_e}")
                _ax_idx += 1

            if _has_gt:
                try:
                    gt_root = _build_assembly_tree(
                        gt_tree, [0], [0], disp_cat, disp_name
                    )
                    _draw_assembly_graph(gt_root, "Ground Truth Assembly Graph", axes[_ax_idx])
                except Exception as _e:
                    axes[_ax_idx].set_title(f"Ground truth graph unavailable: {_e}")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except ImportError as _e:
            st.warning(f"Graph visualization requires networkx and matplotlib: {_e}")
            if _has_pred:
                st.code(render_tree(predicted_tree).strip(), language=None)
        except Exception as _e:
            st.error(f"Graph visualization failed: {_e}")
            if _has_pred:
                st.code(render_tree(predicted_tree).strip(), language=None)

    if _has_pred:
        with st.expander("Raw JSON (predicted tree)", expanded=False):
            st.json(predicted_tree)
    elif not _has_gt:
        st.info("Run the pipeline to see the assembly tree.")

# ── Tab: Stage 3 — Actions ────────────────────────────────────────────────────
with tab_s3:
    if "actions_error" in results:
        st.error(f"Stage 3 failed: {results['actions_error']}")
    elif "actions" in results:
        acts = results["actions"]
        st.subheader(f"Actions — {acts.get('furniture')} ({acts.get('category')})")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Parts", len(acts.get("parts", [])))
        col_b.metric("Primitives used", ", ".join(acts.get("primitives_used", [])))
        col_c.metric("Connections", len(acts.get("all_connections", [])))

        st.markdown("**All Connections**")
        all_conns = acts.get("all_connections", [])
        if all_conns:
            import pandas as pd
            df_ac = pd.DataFrame(all_conns)
            st.dataframe(df_ac, width='stretch', hide_index=True)

        st.markdown("---")
        st.markdown("**Per-Step Actions**")
        for step in acts.get("steps", []):
            idx_s = step.get("step_idx", "?")
            new_p = step.get("new_parts", [])
            sub_p = step.get("subassembly_parts", [])
            label = f"Step {idx_s}  |  new={new_p}  sub={sub_p}"
            with st.expander(label):
                # Reasoning
                if step.get("reasoning"):
                    st.markdown(f"*{step['reasoning']}*")
                # Actions table
                action_list = step.get("actions", [])
                if action_list:
                    import pandas as pd
                    df_acts = pd.DataFrame(action_list)
                    st.dataframe(df_acts, width='stretch', hide_index=True)
                # Connections formed
                conns = step.get("connections_formed", [])
                if conns:
                    st.markdown("**Connections formed:**")
                    for c in conns:
                        st.write(f"- `{c.get('predicate')}` : Part {c.get('part1')} ↔ Part {c.get('part2')}")
                # Warnings
                for w in step.get("warnings", []):
                    st.warning(w)
    else:
        st.info("Run the pipeline with Stage 3 enabled to see actions.")

# ── Tab: PDDL ─────────────────────────────────────────────────────────────────
with tab_pddl:
    if "pddl_error" in results:
        st.error(f"PDDL generation failed: {results['pddl_error']}")
    elif "domain_pddl" in results or "problem_pddl" in results:
        col_d, col_p = st.columns(2)
        with col_d:
            st.subheader("domain.pddl")
            st.code(results.get("domain_pddl", ""), language="lisp")
        with col_p:
            st.subheader("problem.pddl")
            st.code(results.get("problem_pddl", ""), language="lisp")
    else:
        st.info("Run the pipeline with PDDL enabled to see output.")

# ── Tab: Behavior Tree ────────────────────────────────────────────────────────
with tab_bt:
    if "bt_error" in results:
        st.error(f"Behavior Tree generation failed: {results['bt_error']}")
    elif "bt_ascii" in results or "bt_xml" in results:
        import streamlit.components.v1 as components

        st.subheader("Behavior Tree")

        zoom_pct = st.slider(
            "Zoom", min_value=25, max_value=300, value=100, step=25,
            key="bt_zoom", format="%d%%",
        )

        if "bt_svg" in results and os.path.exists(results["bt_svg"]):
            with open(results["bt_svg"], encoding="utf-8") as _f:
                svg_content = _f.read()
            scale = zoom_pct / 100
            _html = (
                '<div style="overflow:auto;width:100%;height:600px;'
                'border:1px solid #ddd;border-radius:4px;">'
                f'<div style="transform:scale({scale});transform-origin:top left;'
                'display:inline-block;padding:8px;">'
                f"{svg_content}"
                "</div></div>"
            )
            components.html(_html, height=620, scrolling=True)
        elif "bt_png" in results:
            img_width = max(200, int(900 * zoom_pct / 100))
            st.image(results["bt_png"], width=img_width)
        else:
            st.info("PNG/SVG visualization not available (graphviz may not be installed).")

        col_ascii, col_xml = st.columns(2)
        with col_ascii:
            st.markdown("**ASCII Tree**")
            st.code(results.get("bt_ascii", ""), language=None)
        with col_xml:
            with st.expander("behavior_tree.xml", expanded=False):
                st.code(results.get("bt_xml", ""), language="xml")
    else:
        st.info("Run the pipeline with Behavior Tree enabled to see output.")
