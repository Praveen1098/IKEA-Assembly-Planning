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


def _plotly_download_buttons(fig, basename: str) -> None:
    """Render PNG + HTML download buttons for a Plotly figure."""
    _c1, _c2 = st.columns(2)
    try:
        _png = fig.to_image(format="png", scale=2)
        _c1.download_button(
            "⬇ Download PNG",
            data=_png,
            file_name=f"{basename}.png",
            mime="image/png",
            key=f"dl_png_{basename}",
        )
    except Exception:
        _c1.caption("PNG download requires `kaleido` (`pip install kaleido`)")
    _html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    _c2.download_button(
        "⬇ Download HTML (interactive)",
        data=_html,
        file_name=f"{basename}.html",
        mime="text/html",
        key=f"dl_html_{basename}",
    )


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
    run_stage3        = st.checkbox("Stage 3 — Actions",       value=True)
    run_pddl          = st.checkbox("Stage 4 — PDDL",          value=True)
    run_bt            = st.checkbox("Stage 4 — Behavior Tree", value=True)
    run_pddl_validate = st.checkbox("Stage 3.5 — PDDL Validate", value=False)
    debug_mode        = st.checkbox("Debug mode",               value=False)

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
        validate_pddl=run_pddl_validate,
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
                    os.path.join(item_dir, "behavior_tree.txt"), encoding="utf-8"
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

tab_inputs, tab_s1, tab_s2, tab_tree, tab_s3, \
tab_pddl, tab_pddl_val, tab_bt, tab_bt_ltl = st.tabs([
    "Inputs",
    "Stage 1 — Parts",
    "Stage 2 — Plan",
    "Assembly Tree",
    "Stage 3 — Actions",
    "PDDL",
    "PDDL Validation",
    "Behavior Tree",
    "BT Verification",
])

# ── Tab: Inputs ───────────────────────────────────────────────────────────────
with tab_inputs:
    col_scene, col_gt = st.columns([1, 1])

    with col_scene:
        st.subheader("Scene Image")
        scene_p = scene_image_path(disp_cat, disp_name, scene_type)
        if os.path.exists(scene_p):
            st.image(scene_p, width='stretch')
            with open(scene_p, "rb") as _f:
                st.download_button(
                    "⬇ Download Scene Image",
                    data=_f.read(),
                    file_name=f"{disp_name}_scene.png",
                    mime="image/png",
                    key="dl_scene_img",
                )
        else:
            st.warning(f"Scene image not found: `{scene_p}`")

        st.subheader("Manual Pages")
        pages = list_manual_pages(disp_cat, disp_name)
        if pages:
            cols = st.columns(min(len(pages), 4))
            for i, pg in enumerate(pages):
                with cols[i % 4]:
                    st.image(pg, caption=os.path.basename(pg), width='stretch')
                    with open(pg, "rb") as _f:
                        st.download_button(
                            "⬇ Download",
                            data=_f.read(),
                            file_name=os.path.basename(pg),
                            mime="image/png",
                            key=f"dl_page_{i}",
                        )
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

    gt_tree        = data[disp_idx].get("assembly_tree")
    predicted_tree = results.get("tree")
    _has_gt        = gt_tree is not None
    _has_pred      = predicted_tree is not None

    if _has_gt or _has_pred:
        from app_viz import make_plotly_assembly_tree

        if _has_pred:
            try:
                pred_root = _build_assembly_tree(predicted_tree, [0], [0], disp_cat, disp_name)
                _pred_fig = make_plotly_assembly_tree(pred_root, "VLM Predicted Assembly Graph")
                st.plotly_chart(_pred_fig, use_container_width=True)
                _plotly_download_buttons(_pred_fig, f"{disp_name}_predicted_tree")
            except Exception as _e:
                st.error(f"Predicted graph unavailable: {_e}")
                st.code(render_tree(predicted_tree).strip(), language=None)

        if _has_gt:
            try:
                gt_root = _build_assembly_tree(gt_tree, [0], [0], disp_cat, disp_name)
                _gt_fig = make_plotly_assembly_tree(gt_root, "Ground Truth Assembly Graph")
                st.plotly_chart(_gt_fig, use_container_width=True)
                _plotly_download_buttons(_gt_fig, f"{disp_name}_gt_tree")
            except Exception as _e:
                st.error(f"Ground truth graph unavailable: {_e}")

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
    # ── 1. Generated domain / problem code (if pipeline ran) ──
    if "pddl_error" in results:
        st.error(f"PDDL generation failed: {results['pddl_error']}")
    elif "domain_pddl" in results or "problem_pddl" in results:
        col_d, col_p = st.columns(2)
        with col_d:
            st.subheader("domain.pddl")
            st.code(results.get("domain_pddl", ""), language="lisp")
            st.download_button(
                "⬇ Download domain.pddl",
                data=results.get("domain_pddl", ""),
                file_name=f"{last[1]}_domain.pddl",
                mime="text/plain",
                key="dl_domain_pddl",
            )
        with col_p:
            st.subheader("problem.pddl")
            st.code(results.get("problem_pddl", ""), language="lisp")
            st.download_button(
                "⬇ Download problem.pddl",
                data=results.get("problem_pddl", ""),
                file_name=f"{last[1]}_problem.pddl",
                mime="text/plain",
                key="dl_problem_pddl",
            )
    else:
        st.info("Run the pipeline with PDDL enabled to see generated files.")

    st.divider()

    # ── 2. Skill Library (always visible — no pipeline run required) ──
    st.subheader("Skill Library")
    st.caption("PDDL-style preconditions and postconditions for each robot primitive")
    try:
        import pandas as pd
        from stage3_5_pddl_validate import SKILL_LIBRARY
        rows = [
            {
                "Primitive": k,
                "Preconditions":   " ∧ ".join(v["pre"]),
                "Postconditions":  " ∧ ".join(v["post"]),
            }
            for k, v in SKILL_LIBRARY.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except Exception as _e:
        st.warning(f"Could not load Skill Library: {_e}")

    st.divider()

    # ── 3. Precondition / Effect graph ──
    st.subheader("Precondition / Effect Graph")
    st.caption("Arrows show which predicates each action requires (left) and produces (right)")
    try:
        from app_viz import make_pddl_skill_graph
        _skill_fig = make_pddl_skill_graph()
        st.plotly_chart(_skill_fig, use_container_width=True)
        _plotly_download_buttons(_skill_fig, "pddl_skill_graph")
    except Exception as _e:
        st.warning(f"Could not render skill graph: {_e}")

# ── Tab: PDDL Validation ──────────────────────────────────────────────────────
with tab_pddl_val:
    st.subheader("PDDL Consistency Validation")
    st.caption(
        "Validates that actions.json is logically executable: "
        "PyPerPlan checks that every connection goal is reachable "
        "from the initial state (all parts accessible, gripper empty)."
    )

    _actions_available = "actions" in results
    _output_path = results.get("output_path", "")
    _last_cat, _last_name = last[0], last[1]

    # Allow validating a previously-generated actions.json even without re-running
    _actions_path = os.path.join(_output_path, _last_cat, _last_name, "actions.json") \
        if _output_path else ""
    _file_exists = os.path.isfile(_actions_path)

    if not _actions_available and not _file_exists:
        st.info("Run the pipeline with Stage 3 enabled, then click Validate.")
    else:
        if _file_exists:
            st.caption(f"Validating: `{_actions_path}`")

        if st.button("▶ Run PDDL Validation", key="btn_pddl_validate",
                     disabled=not _file_exists):
            with st.spinner("Running PyPerPlan …"):
                try:
                    import tempfile
                    from stage3_5_pddl_validate import (
                        generate_pddl_domain, generate_pddl_problem,
                        _run_pyperplan,
                    )
                    import json as _json
                    with open(_actions_path) as _f:
                        _act_data = _json.load(_f)
                    _parts = _act_data.get("parts", [])
                    _conns = _act_data.get("all_connections", [])

                    _dom = generate_pddl_domain()
                    _prob = generate_pddl_problem(_parts, _conns)

                    with tempfile.TemporaryDirectory() as _td:
                        _df = os.path.join(_td, "domain.pddl")
                        _pf = os.path.join(_td, "problem.pddl")
                        with open(_df, "w") as _fh:
                            _fh.write(_dom)
                        with open(_pf, "w") as _fh:
                            _fh.write(_prob)
                        _plan = _run_pyperplan(_df, _pf)

                    if _plan is not None:
                        _plan_strs = [str(op) for op in _plan]
                        st.session_state["pddl_val_result"] = {
                            "valid": True,
                            "furniture": _act_data.get("furniture", ""),
                            "n_steps": len(_plan),
                            "plan": _plan_strs,
                        }
                    else:
                        import pandas as _pd
                        _diag_rows = []
                        for c in _conns:
                            pred = c.get("predicate", "")
                            _diag_rows.append({
                                "Connection": f"Part {c.get('part1')} → Part {c.get('part2')}",
                                "Predicate": pred,
                                "Achievable": pred in ("inserted", "pressed", "screwed"),
                            })
                        st.session_state["pddl_val_result"] = {
                            "valid": False,
                            "furniture": _act_data.get("furniture", ""),
                            "diagnosis": _diag_rows,
                        }
                except Exception as _exc:
                    st.error(f"Validation error: {_exc}")

        _val = st.session_state.get("pddl_val_result")
        if _val:
            if _val["valid"]:
                st.success(
                    f"✓ Valid — plan found ({_val['n_steps']} actions) "
                    f"for **{_val['furniture']}**"
                )
                try:
                    from app_viz import make_plotly_plan_sequence
                    _plan_fig = make_plotly_plan_sequence(_val["plan"])
                    st.plotly_chart(_plan_fig, use_container_width=True)
                    _plotly_download_buttons(_plan_fig, f"{_val.get('furniture', 'plan')}_sequence")
                except Exception as _e:
                    st.code("\n".join(_val["plan"]), language=None)
            else:
                st.error(f"✗ No valid plan found for **{_val['furniture']}**")
                if "diagnosis" in _val:
                    import pandas as _pd
                    st.markdown("**Connection goal diagnosis:**")
                    st.dataframe(
                        _pd.DataFrame(_val["diagnosis"]),
                        use_container_width=True, hide_index=True,
                    )

# ── Tab: Behavior Tree ────────────────────────────────────────────────────────
with tab_bt:
    import streamlit.components.v1 as components

    if "bt_error" in results:
        st.error(f"Behavior Tree generation failed: {results['bt_error']}")
    elif "bt_xml" in results or "actions" in results:
        # ── Causal dependency graph ────────────────────────────────────────────
        if "actions" in results:
            st.subheader("Causal Dependency Graph")
            st.caption(
                "Nodes = connection goals. "
                "Edges = causal dependencies between connections (hover an edge for the rule). "
                "Same color = same parallel group → compiled into a BT **Parallel** node. "
                "Different groups = independent sub-tasks that can run concurrently."
            )
            try:
                from app_viz import make_causal_graph_figure
                _cg_fig = make_causal_graph_figure(results["actions"])
                st.plotly_chart(_cg_fig, use_container_width=True)
                _plotly_download_buttons(_cg_fig, f"{last[1]}_causal_graph")
            except Exception as _e:
                st.warning(f"Causal graph unavailable: {_e}")
            st.divider()

    if "bt_xml" in results:
        st.subheader("Behavior Tree — Interactive View")
        st.caption(
            "Color key: "
            "🟦 Sequence  🔴 Fallback  🟣 Parallel  🟢 Condition  🟠 Action  ⬛ Root  "
            "— hover a node to see its attributes"
        )

        try:
            from app_viz import bt_xml_to_visjs_html
            _visjs_html = bt_xml_to_visjs_html(results["bt_xml"])
            components.html(_visjs_html, height=680, scrolling=False)
            st.download_button(
                "⬇ Download BT Interactive HTML",
                data=_visjs_html,
                file_name=f"{last[1]}_behavior_tree.html",
                mime="text/html",
                key="dl_bt_visjs",
            )
        except Exception as _e:
            st.warning(f"vis.js render failed ({_e}); falling back to ASCII.")

        # BT image files (PNG / SVG) from pipeline output
        _bt_img_cols = []
        if results.get("bt_png") and os.path.exists(results["bt_png"]):
            _bt_img_cols.append(("bt_png", results["bt_png"], "image/png", "⬇ Download BT PNG"))
        if results.get("bt_svg") and os.path.exists(results["bt_svg"]):
            _bt_img_cols.append(("bt_svg", results["bt_svg"], "image/svg+xml", "⬇ Download BT SVG"))
        if _bt_img_cols:
            _img_dl_cols = st.columns(len(_bt_img_cols))
            for _col, (_key, _path, _mime, _label) in zip(_img_dl_cols, _bt_img_cols):
                with open(_path, "rb") as _f:
                    _col.download_button(
                        _label, data=_f.read(),
                        file_name=os.path.basename(_path),
                        mime=_mime, key=f"dl_{_key}",
                    )

        st.divider()
        col_ascii, col_xml = st.columns(2)
        with col_ascii:
            st.markdown("**ASCII Tree**")
            st.code(results.get("bt_ascii", ""), language=None)
            st.download_button(
                "⬇ Download ASCII Tree",
                data=results.get("bt_ascii", ""),
                file_name=f"{last[1]}_behavior_tree.txt",
                mime="text/plain",
                key="dl_bt_ascii",
            )
        with col_xml:
            with st.expander("behavior_tree.xml", expanded=False):
                st.code(results.get("bt_xml", ""), language="xml")
            st.download_button(
                "⬇ Download BT XML",
                data=results.get("bt_xml", ""),
                file_name=f"{last[1]}_behavior_tree.xml",
                mime="application/xml",
                key="dl_bt_xml",
            )
    else:
        st.info("Run the pipeline with Behavior Tree enabled to see output.")

# ── Tab: BT Verification ──────────────────────────────────────────────────────
with tab_bt_ltl:
    st.subheader("Behavior Tree — LTL Property Verification")
    st.caption(
        "Uses BehaVerify (SEFM 2022) to convert the BT to a nuXmv model "
        "and verify Linear Temporal Logic safety / liveness properties. "
        "Requires the nuXmv binary in PATH for full model checking."
    )

    # ── 1. Predefined LTL properties (always visible) ──
    st.markdown("**Standard Assembly Safety Properties**")
    try:
        from app_viz import get_ltl_properties
        _ltl_props = get_ltl_properties()
        for _prop_name, _ltl_formula in _ltl_props.items():
            col_n, col_f = st.columns([1, 2])
            col_n.markdown(f"*{_prop_name}*")
            col_f.code(_ltl_formula, language=None)
    except Exception as _e:
        st.warning(f"Could not load LTL properties: {_e}")

    st.divider()

    # ── 2. nuXmv model display ──
    st.markdown("**Generated nuXmv Model**")
    if "bt_xml" not in results:
        st.info("Run the pipeline with Behavior Tree enabled first.")
    else:
        try:
            from app_viz import bt_to_nuxmv_model
            _smv = bt_to_nuxmv_model(results["bt_xml"])
            if _smv:
                st.code(_smv, language=None)
                st.download_button(
                    "⬇ Download .smv",
                    data=_smv,
                    file_name=f"{last[1]}_bt.smv",
                    mime="text/plain",
                )
            else:
                st.info(
                    "nuXmv model generation requires the `behaverify` package "
                    "and its metamodel files to be present. "
                    "Run: `pip install behaverify` and check the installation."
                )
        except Exception as _e:
            st.warning(f"nuXmv model generation failed: {_e}")

    st.divider()

    # ── 3. BehaVerify .tree DSL preview (always, if bt_xml available) ──
    if "bt_xml" in results:
        with st.expander("BehaVerify .tree DSL (intermediate representation)", expanded=False):
            try:
                from app_viz import _bt_xml_to_behaverify_dsl
                _dsl = _bt_xml_to_behaverify_dsl(results["bt_xml"])
                st.code(_dsl or "Conversion failed.", language=None)
            except Exception as _e:
                st.warning(f"DSL conversion failed: {_e}")
