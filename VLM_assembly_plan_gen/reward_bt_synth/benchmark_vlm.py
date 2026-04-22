"""End-to-end VLM benchmark — Stage 1 (parts association) + C²SPE (plan
extraction) + our BT synthesis, on N items from main_data.json.

Unlike `benchmark.py` (ground-truth-only, deterministic), this module makes
live VLM calls against the recipe's configured default model. For each item:

  1. Load LangChain LLM via llm.model.load_llm_from_recipe(recipe, model_id)
     (default model_id: "gemini-pro" → Gemini 2.5 Pro).
  2. Stage 1 (inference.stage1_associate.select_materials_for_planning) →
     VLM extracts the parts inventory from the scene + manual pages.
     Output: raw JSON string listing parts.
  3. Parse the Stage 1 JSON to get parts = frozenset[int].
  4. C²SPE (reward_bt_synth.stage2_v2_plan.generate_plan_v2) → three VLM
     calls (enumerate, assign, optional repair). Result: StructuredPlan with
     connections + step_assignments covering every part.
  5. Orient each connection by degree → goal literals (matches
     ground_truth.connection_goals convention).
  6. bt_expand → BT; analyse_topology; extract_action_sequence; stochastic
     verify at p ∈ {1.0, 0.9, 0.7}.
  7. compare_with_ground_truth (VLM plan vs main_data connection_relation) —
     shows whether VLM-extracted connections match ground truth.
  8. Persist every intermediate + standards-compliant BT PNG/SVG/DOT.

Each item also records:
  - Stage 1 output (the raw VLM JSON)
  - Structured plan (full C²SPE output)
  - Token usage (via TokenUsageHandler singleton)
  - Wall-clock elapsed per stage
  - Any VLM-path failure (StructuredPlanError, ExpansionFailure, ...)

Per-item directory layout (under a timestamped root):
  reward_bt_outputs_vlm/<timestamp>/<Cat>/<name>/
    01_stage1_parts_inventory.json      (raw VLM output)
    02_structured_plan.json             (full C²SPE output)
    03_goals.json                       (oriented connection goals)
    04_initial_state.json               (s0)
    05_action_sequence.json
    06_topology.json
    07_verification.json
    08_comparison_vs_ground_truth.json
    09_baseline_stage4_stats.json
    bt.txt / bt.xml / bt.dot / bt.png / bt.svg

Runtime: each item takes ~30-90s with Gemini 2.5 Pro (Stage 1 ≈ 15s; each of
C²SPE's 2-3 calls ≈ 10-30s). For 50 items, expect 30-75 minutes.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[1]     # VLM_assembly_plan_gen/
_REPO_ROOT = _THIS.parents[2]    # repo root
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "inference"))

from reward_bt_synth import reward_specs  # noqa: E402
from reward_bt_synth.action_model import ActionModel, Literal  # noqa: E402
from reward_bt_synth.analysis import (  # noqa: E402
    analyse_topology,
    extract_action_sequence,
)
from reward_bt_synth.ast_extractor import extract_action_model  # noqa: E402
from reward_bt_synth.bt_expansion import BTNode, bt_expand  # noqa: E402
from reward_bt_synth.comparison import compare_with_ground_truth  # noqa: E402
from reward_bt_synth.exceptions import (  # noqa: E402
    ExpansionFailure,
    IncompleteCoverageError,
    RewardBTSynthError,
    StructuredPlanError,
)
from reward_bt_synth.ground_truth import initial_state_for  # noqa: E402
from reward_bt_synth.pytrees_build import render_unicode, to_btcpp_xml  # noqa: E402
from reward_bt_synth.stage2_v2_plan import generate_plan_v2  # noqa: E402
from reward_bt_synth.stochastic_verifier import stochastic_verify  # noqa: E402
from reward_bt_synth.viz import write_bt_dot, write_bt_png, write_bt_svg  # noqa: E402
from reward_bt_synth.vlm_client import RecipeVLMClient  # noqa: E402

# Stage 1 (VLM-based) comes from the existing inference/ package
from inference.config import DATA_DIR, SCENE_DIR, RECIPE_PATH  # noqa: E402
from inference.stage1_associate import select_materials_for_planning  # noqa: E402
from inference.utils import encode_image  # noqa: E402
from llm.model import load_llm_from_recipe  # noqa: E402
from llm.utils import TokenUsageHandler  # noqa: E402


# ---------- Stage 1 parsing ----------

_STAGE1_ID_KEYS = ("number", "label")  # prompt example uses "number"; accept "label" as alias


def _parse_parts_from_stage1(raw: str) -> frozenset[int]:
    """Extract integer part ids from the Stage 1 VLM JSON output.

    The prompts at `prompts/generate_json.txt` and `prompts/select_material.txt`
    specify entries of the form ``{"name": "...", "number": [<int>, ...]}``.
    Manual2Skill-flavour prompts use ``"label"`` instead — both are accepted.

    If the JSON parse fails or the structure is unexpected, we fall back to a
    regex that catches ``"number": [...]`` or ``"label": [...]`` so a single
    hallucinated key elsewhere doesn't abort the run.
    """
    if not raw:
        return frozenset()
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    parts: set[int] = set()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    ids = None
                    for key in _STAGE1_ID_KEYS:
                        if key in item:
                            ids = item[key]
                            break
                    if isinstance(ids, list):
                        for x in ids:
                            if isinstance(x, int):
                                parts.add(x)
                            elif isinstance(x, str) and x.strip().isdigit():
                                parts.add(int(x))
        if parts:
            return frozenset(parts)
    except json.JSONDecodeError:
        pass

    # Regex fallback: pull "number"/"label": [N, M, ...] and collect ints
    pattern = r'"(?:' + "|".join(_STAGE1_ID_KEYS) + r')"\s*:\s*\[([^\]]*)\]'
    for m in re.finditer(pattern, cleaned):
        for tok in m.group(1).split(","):
            tok = tok.strip().strip('"')
            if tok.isdigit():
                parts.add(int(tok))
    return frozenset(parts)


# ---------- Plan → oriented goals ----------

def _orient_connections_by_degree(
    parts: frozenset[int], connections: list[tuple[int, int]]
) -> list[Literal]:
    """Same degree-based leaf-into-hub rule as ground_truth.connection_goals.

    Each edge (a, b) → inserted(peg, socket) where peg has lower degree
    (ties broken by smaller id). This avoids peg reuse in the synthesised
    BT for tree / cycle-free topologies.
    """
    degree: Counter[int] = Counter()
    for a, b in connections:
        degree[int(a)] += 1
        degree[int(b)] += 1
    goals: list[Literal] = []
    for pair in connections:
        a, b = int(pair[0]), int(pair[1])
        if degree[a] < degree[b]:
            peg, socket = a, b
        elif degree[a] > degree[b]:
            peg, socket = b, a
        else:
            peg, socket = (a, b) if a < b else (b, a)
        goals.append(Literal("inserted", (peg, socket)))
    return goals


# ---------- Stage 4 baseline (same as benchmark.py) ----------

def _synthesize_actions_data_for_stage4(entry: dict, goals: list[Literal]) -> dict:
    all_connections = []
    for g in goals:
        peg, socket = int(g.args[0]), int(g.args[1])
        all_connections.append({
            "predicate": "inserted",
            "part1": peg,
            "part2": socket,
            "active_part": peg,
            "passive_part": socket,
            "direction": "downward",
            "needs_reorient": False,
            "reorient_description": "",
            "visual_description": f"part {peg} inserted into part {socket}",
        })
    steps = []
    for idx, g in enumerate(goals):
        peg, socket = int(g.args[0]), int(g.args[1])
        steps.append({
            "step_idx": idx,
            "all_parts": [peg, socket],
            "new_parts": [peg, socket] if idx == 0 else [peg],
            "subassembly_parts": [] if idx == 0 else [socket],
            "actions": [
                {"primitive": "GRASP", "part_id": peg},
                {
                    "primitive": "INSERT",
                    "active_part": peg,
                    "passive_part": socket,
                    "direction": "downward",
                },
            ],
        })
    return {
        "furniture": entry["name"],
        "category": entry["category"],
        "parts": sorted({int(p) for pair in
                         [(g.args[0], g.args[1]) for g in goals] for p in pair}),
        "all_connections": all_connections,
        "steps": steps,
    }


def _stage4_bt_stats(actions_data: dict) -> dict:
    inference_dir = str(_PKG_ROOT / "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    removed = None
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if getattr(mod, "__file__", None) and "inference" not in (mod.__file__ or ""):
            removed = sys.modules.pop("utils")
    try:
        if "stage4_formalize" in sys.modules:
            del sys.modules["stage4_formalize"]
        stage4 = importlib.import_module("stage4_formalize")
        xml_str = stage4._build_backchain_bt_xml(actions_data)
    finally:
        if removed is not None:
            sys.modules["utils"] = removed

    root = ET.fromstring(xml_str)
    tags = {"Sequence", "Fallback", "Parallel", "Action", "Condition"}
    count = 0
    max_depth = 0

    def walk(e, d):
        nonlocal count, max_depth
        if e.tag in tags:
            count += 1
            max_depth = max(max_depth, d)
        for c in e:
            walk(c, d + 1)

    walk(root, 0)
    return {"nodes": count, "depth": max_depth, "xml_length": len(xml_str)}


# ---------- Serialisation helpers ----------

def _action_model_to_dict(m: ActionModel) -> dict:
    return {
        "name": m.name,
        "params": list(m.params),
        "pre": sorted(str(l) for l in m.pre),
        "add": sorted(str(l) for l in m.add),
        "delete": sorted(str(l) for l in m.delete),
    }


def _state_to_list(s: frozenset[Literal]) -> list[str]:
    return sorted(str(l) for l in s)


def _verification_to_dict(report) -> dict:
    return {
        "passed": report.passed,
        "liveness_mean": report.liveness_mean,
        "liveness_95_ci": list(report.liveness_95_ci),
        "p1_gripper_consistency": report.p1_gripper_consistency,
        "p2_no_double_grasp": report.p2_no_double_grasp,
        "p3_hold_before_connect": report.p3_hold_before_connect,
        "p4_causal_completeness": report.p4_causal_completeness,
        "violations": report.violations,
        "n_rollouts": report.n_rollouts,
        "p_success": report.p_success,
    }


# ---------- Image loaders ----------

def _load_final_assembly_image(entry: dict) -> str:
    """Base64 of the labeled scene. Stage 1 uses scene_annotated.png."""
    scene_path = Path(SCENE_DIR) / entry["category"] / entry["name"] / "scene_annotated.png"
    if not scene_path.exists():
        raise FileNotFoundError(f"scene_annotated.png missing for {entry['name']}: {scene_path}")
    return encode_image(str(scene_path))


def _load_manual_pages(entry: dict) -> list[str]:
    """Base64 of manual pages, sorted naturally (page_1, page_2, ...)."""
    manual_dir = Path(DATA_DIR) / "pdfs" / entry["category"] / entry["name"]
    if not manual_dir.exists():
        raise FileNotFoundError(f"manual dir missing for {entry['name']}: {manual_dir}")
    pngs = sorted(
        [p for p in manual_dir.iterdir() if p.suffix == ".png"],
        key=lambda p: [
            int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)
        ],
    )
    if not pngs:
        raise FileNotFoundError(f"no .png manual pages found in {manual_dir}")
    return [encode_image(str(p)) for p in pngs]


# ---------- Per-item runner ----------

def _run_one(
    entry: dict,
    llm,
    vlm_client: RecipeVLMClient,
    lift_model: ActionModel,
    insert_model: ActionModel,
    output_root: Path,
    args: argparse.Namespace,
) -> dict:
    name = entry["name"]
    category = entry["category"]
    parts_ct = int(entry["parts_ct"])
    item_dir = output_root / category / name
    item_dir.mkdir(parents=True, exist_ok=True)

    t_total0 = time.time()

    row: dict = {
        "name": name,
        "category": category,
        "parts_ct": parts_ct,
        "stage1_ok": False,
        "stage1_parts_found": "",
        "c2spe_ok": False,
        "c2spe_connections_found": "",
        "c2spe_steps": "",
        "expansion_ok": False,
        "connection_set_match": "",
        "missing_connections": "",
        "extra_connections": "",
        "precedence_violations": "",
        "baseline_nodes": "",
        "ours_nodes": "",
        "ours_depth": "",
        "ppa_count": "",
        "p1_p2_p3_p4_all_pass": "",
        "liveness_p1.0": "",
        "liveness_p0.9": "",
        "liveness_p0.7": "",
        "stage1_elapsed_s": "",
        "c2spe_elapsed_s": "",
        "synth_elapsed_s": "",
        "total_elapsed_s": "",
        "vlm_calls": "",
        "error": "",
    }

    # Extracted models persisted once per run (but keep a copy per item for
    # auditability).
    (item_dir / "00_extracted_models.json").write_text(
        json.dumps(
            {
                "lift": _action_model_to_dict(lift_model),
                "insert": _action_model_to_dict(insert_model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------- Stage 1 (VLM parts association) --------
    t_s1 = time.time()
    try:
        stage1_raw = select_materials_for_planning(
            name, category, str(output_root), args, llm
        )
        # Persist the raw VLM output BEFORE parsing, so a parse failure leaves
        # the artefact on disk for diagnosis.
        (item_dir / "01_stage1_parts_inventory.json").write_text(
            (stage1_raw or "").replace("```json", "").replace("```", "").strip(),
            encoding="utf-8",
        )
        parts = _parse_parts_from_stage1(stage1_raw)
        if not parts:
            raise RuntimeError(
                f"Stage 1 returned no parseable part ids "
                f"(raw length={len(stage1_raw or '')})"
            )
        row["stage1_ok"] = True
        row["stage1_parts_found"] = len(parts)
    except Exception as exc:
        row["error"] = f"stage1: {exc}"
        row["stage1_elapsed_s"] = round(time.time() - t_s1, 3)
        row["total_elapsed_s"] = round(time.time() - t_total0, 3)
        return row
    row["stage1_elapsed_s"] = round(time.time() - t_s1, 3)

    # -------- C²SPE (three-call plan extraction) --------
    t_c2 = time.time()
    try:
        final_image_b64 = _load_final_assembly_image(entry)
        manual_pages_b64 = _load_manual_pages(entry)

        # Give Call 1 (enumerate connections) more context than just the
        # labeled pre-assembly scene: append all manual pages so the VLM can
        # observe the final-assembly structure that the last few pages depict.
        call1_context = [final_image_b64] + manual_pages_b64

        plan = generate_plan_v2(
            vlm=vlm_client,
            parts=parts,
            final_assembly_image_b64=call1_context,
            manual_page_images_b64=manual_pages_b64,
            max_enum_retries=1,
        )
        row["c2spe_ok"] = True
        row["c2spe_connections_found"] = len(plan.connections)
        row["c2spe_steps"] = plan.num_steps
        (item_dir / "02_structured_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2), encoding="utf-8"
        )
    except StructuredPlanError as exc:
        row["error"] = f"c2spe: {exc}"
        row["c2spe_elapsed_s"] = round(time.time() - t_c2, 3)
        row["total_elapsed_s"] = round(time.time() - t_total0, 3)
        return row
    except Exception as exc:
        row["error"] = f"c2spe_unexpected: {exc}"
        row["c2spe_elapsed_s"] = round(time.time() - t_c2, 3)
        row["total_elapsed_s"] = round(time.time() - t_total0, 3)
        return row
    row["c2spe_elapsed_s"] = round(time.time() - t_c2, 3)

    # -------- BT synthesis + verification + comparison --------
    t_synth = time.time()
    try:
        goals = _orient_connections_by_degree(parts, list(plan.connections))
        s0 = initial_state_for(entry)
        (item_dir / "03_goals.json").write_text(
            json.dumps(
                [{"name": g.name, "args": list(g.args)} for g in goals],
                indent=2,
            ),
            encoding="utf-8",
        )
        (item_dir / "04_initial_state.json").write_text(
            json.dumps(_state_to_list(s0), indent=2), encoding="utf-8"
        )

        actions_data = _synthesize_actions_data_for_stage4(entry, goals)
        baseline = _stage4_bt_stats(actions_data)
        row["baseline_nodes"] = baseline["nodes"]
        (item_dir / "09_baseline_stage4_stats.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )

        tree = bt_expand(frozenset(goals), [lift_model, insert_model], s0)
        row["expansion_ok"] = True

        topo = analyse_topology(tree)
        row["ours_nodes"] = topo.total_nodes
        row["ours_depth"] = topo.max_depth
        row["ppa_count"] = topo.ppa_pattern_count
        (item_dir / "06_topology.json").write_text(
            json.dumps(topo.to_dict(), indent=2), encoding="utf-8"
        )

        action_seq = extract_action_sequence(tree, s0)
        (item_dir / "05_action_sequence.json").write_text(
            json.dumps(
                [{"name": a.name, "args": list(a.args), "step": a.at_step_index}
                 for a in action_seq],
                indent=2,
            ),
            encoding="utf-8",
        )

        comp = compare_with_ground_truth(action_seq, entry)
        row["connection_set_match"] = comp.connection_set_match
        row["missing_connections"] = len(comp.missing_bt_connections)
        row["extra_connections"] = len(comp.extra_bt_connections)
        row["precedence_violations"] = len(comp.precedence_violations)
        (item_dir / "08_comparison_vs_ground_truth.json").write_text(
            json.dumps(comp.to_dict(), indent=2), encoding="utf-8"
        )

        (item_dir / "bt.txt").write_text(render_unicode(tree), encoding="utf-8")
        (item_dir / "bt.xml").write_text(to_btcpp_xml(tree), encoding="utf-8")
        title = f"{category}/{name} (parts_ct={parts_ct}) [VLM]"
        write_bt_dot(tree, item_dir / "bt", title=title)
        write_bt_png(tree, item_dir / "bt", title=title, dpi=100)
        write_bt_svg(tree, item_dir / "bt", title=title)

        all_ok = True
        for p in (1.0, 0.9, 0.7):
            report = stochastic_verify(tree, s0, p_success=p, n_rollouts=500, seed=42)
            row[f"liveness_p{p}"] = f"{report.liveness_mean:.3f}"
            if p == 1.0:
                all_ok = all([
                    report.p1_gripper_consistency,
                    report.p2_no_double_grasp,
                    report.p3_hold_before_connect,
                    report.p4_causal_completeness,
                ])
                (item_dir / "07_verification.json").write_text(
                    json.dumps(_verification_to_dict(report), indent=2),
                    encoding="utf-8",
                )
        row["p1_p2_p3_p4_all_pass"] = all_ok
    except ExpansionFailure as exc:
        row["error"] = f"expansion: {exc}"
    except RewardBTSynthError as exc:
        row["error"] = f"synthesis: {exc}"
    except Exception as exc:
        row["error"] = f"synth_unexpected: {exc}\n{traceback.format_exc()}"
    row["synth_elapsed_s"] = round(time.time() - t_synth, 3)
    row["total_elapsed_s"] = round(time.time() - t_total0, 3)
    row["vlm_calls"] = vlm_client.call_count
    return row


# ---------- Item selection + summary writers ----------

def _select_items(main_data: list[dict], n: int) -> list[dict]:
    valid = [e for e in main_data if int(e.get("parts_ct", 0)) >= 2 and e.get("connection_relation")]
    sorted_items = sorted(valid, key=lambda e: (int(e["parts_ct"]), e.get("name", "")))
    seen_parts: set[int] = set()
    primary: list[dict] = []
    for item in sorted_items:
        pc = int(item["parts_ct"])
        if pc in seen_parts:
            continue
        seen_parts.add(pc)
        primary.append(item)
        if len(primary) >= n:
            break
    seen_names = {e["name"] + "_" + e["category"] for e in primary}
    for item in sorted_items:
        key = item["name"] + "_" + item["category"]
        if key in seen_names:
            continue
        if len(primary) >= n:
            break
        primary.append(item)
        seen_names.add(key)
    return primary[:n]


_COLUMNS = [
    "name", "category", "parts_ct",
    "stage1_ok", "stage1_parts_found",
    "c2spe_ok", "c2spe_connections_found", "c2spe_steps",
    "expansion_ok",
    "connection_set_match", "missing_connections", "extra_connections", "precedence_violations",
    "ppa_count",
    "baseline_nodes", "ours_nodes", "ours_depth",
    "p1_p2_p3_p4_all_pass",
    "liveness_p1.0", "liveness_p0.9", "liveness_p0.7",
    "stage1_elapsed_s", "c2spe_elapsed_s", "synth_elapsed_s", "total_elapsed_s",
    "vlm_calls", "error",
]


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in _COLUMNS})


def _write_md(rows: list[dict], path: Path) -> None:
    hdr = "| " + " | ".join(_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLUMNS) + " |"
    out = [hdr, sep]
    for row in rows:
        vals = []
        for c in _COLUMNS:
            v = row.get(c, "")
            if isinstance(v, bool):
                v = "yes" if v else "no"
            vals.append(str(v) if v != "" else "-")
        out.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(out), encoding="utf-8")


def _write_aggregate(rows: list[dict], path: Path, total_elapsed: float, model_id: str) -> None:
    def _bool(v) -> bool:
        return isinstance(v, bool) and v

    def _flt(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    total = len(rows)
    stage1_ok = sum(1 for r in rows if _bool(r.get("stage1_ok")))
    c2spe_ok = sum(1 for r in rows if _bool(r.get("c2spe_ok")))
    expansion_ok = sum(1 for r in rows if _bool(r.get("expansion_ok")))
    set_match_ok = sum(1 for r in rows if _bool(r.get("connection_set_match")))
    all_props = sum(1 for r in rows if _bool(r.get("p1_p2_p3_p4_all_pass")))

    successful = [r for r in rows if _bool(r.get("expansion_ok"))]

    def _mean(key):
        xs = [_flt(r.get(key)) for r in successful]
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    agg = {
        "model": model_id,
        "total_items": total,
        "stage1_success": stage1_ok,
        "c2spe_success": c2spe_ok,
        "expansion_success": expansion_ok,
        "connection_set_match_count": set_match_ok,
        "connection_set_match_rate": (set_match_ok / expansion_ok) if expansion_ok else None,
        "structural_properties_all_pass_count": all_props,
        "structural_properties_all_pass_rate": (all_props / expansion_ok) if expansion_ok else None,
        "mean_missing_connections": _mean("missing_connections"),
        "mean_extra_connections": _mean("extra_connections"),
        "mean_precedence_violations": _mean("precedence_violations"),
        "mean_liveness_p1.0": _mean("liveness_p1.0"),
        "mean_liveness_p0.9": _mean("liveness_p0.9"),
        "mean_liveness_p0.7": _mean("liveness_p0.7"),
        "mean_stage1_elapsed_s": _mean("stage1_elapsed_s"),
        "mean_c2spe_elapsed_s": _mean("c2spe_elapsed_s"),
        "mean_synth_elapsed_s": _mean("synth_elapsed_s"),
        "total_elapsed_s": round(total_elapsed, 2),
    }
    path.write_text(json.dumps(agg, indent=2), encoding="utf-8")


# ---------- Entry point ----------

def run_vlm_benchmark(
    main_data_path: str,
    output_root: str,
    n_items: int,
    model_id: str,
    args: argparse.Namespace,
) -> Path:
    with open(main_data_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    items = _select_items(main_data, n=n_items)
    print(f"[vlm_bench] {len(items)} items, parts_ct "
          f"{min(int(i['parts_ct']) for i in items)}..{max(int(i['parts_ct']) for i in items)}")

    # LLM is used by Stage 1 directly (raw invoke_multimodal) and by C²SPE
    # via the RecipeVLMClient wrapper. Both hit the same model.
    llm = load_llm_from_recipe(str(RECIPE_PATH), model_id)
    vlm_client = RecipeVLMClient(recipe_path=str(RECIPE_PATH), model_id=model_id)
    print(f"[vlm_bench] model={model_id} -> {type(llm).__name__}")

    lift_model = extract_action_model("lift", reward_specs)
    insert_model = extract_action_model("insert", reward_specs)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    root = Path(output_root) / timestamp
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_global = time.time()
    for i, item in enumerate(items, 1):
        print(f"[vlm_bench {i:>3}/{len(items)}] {item['category']}/{item['name']} "
              f"(parts_ct={item['parts_ct']})  ...", end="", flush=True)
        row = _run_one(item, llm, vlm_client, lift_model, insert_model, root, args)
        rows.append(row)
        status = "OK" if row["expansion_ok"] else (
            "stage1-FAIL" if not row["stage1_ok"] else
            ("c2spe-FAIL" if not row["c2spe_ok"] else "synth-FAIL")
        )
        print(f"  {status}  [{row['total_elapsed_s']}s, calls={row['vlm_calls']}]")
        # Flush the CSV after every item so a crash doesn't lose progress
        _write_csv(rows, root / "benchmark_summary.csv")

    total_elapsed = time.time() - t_global
    _write_csv(rows, root / "benchmark_summary.csv")
    _write_md(rows, root / "benchmark_summary.md")
    _write_aggregate(rows, root / "benchmark_aggregate.json", total_elapsed, model_id)

    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "n_items_requested": n_items,
                "n_items_actual": len(items),
                "main_data_path": main_data_path,
                "model_id": model_id,
                "llm_class": type(llm).__name__,
                "verifier_n_rollouts": 500,
                "verifier_seed": 42,
                "p_success_levels": [1.0, 0.9, 0.7],
                "token_usage": TokenUsageHandler().get_usage_summary(),
                "total_elapsed_s": round(total_elapsed, 2),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[vlm_bench] summary: {root / 'benchmark_summary.md'}")
    print(f"[vlm_bench] aggregate: {root / 'benchmark_aggregate.json'}")
    print(f"[vlm_bench] metadata : {root / 'run_metadata.json'}")
    print(f"[vlm_bench] tokens   : {TokenUsageHandler().get_usage_summary()}")
    print(f"[vlm_bench] elapsed  : {total_elapsed:.1f}s "
          f"({total_elapsed / max(1, len(items)):.1f}s/item)")
    return root


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--main_data", default=str(_PKG_ROOT / "data" / "main_data.json"))
    p.add_argument("--output_root", default=str(_PKG_ROOT / "reward_bt_outputs_vlm"))
    p.add_argument("--items", type=int, default=50)
    p.add_argument("--model", default="gemini-pro",
                   help="recipe model id (default gemini-pro = Gemini 2.5 Pro)")
    # Stage 1 expects these as attributes on `args`:
    p.add_argument("--scene_type", default="original")
    p.add_argument("--prompt_type", default="numbered")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _build_args()
    run_vlm_benchmark(
        main_data_path=args.main_data,
        output_root=args.output_root,
        n_items=args.items,
        model_id=args.model,
        args=args,
    )
