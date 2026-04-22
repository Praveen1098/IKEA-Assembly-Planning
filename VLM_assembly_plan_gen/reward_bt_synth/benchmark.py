"""50-item benchmark — per-stage artefacts, BT visualisation, and
ground-truth comparison.

For each item:
  1. Read its entry from main_data.json
  2. Run ground-truth → oriented connection goals + initial state s0
  3. Extract Lift / Insert STRIPS models via ast_extractor (once, cached)
  4. bt_expand → internal BTNode tree
  5. analyse_topology → structural metrics (PPA count, McCabe, balance, etc.)
  6. stochastic_verify at p ∈ {1.0, 0.9, 0.7}
  7. extract_action_sequence under s0 → deterministic Tick trace
  8. compare_with_ground_truth → connection-set match, precedence violations,
     per-gt-step alignment
  9. Invoke Stage 4's `_build_backchain_bt_xml` as baseline for node counts
  10. Render BT as .dot, .png, .svg (Groot2-style), .xml (BT.CPP v4), .txt (unicode)
  11. Persist every intermediate as JSON under the item's directory
  12. Aggregate a summary CSV + markdown

Directory layout per item:
  reward_bt_outputs/<timestamp>/<Category>/<name>/
    01_extracted_models.json
    02_goals.json
    03_initial_state.json
    04_action_sequence.json
    05_topology.json
    06_verification.json
    07_comparison_vs_ground_truth.json
    08_baseline_stage4_stats.json
    bt.txt             (unicode tree)
    bt.xml             (BT.CPP v4 XML, Groot2-compatible)
    bt.dot             (Graphviz source)
    bt.png             (Groot2-style rendered BT)
    bt.svg             (vector form)

Plus at the root:
  benchmark_summary.csv
  benchmark_summary.md
  run_metadata.json    (timestamp, item count, seed, model_id)
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))  # repo root
sys.path.insert(0, str(_THIS.parents[1]))  # VLM_assembly_plan_gen/

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
)
from reward_bt_synth.ground_truth import connection_goals, initial_state_for  # noqa: E402
from reward_bt_synth.pytrees_build import render_unicode, to_btcpp_xml  # noqa: E402
from reward_bt_synth.stochastic_verifier import stochastic_verify  # noqa: E402
from reward_bt_synth.viz import write_bt_dot, write_bt_png, write_bt_svg  # noqa: E402


# ---------- Item selection ----------

def _select_items(main_data: list[dict], n: int) -> list[dict]:
    """Return `n` entries from main_data.json with parts_ct ≥ 2.

    Strategy: first, one item per distinct parts_ct (ascending) — gives
    coverage across the difficulty spectrum. Then top up from the smallest
    parts_ct buckets until we hit n, avoiding name duplicates.
    """
    valid = [
        e for e in main_data
        if int(e.get("parts_ct", 0)) >= 2 and e.get("connection_relation")
    ]
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


# ---------- Stage 4 baseline (PDDL-free XML builder) ----------

def _synthesize_actions_data_for_stage4(entry: dict, oriented_goals: list[Literal]) -> dict:
    """Minimal actions_data compatible with `stage4_formalize._validate_actions_data`.

    Used only to invoke Stage 4's XML builder for the head-to-head node-count
    comparison. Does NOT invoke stage3_5 PDDL anywhere.
    """
    all_connections = []
    for g in oriented_goals:
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
    for idx, g in enumerate(oriented_goals):
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
        "parts": list(range(int(entry["parts_ct"]))),
        "all_connections": all_connections,
        "steps": steps,
    }


def _stage4_bt_stats(actions_data: dict) -> dict:
    """Invoke stage4_formalize's XML builder directly; count nodes + depth.

    Works around the fact that `from utils import ensure_dir` in stage4
    resolves ambiguously (VLM_assembly_plan_gen/utils/ is a package, and
    inference/utils.py is a module with ensure_dir). We prepend inference/
    to sys.path so utils resolves to the module.
    """
    inference_dir = str(_THIS.parents[1] / "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    removed_utils = None
    if "utils" in sys.modules:
        mod = sys.modules["utils"]
        if getattr(mod, "__file__", None) and "inference" not in (mod.__file__ or ""):
            removed_utils = sys.modules.pop("utils")
    try:
        if "stage4_formalize" in sys.modules:
            del sys.modules["stage4_formalize"]
        stage4 = importlib.import_module("stage4_formalize")
        xml_str = stage4._build_backchain_bt_xml(actions_data)
    finally:
        if removed_utils is not None:
            sys.modules["utils"] = removed_utils

    root = ET.fromstring(xml_str)
    tags = {"Sequence", "Fallback", "Parallel", "Action", "Condition"}
    count = 0
    max_depth = 0

    def walk(elem, depth):
        nonlocal count, max_depth
        if elem.tag in tags:
            count += 1
            max_depth = max(max_depth, depth)
        for c in elem:
            walk(c, depth + 1)

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
        "cost": m.cost,
    }


def _goals_to_dict(goals: list[Literal]) -> list[dict]:
    return [{"name": g.name, "args": list(g.args)} for g in goals]


def _state_to_dict(s: frozenset[Literal]) -> list[str]:
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


# ---------- Per-item runner ----------

def _run_one(
    entry: dict,
    lift_model: ActionModel,
    insert_model: ActionModel,
    output_root: Path,
) -> dict:
    name = entry["name"]
    category = entry["category"]
    parts_ct = int(entry["parts_ct"])
    item_dir = output_root / category / name
    item_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    row: dict = {
        "name": name,
        "category": category,
        "parts_ct": parts_ct,
        "extraction_ok": True,
        "expansion_ok": False,
        "coverage_ok": False,
        "connection_set_match": "",
        "precedence_violations": "",
        "ppa_count": "",
        "mccabe": "",
        "balance_ratio": "",
        "baseline_nodes": "",
        "baseline_depth": "",
        "ours_nodes": "",
        "ours_depth": "",
        "lift_count": "",
        "insert_count": "",
        "p1": "",
        "p2": "",
        "p3": "",
        "p4": "",
        "liveness_p1.0": "",
        "liveness_p0.9": "",
        "liveness_p0.7": "",
        "elapsed_s": "",
        "error": "",
    }

    # Stage 1 — persist extracted action models
    (item_dir / "01_extracted_models.json").write_text(
        json.dumps(
            {
                "lift": _action_model_to_dict(lift_model),
                "insert": _action_model_to_dict(insert_model),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Stage 2 — oriented goals + initial state
    try:
        goals = connection_goals(entry)
        s0 = initial_state_for(entry)
        row["coverage_ok"] = True
        (item_dir / "02_goals.json").write_text(
            json.dumps(_goals_to_dict(goals), indent=2), encoding="utf-8"
        )
        (item_dir / "03_initial_state.json").write_text(
            json.dumps(_state_to_dict(s0), indent=2), encoding="utf-8"
        )
    except IncompleteCoverageError as exc:
        row["error"] = f"coverage: {exc}"
        row["elapsed_s"] = round(time.time() - t0, 3)
        return row

    # Stage 3 — Stage 4 baseline
    try:
        actions_data = _synthesize_actions_data_for_stage4(entry, goals)
        baseline = _stage4_bt_stats(actions_data)
        row["baseline_nodes"] = baseline["nodes"]
        row["baseline_depth"] = baseline["depth"]
        (item_dir / "08_baseline_stage4_stats.json").write_text(
            json.dumps(baseline, indent=2), encoding="utf-8"
        )
    except Exception as exc:  # pragma: no cover
        row["error"] = f"baseline: {exc}"

    # Stage 4 — BT synthesis
    try:
        tree = bt_expand(frozenset(goals), [lift_model, insert_model], s0)
        row["expansion_ok"] = True

        # Topology analysis
        topo = analyse_topology(tree)
        row["ppa_count"] = topo.ppa_pattern_count
        row["mccabe"] = topo.mccabe_complexity
        row["balance_ratio"] = round(topo.balance_ratio, 3)
        row["ours_nodes"] = topo.total_nodes
        row["ours_depth"] = topo.max_depth
        (item_dir / "05_topology.json").write_text(
            json.dumps(topo.to_dict(), indent=2), encoding="utf-8"
        )

        # Action sequence trace
        action_seq = extract_action_sequence(tree, s0)
        (item_dir / "04_action_sequence.json").write_text(
            json.dumps(
                [
                    {"name": a.name, "args": list(a.args), "step": a.at_step_index}
                    for a in action_seq
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        row["lift_count"] = sum(1 for a in action_seq if a.name == "lift")
        row["insert_count"] = sum(1 for a in action_seq if a.name == "insert")

        # Ground-truth comparison
        comp = compare_with_ground_truth(action_seq, entry)
        row["connection_set_match"] = comp.connection_set_match
        row["precedence_violations"] = len(comp.precedence_violations)
        (item_dir / "07_comparison_vs_ground_truth.json").write_text(
            json.dumps(comp.to_dict(), indent=2), encoding="utf-8"
        )

        # Renderings
        (item_dir / "bt.txt").write_text(render_unicode(tree), encoding="utf-8")
        (item_dir / "bt.xml").write_text(to_btcpp_xml(tree), encoding="utf-8")
        title = f"{category}/{name} (parts_ct={parts_ct})"
        write_bt_dot(tree, item_dir / "bt", title=title)
        png_path = write_bt_png(tree, item_dir / "bt", title=title, dpi=100)
        svg_path = write_bt_svg(tree, item_dir / "bt", title=title)

        # Stochastic verification at three p_success levels
        for p in (1.0, 0.9, 0.7):
            report = stochastic_verify(tree, s0, p_success=p, n_rollouts=500, seed=42)
            row[f"liveness_p{p}"] = f"{report.liveness_mean:.3f}"
            if p == 1.0:
                row["p1"] = report.p1_gripper_consistency
                row["p2"] = report.p2_no_double_grasp
                row["p3"] = report.p3_hold_before_connect
                row["p4"] = report.p4_causal_completeness
                (item_dir / "06_verification.json").write_text(
                    json.dumps(_verification_to_dict(report), indent=2),
                    encoding="utf-8",
                )
    except ExpansionFailure as exc:
        row["error"] = f"expansion: {exc}"
    except RewardBTSynthError as exc:
        row["error"] = f"synthesis: {exc}"
    except Exception as exc:  # pragma: no cover
        row["error"] = f"unexpected: {exc}\n{traceback.format_exc()}"

    row["elapsed_s"] = round(time.time() - t0, 3)
    return row


# ---------- Summary writers ----------

_COLUMNS = [
    "name", "category", "parts_ct",
    "extraction_ok", "coverage_ok", "expansion_ok",
    "connection_set_match", "precedence_violations",
    "ppa_count", "mccabe", "balance_ratio",
    "baseline_nodes", "baseline_depth",
    "ours_nodes", "ours_depth",
    "lift_count", "insert_count",
    "p1", "p2", "p3", "p4",
    "liveness_p1.0", "liveness_p0.9", "liveness_p0.7",
    "elapsed_s", "error",
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


def _write_aggregate(rows: list[dict], path: Path) -> None:
    """Summary statistics across the run — one JSON with per-property averages."""
    def _as_bool(v) -> bool:
        return isinstance(v, bool) and v

    def _as_float(v) -> float | None:
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    successful = [r for r in rows if r.get("expansion_ok")]
    coverage_failures = [r for r in rows if r.get("coverage_ok") is False]

    liveness_means = {
        "1.0": [_as_float(r.get("liveness_p1.0")) for r in successful],
        "0.9": [_as_float(r.get("liveness_p0.9")) for r in successful],
        "0.7": [_as_float(r.get("liveness_p0.7")) for r in successful],
    }
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    summary = {
        "total_items": len(rows),
        "successful_expansions": len(successful),
        "coverage_failures": len(coverage_failures),
        "connection_set_match_rate": (
            sum(1 for r in successful if _as_bool(r.get("connection_set_match")))
            / len(successful)
            if successful else None
        ),
        "mean_precedence_violations": (
            sum(int(r.get("precedence_violations", 0) or 0) for r in successful)
            / len(successful)
            if successful else None
        ),
        "p1_p2_p3_p4_all_pass_rate": (
            sum(
                1
                for r in successful
                if all(_as_bool(r.get(p)) for p in ("p1", "p2", "p3", "p4"))
            )
            / len(successful)
            if successful else None
        ),
        "liveness_mean_per_p": {p: _mean(xs) for p, xs in liveness_means.items()},
        "total_elapsed_s": round(
            sum(float(r.get("elapsed_s", 0) or 0) for r in rows), 3
        ),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


# ---------- Entry point ----------

def run_benchmark(main_data_path: str, output_root: str, n_items: int = 50) -> Path:
    with open(main_data_path, "r", encoding="utf-8") as f:
        main_data = json.load(f)

    items = _select_items(main_data, n=n_items)
    print(f"[benchmark] selected {len(items)} items spanning parts_ct "
          f"{min(int(i['parts_ct']) for i in items)}..{max(int(i['parts_ct']) for i in items)}")

    lift_model = extract_action_model("lift", reward_specs)
    insert_model = extract_action_model("insert", reward_specs)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    root = Path(output_root) / timestamp
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, item in enumerate(items, 1):
        print(f"[benchmark {i:>3}/{len(items)}] "
              f"{item['category']}/{item['name']} (parts_ct={item['parts_ct']})")
        rows.append(_run_one(item, lift_model, insert_model, root))

    _write_csv(rows, root / "benchmark_summary.csv")
    _write_md(rows, root / "benchmark_summary.md")
    _write_aggregate(rows, root / "benchmark_aggregate.json")

    (root / "run_metadata.json").write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "n_items_selected": len(items),
                "main_data_path": main_data_path,
                "verifier_n_rollouts": 500,
                "verifier_seed": 42,
                "p_success_levels": [1.0, 0.9, 0.7],
                "vlm_called": False,
                "vlm_default_for_pipeline": "gemini-pro (Gemini 2.5 Pro)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n[benchmark] wrote {root / 'benchmark_summary.csv'}")
    print(f"[benchmark] wrote {root / 'benchmark_summary.md'}")
    print(f"[benchmark] wrote {root / 'benchmark_aggregate.json'}")
    print(f"[benchmark] per-item artefacts in {root}/<Category>/<name>/")
    return root


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--main_data",
        default=str(_THIS.parents[1] / "data" / "main_data.json"),
    )
    p.add_argument(
        "--output_root",
        default=str(_THIS.parents[1] / "reward_bt_outputs"),
    )
    p.add_argument("--items", type=int, default=50)
    args = p.parse_args()
    run_benchmark(args.main_data, args.output_root, n_items=args.items)
