"""Command-line interface for the reward_bt_synth pipeline.

Subcommands:
  extract       Print Lift + Insert derivation tables (slide artefact)
  synthesize    End-to-end: ground-truth → BT for one furniture
  benchmark     Run the 15-furniture benchmark, emit CSV + markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parents[2]))  # repo root
sys.path.insert(0, str(_THIS.parents[1]))  # VLM_assembly_plan_gen

from reward_bt_synth import reward_specs  # noqa: E402
from reward_bt_synth.ast_extractor import extract_action_model  # noqa: E402
from reward_bt_synth.bt_expansion import (  # noqa: E402
    bt_expand,
    count_nodes_by_kind,
    is_ppa_compliant,
    tree_depth,
    tree_node_count,
)
from reward_bt_synth.ground_truth import connection_goals, initial_state_for  # noqa: E402
from reward_bt_synth.pytrees_build import render_unicode, to_btcpp_xml  # noqa: E402
from reward_bt_synth.stochastic_verifier import stochastic_verify  # noqa: E402


def _cmd_extract(args: argparse.Namespace) -> int:
    """Print Lift and Insert STRIPS action models as a derivation table."""
    for skill in ("lift", "insert"):
        model = extract_action_model(skill, reward_specs)
        print("=" * 60)
        print(model.to_pretty_string())
        print()
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    """End-to-end: one furniture → BT + verification."""
    with open(args.main_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    entry = next((e for e in data if e.get("name") == args.name), None)
    if entry is None:
        print(f"[synthesize] ERROR: furniture {args.name!r} not found in main_data",
              file=sys.stderr)
        return 2

    goals = connection_goals(entry)
    s0 = initial_state_for(entry)
    lift = extract_action_model("lift", reward_specs)
    insert = extract_action_model("insert", reward_specs)

    tree = bt_expand(frozenset(goals), [lift, insert], s0)

    # Render to stdout
    print(f"\n=== {entry['category']}/{entry['name']} "
          f"(parts_ct={entry['parts_ct']}) ===\n")
    print(render_unicode(tree))
    print()

    # Summary
    print(f"Tree: nodes={tree_node_count(tree)}, depth={tree_depth(tree)}, "
          f"PPA-compliant={is_ppa_compliant(tree)}")
    print(f"Node counts: {count_nodes_by_kind(tree)}")

    # Verification
    report = stochastic_verify(tree, s0, p_success=1.0, n_rollouts=500)
    print(f"Verification: passed={report.passed}, "
          f"liveness={report.liveness_mean:.3f}")
    for pname, ok in [
        ("P1 gripper_consistency", report.p1_gripper_consistency),
        ("P2 no_double_grasp", report.p2_no_double_grasp),
        ("P3 hold_before_connect", report.p3_hold_before_connect),
        ("P4 causal_completeness", report.p4_causal_completeness),
    ]:
        mark = "[OK]" if ok else "[FAIL]"
        print(f"  {mark} {pname}")

    # Optional output directory
    if args.output_dir:
        out = Path(args.output_dir) / entry["category"] / entry["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "bt.txt").write_text(render_unicode(tree), encoding="utf-8")
        (out / "bt.xml").write_text(to_btcpp_xml(tree), encoding="utf-8")
        print(f"\n[synthesize] wrote {out / 'bt.txt'}, {out / 'bt.xml'}")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from reward_bt_synth.benchmark import run_benchmark

    csv_path, md_path = run_benchmark(
        main_data_path=args.main_data,
        output_root=args.output_root,
        n_items=args.items,
    )
    print(f"\nbenchmark_summary.csv: {csv_path}")
    print(f"benchmark_summary.md : {md_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reward_bt_synth.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # extract
    sub.add_parser("extract", help="Print Lift + Insert derivation tables")

    # synthesize
    sp = sub.add_parser("synthesize", help="End-to-end run on one furniture")
    sp.add_argument("--name", required=True, help="furniture name (e.g. applaro)")
    sp.add_argument(
        "--main_data",
        default=str(_THIS.parents[1] / "data" / "main_data.json"),
    )
    sp.add_argument("--output_dir", default=None)

    # benchmark
    bp = sub.add_parser("benchmark", help="Run the 15-item benchmark")
    bp.add_argument(
        "--main_data",
        default=str(_THIS.parents[1] / "data" / "main_data.json"),
    )
    bp.add_argument(
        "--output_root",
        default=str(_THIS.parents[1] / "reward_bt_outputs"),
    )
    bp.add_argument("--items", type=int, default=15)

    args = parser.parse_args(argv)
    if args.cmd == "extract":
        return _cmd_extract(args)
    if args.cmd == "synthesize":
        return _cmd_synthesize(args)
    if args.cmd == "benchmark":
        return _cmd_benchmark(args)
    parser.error(f"unknown subcommand {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
