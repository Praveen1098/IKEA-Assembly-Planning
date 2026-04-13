import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add VLM_assembly_plan_gen/ to path so llm/ package is importable
sys.path.append(str(Path(__file__).parent.parent))

from config import MANUAL_DATA_PATH, OUTPUT_DIR, RECIPE_PATH
from llm.model import load_llm_from_recipe
from utils import load_json
from stage1_associate import select_materials_for_planning
from stage2_planning import create_plan
from convert import convert_to_tree
from stage3_action_extraction import extract_actions
from stage4_formalize import to_behavior_tree

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=102)
    p.add_argument("--model", type=str, default="gpt4o",
                   help="Recipe model ID (e.g. gpt4o, gemini-pro, gemini-flash, gpt4o-mini, gpt-5.1)")
    p.add_argument("--debug", action="store_true", help="Enable debug mode")
    p.add_argument("--prompt_type",  type=str, default="numbered")
    p.add_argument("--scene_type",   type=str, default="original")
    p.add_argument("--output_format", type=str, default="tree",
                   choices=["tree", "actions", "bt"],
                   help="Output format: tree (default) | actions | bt. "
                        "'actions' runs Stage 3 only (outputs actions.json). "
                        "'bt' runs Stages 3+4: outputs actions.json, behavior_tree.xml, "
                        "and behavior_tree.txt (ASCII).")
    p.add_argument("--validate_pddl", action="store_true",
                   help="Run Stage 3.5 PDDL consistency check on actions.json "
                        "(requires pyperplan: pip install pyperplan). "
                        "Only active when --output_format is 'actions' or 'bt'.")
    return p.parse_args()

def main():
    args = parse_args()
    data = load_json(MANUAL_DATA_PATH)

    output_name = f"{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # Load main pipeline LLM once for all items
    llm = load_llm_from_recipe(RECIPE_PATH, args.model)

    for idx in tqdm(range(args.start, min(args.end, len(data))), desc="Generating Assembly Graphs"):
        item = data[idx]
        name, cat = item["name"], item["category"]

        stage1_output = select_materials_for_planning(name, cat, output_path, args, llm)

        stage2_output = create_plan(name, cat, output_path, stage1_output, args, llm=llm)

        convert_to_tree(name, cat, output_path, stage2_output, args, llm)

        print(f"Generated Assembly Graph for Furniture Item {cat}\\{name} at {output_path}/{cat}/{name}/tree.json")

        if args.output_format in ("actions", "bt"):
            actions_data = extract_actions(
                name, cat, output_path, stage2_output, stage1_output, args, llm
            )
            print(f"Extracted robot actions for {cat}\\{name} → {output_path}/{cat}/{name}/actions.json")

            if args.validate_pddl:
                from stage3_5_pddl_validate import validate_actions_json
                actions_path = os.path.join(output_path, cat, name, "actions.json")
                validate_actions_json(actions_path, verbose=True)

            if args.output_format == "bt":
                bt_path = to_behavior_tree(actions_data, output_path)
                print(f"Generated Behavior Tree for {cat}\\{name} → {bt_path}")

if __name__ == "__main__":
    main()
