import json
import os

from utils import load_prompt, ensure_dir
from llm.utils import invoke_multimodal

def convert_to_tree(furniture_name, furniture_type, pdf_path, current_plan, args, llm):
    """Convert furniture assembly plan to a tree structure."""
    prompt_text = "tree_ikea_manual"

    # Load tree conversion prompt
    prompt = load_prompt(prompt_text)
    full_prompt = prompt + "\n" + current_plan + "\n\nYOUR REAL OUTPUT:\n"

    if args.debug:
        prompt_file = os.path.join(pdf_path, furniture_type, furniture_name, "debug", "convert_prompt.txt")
        with open(prompt_file, "w") as f:
            f.write(full_prompt)
        print(f"Saved convert prompt to {prompt_file}")

    # Generate tree structure (text-only call — no images)
    raw = invoke_multimodal(llm, full_prompt, [])
    final_output = raw.replace("```python", "").replace("```", "")

    # Save tree output
    if args.debug:
        tree_file = os.path.join(pdf_path, furniture_type, furniture_name, "debug", f"convert_start_{args.start}.txt")
        with open(tree_file, "a") as f:
            f.write("\n=================================================================\n")
            f.write(f"{furniture_type} {furniture_name}\n\n")
            f.write(final_output)
        print(f"Appended final output assembly graph to {tree_file}")

    # Save JSON version
    serialized_tree = json.dumps(final_output)
    output_folder = os.path.join(pdf_path, furniture_type, furniture_name)
    ensure_dir(output_folder)

    with open(os.path.join(output_folder, "tree.json"), "w") as f:
        f.write(serialized_tree)
    print(f"Saved assembly graph to {os.path.join(output_folder, 'tree.json')}")
