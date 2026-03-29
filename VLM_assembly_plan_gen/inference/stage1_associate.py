import os

from config import DATA_DIR, SCENE_DIR
from utils import encode_image, load_prompt, ensure_dir, alphanumeric_sort_key
from llm.utils import invoke_multimodal

def generate_json(img_path, manual_path, output_path, llm, debug=False):
    """Generate JSON label data from scene + page_1 images using the provided LLM."""
    base64_image = encode_image(img_path)
    base64_image2 = encode_image(manual_path)

    prompt = load_prompt("generate_json")

    raw = invoke_multimodal(llm, prompt, [base64_image, base64_image2], mime_type="image/png")
    final_output = raw.replace("```json", "").replace("```", "")

    if debug:
        with open(os.path.join(output_path, "stage1_prompt_partial.txt"), "w") as f:
            f.write(prompt)
        print(f"Saved stage1 input prompt A to {os.path.join(output_path, 'stage1_prompt_partial.txt')}")
        with open(os.path.join(output_path, "stage1_output_partial.json"), "w") as f:
            f.write(final_output)
        print(f"Saved stage1 output A to {os.path.join(output_path, 'stage1_output_partial.json')}")

    return final_output

def select_materials_for_planning(furniture_name, furniture_type, pdf_path, args, llm):
    """Select materials for planning based on furniture specifications."""
    if args.scene_type == "original":
        scene_type = "scene_annotated.png"
    else:
        scene_type = "scene_rot_annotated.png"
    scene_path = os.path.join(SCENE_DIR, furniture_type, furniture_name, scene_type)

    manual_path = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name, "page_1.png")

    output_folder = os.path.join(pdf_path, furniture_type, furniture_name, "debug")
    ensure_dir(output_folder)

    raw_table = generate_json(scene_path, manual_path, output_folder, llm, args.debug)

    prompt_text = "select_material"

    # Get manual pages
    b64_pages = []
    manual_dir = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name)
    for page in os.listdir(manual_dir):
        if page.endswith(".png"):
            page_path = os.path.join(manual_dir, page)
            b64_pages.append(encode_image(page_path))

    b64_pages_sorted = sorted(b64_pages, key=alphanumeric_sort_key)
    base64_image = encode_image(scene_path)

    prompt = load_prompt(prompt_text)
    full_prompt = prompt + "\n\nAnd here is the json file: \n" + raw_table

    raw = invoke_multimodal(llm, full_prompt, [base64_image] + b64_pages_sorted, mime_type="image/png")

    if args.debug:
        with open(os.path.join(output_folder, "stage1_prompt.txt"), "w") as f:
            f.write(full_prompt)
        print(f"Saved stage1 input prompt B to {os.path.join(output_folder, 'stage1_prompt.txt')}")
        with open(os.path.join(output_folder, "stage1_output.json"), "w") as f:
            f.write(raw.replace("```json", "").replace("```", ""))
        print(f"Saved stage1 output B to {os.path.join(output_folder, 'stage1_output.json')}")

    return raw
