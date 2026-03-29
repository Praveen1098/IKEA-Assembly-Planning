"""Orchestrator: stage1 part identification + contextual step/bbox detection.

Mirrors run.py structure exactly:
  1. Runs select_materials_for_planning() (stage1) for a furniture item.
  2. Formats the stage1 parts table as a context string.
  3. Runs detect_steps_bbox on a single manual page using that context,
     so part detection is constrained to the known labeled parts.

Usage:
    python inference/run_detect.py \\
        --name applaro \\
        --type Bench \\
        --manual_image data/pdfs/Bench/applaro/page_2.jpg \\
        --debug
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).parent.parent))

from config import OUTPUT_DIR, RECIPE_PATH, SCENE_DIR
from llm.model import load_llm_from_recipe
from utils import ensure_dir
from stage1_associate import select_materials_for_planning
from detect_steps_bbox import (
    _resize_for_api,
    _encode_np_as_b64,
    _detect_steps,
    _detect_and_annotate_parts,
    _format_parts_context,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run stage1 part identification then contextual step/bbox detection on one manual page."
    )
    p.add_argument("--name",         required=True,
                   help="Furniture name (e.g. applaro)")
    p.add_argument("--type",         required=True,
                   help="Furniture category (e.g. Bench)")
    p.add_argument("--manual_image", required=True,
                   help="Path to a single manual page image to run bbox detection on")
    p.add_argument("--model",        type=str, default="gemini-flash",
                   help="Recipe model ID (default: gemini-flash)")
    p.add_argument("--scene_type",   type=str, default="original",
                   help="Scene image variant: original | rot (default: original)")
    p.add_argument("--output_dir",   type=str, default=None,
                   help="Output directory (default: outputs/<timestamp>/)")
    p.add_argument("--debug",        action="store_true",
                   help="Enable debug mode")
    return p.parse_args()


def main():
    args = parse_args()

    output_path = args.output_dir or os.path.join(
        OUTPUT_DIR, datetime.now().strftime("%Y_%m_%d_%H%M%S")
    )
    ensure_dir(output_path)

    llm = load_llm_from_recipe(RECIPE_PATH, args.model)

    # Stage 1
    print(f"[run_detect] Stage 1: identifying parts for {args.type}/{args.name}")
    stage1_output = select_materials_for_planning(args.name, args.type, output_path, args, llm)

    parts_context = _format_parts_context(stage1_output)
    if args.debug:
        print(f"[run_detect] Parts context:\n{parts_context}\n")

    context_path = os.path.join(output_path, "parts_context.txt")
    with open(context_path, "w", encoding="utf-8") as f:
        f.write(parts_context)

    # Resolve scene image
    scene_filename = "scene_annotated.png" if args.scene_type == "original" else "scene_rot_annotated.png"
    scene_img_path = os.path.join(SCENE_DIR, args.type, args.name, scene_filename)

    scene_np = cv2.imread(scene_img_path)
    if scene_np is None:
        raise RuntimeError(f"[run_detect] Cannot open scene image: {scene_img_path}")
    scene_b64 = _encode_np_as_b64(_resize_for_api(scene_np))

    # Read manual page
    page_np = cv2.imread(args.manual_image)
    if page_np is None:
        raise RuntimeError(f"[run_detect] Cannot open manual image: {args.manual_image}")
    ph, pw = page_np.shape[:2]

    # Detect step regions
    print(f"[run_detect] Detecting steps in: {args.manual_image}")
    page_b64 = _encode_np_as_b64(_resize_for_api(page_np))
    steps = _detect_steps(llm, page_b64, args.manual_image)

    print(f"[run_detect] {len(steps)} step(s) detected: {[s['step_number'] for s in steps]}")

    if not steps:
        print("[run_detect] No assembly steps detected on this page.")
        return

    # [FIX 2 + 4] Persistent state across the step loop.
    #
    # prev_detections: {label_str: [[y0,x0,y1,x1], ...]}
    #   Populated after each step completes. On step N+1 this becomes the
    #   "spatial prior" block appended to the detection prompt, anchoring
    #   Gemini's search to plausible regions for already-assembled parts.
    #
    # prev_crop_b64: base64 JPEG of the previous step crop.
    #   Used together with prev_detections to run the step-diff call
    #   (_identify_new_part) before the main detection prompt, so Gemini
    #   knows which part to apply extra localisation effort to.
    prev_detections: dict = {}
    prev_crop_b64: str | None = None

    for step in steps:
        step_num = int(step["step_number"])
        y0, x0, y1, x1 = step["box_2d"]

        y0_px = int(y0 / 1000 * ph)
        x0_px = max(0, int(x0 / 1000 * pw))
        y1_px = min(ph, int(y1 / 1000 * ph))
        x1_px = min(pw, int(x1 / 1000 * pw))

        step_label_pad = max(40, int(0.05 * ph))
        y0_px = max(0, y0_px - step_label_pad)

        if y1_px <= y0_px or x1_px <= x0_px:
            print(f"[run_detect] WARNING: degenerate bbox for step {step_num}, skipping")
            continue

        step_crop_np = page_np[y0_px:y1_px, x0_px:x1_px]
        crop_path = os.path.join(output_path, f"step_{step_num}_crop.jpg")
        cv2.imwrite(crop_path, step_crop_np)
        if args.debug:
            print(f"[run_detect] step {step_num}: crop saved → {crop_path}")

        # Encode current crop before the LLM call so we can store it for the
        # next iteration regardless of whether detection succeeds.
        curr_crop_b64 = _encode_np_as_b64(_resize_for_api(step_crop_np))

        # [FIX 2 + 4] Pass prev state in; capture current step's detections out.
        parsed_detections = _detect_and_annotate_parts(
            llm, scene_b64, step_crop_np, step_num, output_path, args.debug,
            parts_context=parts_context,
            prev_detections=prev_detections if prev_detections else None,
            prev_crop_b64=prev_crop_b64,
        )

        # Advance state: current step becomes "previous" for the next iteration.
        prev_detections = parsed_detections
        prev_crop_b64 = curr_crop_b64

    print(f"\n[run_detect] Done. Output in: {output_path}")


if __name__ == "__main__":
    main()
