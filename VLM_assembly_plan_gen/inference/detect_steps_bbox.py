"""Detect assembly steps and part bounding boxes from IKEA manual pages.

Standalone script — no SAM / ultralytics dependency.

Two-phase pipeline:
  1. Gemini detects numbered assembly step regions on the manual page;
     each step is cropped and saved as step_{N}_crop.jpg.
  2. For each crop, Gemini detects part bounding boxes (using the annotated
     scene image as a reference). The crop is annotated with supervision's
     BoxAnnotator + LabelAnnotator (exact Roboflow pattern) and saved as
     step_{N}_annotated.jpg. The raw Gemini JSON is saved as step_{N}_parts.json.

Usage:
    python inference/detect_steps_bbox.py \\
        --manual_image data/pdfs/Bench/applaro/page_2.jpg \\
        --scene_image  data/preassembly_scenes/Bench/applaro/scene_annotated.png \\
        --debug
"""

import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import supervision as sv

# Add VLM_assembly_plan_gen/ to path so llm/ and utils/ are importable
sys.path.append(str(Path(__file__).parent.parent))

from config import OUTPUT_DIR, RECIPE_PATH
from llm.model import load_llm_from_recipe
from llm.utils import invoke_multimodal
from utils import ensure_dir


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

STEP_DETECTION_PROMPT = """Examine the assembly manual page image. Locate every diagram that shows a numbered assembly step — these are diagrams labelled with a bold integer (1, 2, 3, …) indicating the assembly order.

Return ONLY a raw JSON array with no explanation, markdown, or code fences. Each item in the array must have exactly two keys:
- "step_number": the integer label of the step
- "box_2d": the bounding box of the step diagram as [y_min, x_min, y_max, x_max], with all four coordinates normalized to the range 0–1000 (where 0 = top or left edge, 1000 = bottom or right edge of the image)

Exclude: overview illustrations, part quantity tables, tool lists, text-only sections, and any diagram that does not carry a numbered step label.

The bounding box must include the step number label.

If no numbered assembly steps are visible on this page, return an empty array: []

Example output:
[{"step_number": 1, "box_2d": [120, 30, 480, 510]}, {"step_number": 2, "box_2d": [520, 30, 880, 510]}]"""

PART_DETECTION_PROMPT = """You are given two images:
  Image 1: An RGB image of the scene consisting of furniture parts labeled with white numbers on a black background.
  Image 2: A cropped diagram from an assembly manual showing a single assembly step.

Identify every furniture part visible in Image 2. For each part, match it to its numbered label from Image 1. Your focus should only be on the furniture parts.

Return ONLY a raw JSON array with no explanation, markdown, or code fences. Each item must have exactly two keys:
- "label": the integer label string (e.g. "3") matching the part's numbered marker in Image 1
- "box_2d": a bounding box around the furniture part in Image 2 as [y_min, x_min, y_max, x_max], with all four coordinates normalized to 0–1000 (relative to Image 2 dimensions)

Rules:
- Coordinates are relative to Image 2 (the step crop), not Image 1.
- Include one entry per distinct visible part instance.
- Bounding boxes must be as tight as possible around the physical silhouette of each part.
  Draw the smallest rectangle that hugs just the part's own outline.

CALLOUT CIRCLES (highest priority rule):
  IKEA diagrams include circular or oval close-up callout insets in the corners, connected to the
  main assembly by a pointer line. These callout circles are diagram annotations, not furniture
  parts. You MUST stop every bounding box at the main assembly boundary — never let a box extend
  into or include a callout circle. Treat the callout circle as a hard wall: if the box edge would
  reach it, pull the edge back to where the actual part ends in the main diagram.

PARTS BEING ACTIVELY HANDLED BY A PERSON:
  When a person is shown inserting or attaching a part, draw the box around the part itself only —
  specifically around the part's physical body (bar, plank, leg, etc.) as it appears in the
  diagram. Do NOT include the person's hands, arms, or torso in the box, even if the person is
  visually overlapping the part. Locate the part's own silhouette within or near the person's
  grip and box only that.

ELONGATED OR ASSEMBLED PARTS:
  For bars, rails, legs, and planks that span across the diagram, the bounding box must cover
  the full visible length of the part end-to-end, even if portions overlap with other parts.
  Do not box only the most visually distinct sub-section.

- Exclude: the human figure or hands (if shown), directional arrows, circular/oval zoom callout
  insets, small hardware (screws, bolts, washers) unless they are the primary subject of the step,
  and the background surface.
- Labels must be integer strings matching the numbered markers in Image 1.

Example output:
[{"label": "2", "box_2d": [100, 50, 600, 450]}, {"label": "5", "box_2d": [620, 300, 900, 700]}]"""

# ---------------------------------------------------------------------------
# [FIX 4] Step-diff prompt — identifies the newly introduced part in the
# current step by comparing it against the previous step crop.
#
# Design notes for Gemini:
#   - Two crops are passed in Image 1 / Image 2 order (prev, curr).
#   - We ask for chain-of-thought via "reasoning" to stabilise predictions;
#     Gemini Flash is more reliable when forced to articulate before committing.
#   - The {parts_context} and {prev_detections_summary} placeholders are filled
#     at call time via str.format().
# ---------------------------------------------------------------------------

PART_DIFF_PROMPT = """You are given two images:
  Image 1: A cropped diagram from an assembly manual showing the PREVIOUS assembly step.
  Image 2: A cropped diagram from an assembly manual showing the CURRENT assembly step.

Your task: identify which furniture part label(s) are NEWLY INTRODUCED or ACTIVELY BEING ASSEMBLED in Image 2 that were not the action focus in Image 1.

A part counts as "newly introduced" if ANY of the following are true in Image 2:
  - It appears for the first time (absent from Image 1 entirely).
  - It is shown floating, partially detached, or being inserted — indicated by a motion arrow pointing toward the main structure, or by a hand gripping it.
  - It is shown in a clearly different spatial position compared to Image 1 (e.g. moved from bottom-right corner to near the main assembly).

{parts_context}

Previous step detected parts at these approximate positions (label: [y_min, x_min, y_max, x_max] in 0–1000 range):
{prev_detections_summary}

Step-by-step reasoning instructions:
  1. Compare the overall scene structure between Image 1 and Image 2.
  2. Note which labeled part changed position, appeared for the first time, or is shown with an action arrow.
  3. Commit to at most 2 new_part_labels.

Return ONLY a raw JSON object — no explanation, no markdown, no code fences:
{{"new_part_labels": ["<label>"], "reasoning": "<one sentence>"}}

If you cannot confidently identify a new part, return:
{{"new_part_labels": [], "reasoning": "<why uncertain>"}}"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _resize_for_api(img_np: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """Downscale image so the longer dimension does not exceed max_side."""
    h, w = img_np.shape[:2]
    if max(h, w) <= max_side:
        return img_np
    scale = max_side / max(h, w)
    return cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _encode_np_as_b64(img_np: np.ndarray, quality: int = 90) -> str:
    """JPEG-encode a BGR ndarray and return a base64 string."""
    ok, buf = cv2.imencode(".jpg", img_np, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("[detect_steps_bbox] _encode_np_as_b64: cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences Gemini sometimes wraps around JSON."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return text.strip()


# ---------------------------------------------------------------------------
# Phase 1 — Step detection
# ---------------------------------------------------------------------------

def _detect_steps(llm, page_img_b64: str, page_path: str) -> list:
    """Call Gemini to detect numbered assembly step bounding boxes on a manual page.

    Returns list of dicts: [{"step_number": int, "box_2d": [y0, x0, y1, x1]}, ...]
    Coordinates are in 0-1000 normalized range (Gemini native format).
    """
    raw = invoke_multimodal(llm, STEP_DETECTION_PROMPT, [page_img_b64], mime_type="image/jpeg")

    try:
        steps = json.loads(_strip_code_fences(raw))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"[detect_steps_bbox] Step detection: JSON parse error\n"
            f"  Page      : {page_path}\n"
            f"  Parse err : {exc}\n"
            f"  Model resp: {raw!r}"
        ) from exc

    if not isinstance(steps, list):
        raise RuntimeError(
            f"[detect_steps_bbox] Step detection: expected JSON array, got {type(steps).__name__}\n"
            f"  Page      : {page_path}\n"
            f"  Model resp: {raw!r}"
        )

    for i, s in enumerate(steps):
        if not isinstance(s, dict) or "step_number" not in s or "box_2d" not in s:
            raise RuntimeError(
                f"[detect_steps_bbox] Step detection: item {i} missing required keys\n"
                f"  Page      : {page_path}\n"
                f"  Item      : {s!r}\n"
                f"  Model resp: {raw!r}"
            )

    return steps


# ---------------------------------------------------------------------------
# Phase 2 helpers — [FIX 2] prev-detections context + [FIX 4] step-diff
# ---------------------------------------------------------------------------

def _format_prev_detections_context(prev_detections: dict) -> str:
    """Render prev_detections {label: [box, ...]} as a compact summary string.

    Used both for the PART_DIFF_PROMPT placeholder and as an appended block in
    the main PART_DETECTION_PROMPT so Gemini has spatial priors.
    """
    if not prev_detections:
        return "No prior step detections available."
    lines = []
    for label in sorted(prev_detections.keys(), key=lambda x: int(x)):
        for box in prev_detections[label]:
            lines.append(f"  Label {label}: {box}")
    return "\n".join(lines)


def _identify_new_part(
    llm,
    prev_crop_b64: str,
    curr_crop_b64: str,
    prev_detections: dict,
    parts_context: str,
    debug: bool = False,
) -> list[str]:
    """[FIX 4] Ask Gemini which part label is newly introduced in curr vs prev step.

    Args:
        prev_crop_b64: base64-encoded JPEG of the previous step crop.
        curr_crop_b64: base64-encoded JPEG of the current step crop.
        prev_detections: {label_str: [box, ...]} from the previous step.
        parts_context: the parts_context string from stage1 (may be None).
        debug: print reasoning output.

    Returns:
        List of label strings identified as newly introduced (often length 1).
        Returns [] if the call fails or Gemini is uncertain.
    """
    prev_summary = _format_prev_detections_context(prev_detections)
    prompt = PART_DIFF_PROMPT.format(
        parts_context=parts_context or "",
        prev_detections_summary=prev_summary,
    )

    raw = invoke_multimodal(
        llm, prompt, [prev_crop_b64, curr_crop_b64], mime_type="image/jpeg"
    )

    try:
        result = json.loads(_strip_code_fences(raw))
        new_labels = [str(lbl) for lbl in result.get("new_part_labels", [])]
        if debug:
            print(
                f"[detect_steps_bbox]   step-diff → new_part_labels={new_labels} "
                f"| reasoning: {result.get('reasoning', '')}"
            )
        return new_labels
    except (json.JSONDecodeError, AttributeError, TypeError):
        if debug:
            print(f"[detect_steps_bbox]   step-diff: JSON parse failed — raw={raw!r}")
        return []


def _parse_detections_to_dict(raw_json: str) -> dict:
    """Parse raw Gemini JSON into {label_str: [[y0,x0,y1,x1], ...]} dict.

    Used to accumulate detections for passing to the next step.
    Returns {} on any parse failure (non-fatal — prev context is optional).
    """
    try:
        items = json.loads(_strip_code_fences(raw_json))
        result: dict[str, list] = {}
        for item in items:
            lbl = str(item.get("label", ""))
            box = item.get("box_2d")
            if lbl and box:
                result.setdefault(lbl, []).append(box)
        return result
    except (json.JSONDecodeError, TypeError, AttributeError):
        return {}


# ---------------------------------------------------------------------------
# Phase 2 — Part detection + supervision annotation
# ---------------------------------------------------------------------------

def _detect_and_annotate_parts(
    llm,
    scene_b64: str,
    step_crop_np: np.ndarray,
    step_num: int,
    output_dir: str,
    debug: bool,
    parts_context: str | None = None,
    # [FIX 2] previous step state -----------------------------------------
    prev_detections: dict | None = None,   # {label_str: [[y0,x0,y1,x1],...]}
    prev_crop_b64: str | None = None,      # encoded JPEG of previous step crop
    # [FIX 4] step-diff result (injected from caller or computed internally) -
    new_part_labels: list | None = None,
) -> dict:
    """Detect part bboxes with Gemini and annotate the step crop using supervision.

    Returns:
        parsed_detections: {label_str: [[y0,x0,y1,x1], ...]} for the current
        step, to be passed as prev_detections to the next call.

    Saves:
      output_dir/step_{step_num}_parts.json    — raw Gemini response
      output_dir/step_{step_num}_annotated.jpg — supervision-annotated image

    [FIX 2] If prev_detections is provided, a spatial-prior block is appended
    to the prompt so Gemini knows where parts were in the prior step and can
    reason about what moved.

    [FIX 4] If prev_crop_b64 is also provided and new_part_labels is None,
    _identify_new_part is called first; the result is injected into the prompt
    as a focus hint so Gemini applies tighter localisation to the new part.
    """
    resized_crop = _resize_for_api(step_crop_np)
    resized_h, resized_w = resized_crop.shape[:2]
    step_crop_b64 = _encode_np_as_b64(resized_crop)

    # ------------------------------------------------------------------
    # [FIX 4] Step-diff: identify newly introduced part if we have prev
    # ------------------------------------------------------------------
    if new_part_labels is None and prev_crop_b64 is not None and prev_detections:
        new_part_labels = _identify_new_part(
            llm,
            prev_crop_b64=prev_crop_b64,
            curr_crop_b64=step_crop_b64,
            prev_detections=prev_detections,
            parts_context=parts_context or "",
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Build the augmented prompt
    # ------------------------------------------------------------------
    prompt_blocks = [PART_DETECTION_PROMPT]

    if parts_context:
        prompt_blocks.append(parts_context)

    # [FIX 2] Append previous step's spatial priors
    if prev_detections:
        prev_ctx = _format_prev_detections_context(prev_detections)
        prompt_blocks.append(
            "SPATIAL PRIORS FROM THE PREVIOUS STEP — use these to anchor your "
            "detections. Parts that are already assembled will appear in similar "
            "regions; a part that shifted significantly is likely the action target:\n"
            + prev_ctx
        )

    # [FIX 4] Append focus hint for the newly introduced part
    if new_part_labels:
        label_list = ", ".join(new_part_labels)
        prompt_blocks.append(
            f"ASSEMBLY ACTION FOCUS — label(s) {label_list} appear to be newly "
            f"introduced or actively inserted in this step. Apply especially tight "
            f"bounding boxes to these part(s) and do not merge them with adjacent "
            f"structure."
        )

    prompt = "\n\n".join(prompt_blocks)

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    raw = invoke_multimodal(
        llm, prompt, [scene_b64, step_crop_b64], mime_type="image/jpeg"
    )

    # Save raw response
    parts_json_path = os.path.join(output_dir, f"step_{step_num}_parts.json")
    with open(parts_json_path, "w", encoding="utf-8") as f:
        f.write(raw)
    if debug:
        print(f"[detect_steps_bbox]   step {step_num}: parts JSON → {parts_json_path}")

    # Parse for next-step accumulation [FIX 2]
    parsed_detections = _parse_detections_to_dict(raw)

    # ------------------------------------------------------------------
    # Supervision annotation
    # ------------------------------------------------------------------
    # resolution_wh must match the image Gemini actually received (resized, not original)
    resolution_wh = (resized_w, resized_h)

    try:
        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.GOOGLE_GEMINI_2_5,
            result=raw,
            resolution_wh=resolution_wh,
        )
    except Exception as exc:
        print(
            f"[detect_steps_bbox]   step {step_num}: WARNING — supervision parse failed "
            f"({exc}); skipping annotation"
        )
        return parsed_detections

    if len(detections) == 0:
        if debug:
            print(f"[detect_steps_bbox]   step {step_num}: no parts detected, skipping annotation")
        return parsed_detections

    if debug:
        print(f"[detect_steps_bbox]   step {step_num}: {len(detections)} part(s) detected")

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        smart_position=True,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        text_position=sv.Position.CENTER,
    )

    step_crop_pil = Image.fromarray(cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB))

    annotated = step_crop_pil
    for annotator in (box_annotator, label_annotator):
        annotated = annotator.annotate(scene=annotated, detections=detections)

    annotated_path = os.path.join(output_dir, f"step_{step_num}_annotated.jpg")
    if isinstance(annotated, np.ndarray):
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    else:
        annotated.save(annotated_path)

    if debug:
        print(f"[detect_steps_bbox]   step {step_num}: annotated → {annotated_path}")

    return parsed_detections


# ---------------------------------------------------------------------------
# Parts context helper
# ---------------------------------------------------------------------------

def _format_parts_context(stage1_raw: str) -> str:
    """Format a stage1 JSON parts list into a natural-language context block."""
    cleaned = stage1_raw.replace("```json", "").replace("```", "").strip()
    parts = json.loads(cleaned)
    lines = ["The following labeled parts exist in Image 1 (the scene image):"]
    for entry in parts:
        for n in entry.get("number", []):
            lines.append(f"- Label {n}: {entry['name']}")
    lines.append(
        "Detect ONLY parts with these exact label numbers. "
        "Do not use any other label numbers outside this list."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect assembly steps + part bounding boxes from a manual page image."
    )
    p.add_argument("--manual_image", required=True,
                   help="Path to a manual page image (JPEG or PNG)")
    p.add_argument("--scene_image",  required=True,
                   help="Path to annotated scene image with numbered part labels")
    p.add_argument("--output_dir",   type=str, default=None,
                   help="Output directory (default: outputs/<timestamp>/)")
    p.add_argument("--model",        type=str, default="gemini-flash",
                   help="Recipe model ID (default: gemini-flash)")
    p.add_argument("--parts_json",   type=str, default=None,
                   help="Optional: path to stage1-format JSON to constrain part detection")
    p.add_argument("--debug",        action="store_true",
                   help="Print step-by-step progress messages")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(
        OUTPUT_DIR, datetime.now().strftime("%Y_%m_%d_%H%M%S")
    )
    ensure_dir(output_dir)

    llm = load_llm_from_recipe(RECIPE_PATH, args.model)

    parts_context = None
    if args.parts_json:
        with open(args.parts_json, "r", encoding="utf-8") as f:
            raw_parts = f.read()
        parts_context = _format_parts_context(raw_parts)
        if args.debug:
            print(f"[detect_steps_bbox] Parts context loaded from: {args.parts_json}")

    page_np = cv2.imread(args.manual_image)
    if page_np is None:
        raise RuntimeError(f"[detect_steps_bbox] Cannot open manual image: {args.manual_image}")
    ph, pw = page_np.shape[:2]

    scene_np = cv2.imread(args.scene_image)
    if scene_np is None:
        raise RuntimeError(f"[detect_steps_bbox] Cannot open scene image: {args.scene_image}")
    scene_b64 = _encode_np_as_b64(_resize_for_api(scene_np))

    page_b64 = _encode_np_as_b64(_resize_for_api(page_np))
    if args.debug:
        print(f"[detect_steps_bbox] Detecting steps in: {args.manual_image}")

    steps = _detect_steps(llm, page_b64, args.manual_image)

    if args.debug:
        print(f"[detect_steps_bbox] {len(steps)} step(s) detected: "
              f"{[s['step_number'] for s in steps]}")

    if not steps:
        print("[detect_steps_bbox] No assembly steps detected on this page.")
        return

    # [FIX 2 + 4] State carried across steps
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
            print(f"[detect_steps_bbox] WARNING: degenerate bbox for step {step_num}, skipping")
            continue

        step_crop_np = page_np[y0_px:y1_px, x0_px:x1_px]
        crop_path = os.path.join(output_dir, f"step_{step_num}_crop.jpg")
        cv2.imwrite(crop_path, step_crop_np)
        if args.debug:
            print(f"[detect_steps_bbox] step {step_num}: crop saved → {crop_path}")

        curr_crop_b64 = _encode_np_as_b64(_resize_for_api(step_crop_np))

        # [FIX 2 + 4] Pass accumulated state; get back current step's detections
        parsed_detections = _detect_and_annotate_parts(
            llm, scene_b64, step_crop_np, step_num, output_dir, args.debug,
            parts_context=parts_context,
            prev_detections=prev_detections if prev_detections else None,
            prev_crop_b64=prev_crop_b64,
        )

        # Advance state for next step
        prev_detections = parsed_detections
        prev_crop_b64 = curr_crop_b64

    print(f"\n[detect_steps_bbox] Done. Output in: {output_dir}")


if __name__ == "__main__":
    main()
