"""
Run inference with a fine-tuned SAM2 model on IKEA manual step crop images.

Supports two modes:
  - batch: generate masks for all steps of all furniture items
  - single: predict masks for one image given box prompts

Usage:
    cd VLM_assembly_plan_gen
    python -m part_segmentation.predict \
        --checkpoint checkpoints/sam2_ikea_best.pt \
        --output_dir data/sam2_mask/
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running as a script or as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from part_segmentation.config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    INFER_DEFAULTS,
    MAIN_DATA_JSON,
    MASK_DIR,
    SAM2_MASK_DIR,
    SAM2_MODELS,
)


def load_sam2_model(checkpoint_path: str, model_size: str = "base_plus", device: str = "cuda"):
    """Load SAM2 model with fine-tuned weights."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_cfg = SAM2_MODELS[model_size]["config"]
    model = build_sam2(model_cfg, checkpoint_path)
    model = model.to(device)
    model.eval()
    predictor = SAM2ImagePredictor(model)
    return predictor


def predict_masks(
    predictor,
    image: np.ndarray,
    box_prompts: np.ndarray = None,
    mask_threshold: float = 0.5,
    min_mask_area: int = 100,
) -> list:
    """Run SAM2 inference on a single image.

    Args:
        predictor: SAM2ImagePredictor with loaded model.
        image: RGB image as numpy array (H, W, 3).
        box_prompts: Bounding boxes as (N, 4) array [x1, y1, x2, y2].
            If None, uses automatic mask generation with a point grid.
        mask_threshold: Threshold for binary mask.
        min_mask_area: Minimum mask area in pixels.

    Returns:
        List of binary masks as numpy arrays (H, W).
    """
    predictor.set_image(image)

    if box_prompts is not None and len(box_prompts) > 0:
        masks_list = []
        for box in box_prompts:
            masks, scores, _ = predictor.predict(
                box=box,
                multimask_output=True,
            )
            # Take the highest-scoring mask
            best_idx = scores.argmax()
            mask = (masks[best_idx] > mask_threshold).astype(np.uint8)
            if mask.sum() >= min_mask_area:
                masks_list.append(mask)
        return masks_list

    # Automatic mask generation with point grid
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    mask_generator = SAM2AutomaticMaskGenerator(
        model=predictor.model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        min_mask_region_area=min_mask_area,
    )
    results = mask_generator.generate(image)
    return [r["segmentation"].astype(np.uint8) for r in results]


def nms_masks(masks: list, iou_threshold: float = 0.8) -> list:
    """Non-maximum suppression on overlapping masks."""
    if len(masks) <= 1:
        return masks

    # Sort by area (largest first)
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]

    keep = []
    for i in order:
        suppress = False
        for j in keep:
            intersection = (masks[i] & masks[j]).sum()
            union = (masks[i] | masks[j]).sum()
            if union > 0 and intersection / union > iou_threshold:
                suppress = True
                break
        if not suppress:
            keep.append(i)

    return [masks[i] for i in keep]


def generate_masks_for_item(
    predictor,
    item: dict,
    output_dir: str,
    use_gt_boxes: bool = True,
    mask_threshold: float = 0.5,
    min_mask_area: int = 100,
    nms_iou: float = 0.8,
):
    """Generate mask PNGs for all steps of a furniture item.

    Args:
        predictor: SAM2ImagePredictor.
        item: Furniture item dict from main_data.json.
        output_dir: Base output directory for masks.
        use_gt_boxes: If True, derive box prompts from GT masks.
        mask_threshold: Threshold for binary mask.
        min_mask_area: Minimum mask area.
        nms_iou: NMS IoU threshold.
    """
    cat, name = item["category"], item["name"]
    item_mask_dir = os.path.join(output_dir, cat, name)
    os.makedirs(item_mask_dir, exist_ok=True)

    for step_idx, step in enumerate(item["steps"]):
        no_seg_path = os.path.join(MASK_DIR, cat, name, f"step_{step_idx}_no_seg.png")
        if not os.path.exists(no_seg_path):
            continue

        image = np.array(Image.open(no_seg_path).convert("RGB"))

        # Get box prompts
        box_prompts = None
        if use_gt_boxes and step.get("masks"):
            import pycocotools.mask as mask_util

            boxes = []
            for m in step["masks"]:
                rle = {"size": m["size"], "counts": m["counts"].encode("utf-8")}
                decoded = mask_util.decode(rle)
                rows = np.any(decoded, axis=1)
                cols = np.any(decoded, axis=0)
                if rows.any() and cols.any():
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    # Scale to image dimensions
                    rle_h, rle_w = m["size"]
                    img_h, img_w = image.shape[:2]
                    sx, sy = img_w / rle_w, img_h / rle_h
                    boxes.append([x_min * sx, y_min * sy, x_max * sx, y_max * sy])
            if boxes:
                box_prompts = np.array(boxes, dtype=np.float32)

        # Run prediction
        masks = predict_masks(
            predictor, image, box_prompts, mask_threshold, min_mask_area
        )
        masks = nms_masks(masks, nms_iou)

        if not masks:
            continue

        # Save mask PNGs (same naming convention as data/mask/)
        # no_seg: just the colored mask overlay without numbers
        # no_seg_numbered: with part numbers on centroids
        _save_mask_images(image, masks, item_mask_dir, step_idx)


def _save_mask_images(
    image: np.ndarray, masks: list, output_dir: str, step_idx: int
):
    """Save mask overlay images matching the pipeline naming convention."""
    h, w = image.shape[:2]

    # Generate distinct colors for each instance
    np.random.seed(step_idx)
    colors = []
    for i in range(len(masks)):
        hue = int(180 * i / max(len(masks), 1))
        color_hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(color_rgb.tolist())

    # Create overlay with colored masks
    overlay = image.copy()
    for mask, color in zip(masks, colors):
        overlay[mask > 0] = (
            np.array(color) * 0.4 + overlay[mask > 0] * 0.6
        ).astype(np.uint8)

    # Save unnumbered version
    no_seg_path = os.path.join(output_dir, f"step_{step_idx}_no_seg.png")
    cv2.imwrite(no_seg_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Create numbered version with part labels at centroids
    numbered = overlay.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(h, w) / 1500)
    font_thickness = max(2, int(font_scale * 2.5))

    for i, mask in enumerate(masks):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = str(i)

        text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        tw, th = text_size
        pad = 5

        cv2.rectangle(
            numbered,
            (cx - pad, cy - th - pad),
            (cx + tw + pad, cy + pad),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            numbered, label, (cx, cy), font, font_scale,
            (255, 255, 255), font_thickness,
        )

    # Save numbered version (wider to match convention — add number legend sidebar)
    no_seg_numbered_path = os.path.join(
        output_dir, f"step_{step_idx}_no_seg_numbered.png"
    )
    cv2.imwrite(no_seg_numbered_path, cv2.cvtColor(numbered, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Generate SAM2 masks for IKEA furniture items")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned SAM2 checkpoint")
    parser.add_argument("--model_size", type=str, default=INFER_DEFAULTS["model_size"],
                        choices=list(SAM2_MODELS.keys()))
    parser.add_argument("--output_dir", type=str, default=SAM2_MASK_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=102)
    parser.add_argument("--no_gt_boxes", action="store_true",
                        help="Use automatic mask generation instead of GT boxes")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(MAIN_DATA_JSON) as f:
        data = json.load(f)

    predictor = load_sam2_model(args.checkpoint, args.model_size, args.device)

    for idx in tqdm(range(args.start, min(args.end, len(data))), desc="Generating masks"):
        item = data[idx]
        generate_masks_for_item(
            predictor, item, args.output_dir,
            use_gt_boxes=not args.no_gt_boxes,
            mask_threshold=INFER_DEFAULTS["mask_threshold"],
            min_mask_area=INFER_DEFAULTS["min_mask_area"],
            nms_iou=INFER_DEFAULTS["nms_iou_threshold"],
        )


if __name__ == "__main__":
    main()
