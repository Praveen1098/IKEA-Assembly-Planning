"""
Convert main_data.json RLE masks to a COCO-format dataset for SAM2 fine-tuning.

Projects per-instance RLE masks (at fixed resolution) onto manual step crop images
using an affine transformation derived from the mask overlay files.

Usage:
    cd VLM_assembly_plan_gen
    python -m part_segmentation.data_preparation
"""

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Allow running as a script or as a module
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
from tqdm import tqdm

from part_segmentation.config import (
    DATA_DIR,
    MAIN_DATA_JSON,
    MASK_DIR,
    SAM2_DATASET_DIR,
    SAM2_DATASET_ZIP,
)


def decode_rle(rle_dict: dict) -> np.ndarray:
    """Decode a COCO RLE dict to a binary mask (H, W)."""
    rle = {"size": rle_dict["size"], "counts": rle_dict["counts"].encode("utf-8")}
    return mask_util.decode(rle)


def mask_to_bbox_xywh(binary_mask: np.ndarray) -> list:
    """Derive [x, y, w, h] bounding box from a binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def find_mask_overlay_bbox(overlay_path: str, page_path: str, threshold: float = 50.0):
    """Find bounding box of mask regions in the full-page mask overlay.

    Returns (x_min, y_min, x_max, y_max) in page coordinates, or None if not found.
    """
    overlay = np.array(Image.open(overlay_path))[:, :, :3].astype(np.float32)
    page = np.array(Image.open(page_path)).astype(np.float32)

    if overlay.shape != page.shape:
        return None

    diff = np.abs(overlay - page)
    has_mask = diff.max(axis=2) > threshold

    rows = np.any(has_mask, axis=1)
    cols = np.any(has_mask, axis=0)
    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def find_crop_offset(page_path: str, crop_path: str, scale: float = 0.2):
    """Find where crop_img appears in page using template matching.

    Returns (x_offset, y_offset) in full-page coordinates.
    """
    page = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
    if page is None or crop is None:
        return None

    page_s = cv2.resize(page, None, fx=scale, fy=scale)
    crop_s = cv2.resize(crop, None, fx=scale, fy=scale)

    if crop_s.shape[0] > page_s.shape[0] or crop_s.shape[1] > page_s.shape[1]:
        return None

    result = cv2.matchTemplate(page_s, crop_s, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.5:
        return None

    return (int(max_loc[0] / scale), int(max_loc[1] / scale))


def project_mask_to_crop(
    rle_mask: np.ndarray,
    rle_size: tuple,
    overlay_bbox: tuple,
    crop_offset: tuple,
    crop_size: tuple,
) -> np.ndarray:
    """Project an RLE binary mask onto the step crop coordinate system.

    Args:
        rle_mask: Binary mask at RLE resolution (H_rle, W_rle).
        rle_size: (H_rle, W_rle) of the mask space.
        overlay_bbox: (x_min, y_min, x_max, y_max) of mask region in page space.
        crop_offset: (x_off, y_off) of the step crop in page space.
        crop_size: (W_crop, H_crop) of the step crop image.

    Returns:
        Binary mask at crop resolution (H_crop, W_crop).
    """
    rle_h, rle_w = rle_size
    ox_min, oy_min, ox_max, oy_max = overlay_bbox
    crop_x, crop_y = crop_offset
    crop_w, crop_h = crop_size

    # Affine: mask pixel (mx, my) -> page pixel (px, py)
    #   px = ox_min + mx * (ox_max - ox_min) / rle_w
    #   py = oy_min + my * (oy_max - oy_min) / rle_h
    # Then page -> crop: cx = px - crop_x, cy = py - crop_y
    # Combined: cx = (ox_min - crop_x) + mx * sx, cy = (oy_min - crop_y) + my * sy
    sx = (ox_max - ox_min) / rle_w
    sy = (oy_max - oy_min) / rle_h
    tx = ox_min - crop_x
    ty = oy_min - crop_y

    M = np.array([[sx, 0, tx], [0, sy, ty]], dtype=np.float32)
    projected = cv2.warpAffine(rle_mask, M, (crop_w, crop_h), flags=cv2.INTER_NEAREST)
    return projected


def encode_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Encode a binary mask to COCO RLE format."""
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def process_item(
    item: dict, step_idx: int, step: dict, images_dir: str, image_id: int
) -> tuple:
    """Process a single step: project masks and create COCO annotations.

    Returns:
        (image_info, list_of_annotations) or (None, []) if processing fails.
    """
    cat, name = item["category"], item["name"]
    page_id = step["page_id"]

    # Paths
    no_seg_path = os.path.join(MASK_DIR, cat, name, f"step_{step_idx}_no_seg.png")
    overlay_path = os.path.join(MASK_DIR, cat, name, f"step_{step_idx}_mask_overlay.png")
    page_path = os.path.join(DATA_DIR, "pdfs", cat, name, f"page_{page_id}.png")

    if not os.path.exists(no_seg_path):
        return None, []

    # Get crop image dimensions
    crop_img = Image.open(no_seg_path)
    crop_w, crop_h = crop_img.size

    # Decode RLE masks
    rle_masks = []
    for m in step["masks"]:
        rle_masks.append(decode_rle(m))

    rle_h, rle_w = step["masks"][0]["size"]

    # Try affine projection via overlay
    projected_masks = None
    if os.path.exists(overlay_path) and os.path.exists(page_path):
        overlay_bbox = find_mask_overlay_bbox(overlay_path, page_path)
        crop_offset = find_crop_offset(page_path, no_seg_path)

        if overlay_bbox is not None and crop_offset is not None:
            projected_masks = []
            for mask in rle_masks:
                proj = project_mask_to_crop(
                    mask, (rle_h, rle_w), overlay_bbox, crop_offset, (crop_w, crop_h)
                )
                projected_masks.append(proj)

    # Fallback: simple resize (ignores aspect ratio mismatch but still usable)
    if projected_masks is None:
        projected_masks = []
        for mask in rle_masks:
            resized = cv2.resize(
                mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
            )
            projected_masks.append(resized)

    # Copy image to dataset
    rel_name = f"{cat}_{name}_step_{step_idx}.png"
    dst_path = os.path.join(images_dir, rel_name)
    shutil.copy2(no_seg_path, dst_path)

    # Build COCO image info
    image_info = {
        "id": image_id,
        "file_name": rel_name,
        "width": crop_w,
        "height": crop_h,
        "category": cat,
        "furniture": name,
        "step_idx": step_idx,
    }

    # Build per-instance annotations
    annotations = []
    for i, proj_mask in enumerate(projected_masks):
        if proj_mask.sum() < 10:
            continue

        bbox = mask_to_bbox_xywh(proj_mask)
        rle = encode_mask_to_rle(proj_mask)
        area = int(proj_mask.sum())

        ann = {
            "image_id": image_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
            "part_id": step["parts"][i] if i < len(step["parts"]) else str(i),
        }
        annotations.append(ann)

    return image_info, annotations


def build_dataset(output_dir: str = None, create_zip: bool = True):
    """Build the full COCO-format dataset from main_data.json."""
    output_dir = output_dir or SAM2_DATASET_DIR

    with open(MAIN_DATA_JSON) as f:
        data = json.load(f)

    # Create output directories
    for split in ("train", "test"):
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)

    # Process all steps
    coco_data = {
        "train": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "part"}]},
        "test": {"images": [], "annotations": [], "categories": [{"id": 1, "name": "part"}]},
    }

    image_id = 0
    ann_id = 0
    stats = {"train": 0, "test": 0, "skipped": 0, "total_annotations": 0}

    for item in tqdm(data, desc="Processing furniture items"):
        cat, name = item["category"], item["name"]

        for step_idx, step in enumerate(item["steps"]):
            split = step.get("part_segmentation_split")
            if split not in ("train", "test"):
                stats["skipped"] += 1
                continue

            images_dir = os.path.join(output_dir, split, "images")
            image_info, annotations = process_item(
                item, step_idx, step, images_dir, image_id
            )

            if image_info is None or len(annotations) == 0:
                stats["skipped"] += 1
                continue

            coco_data[split]["images"].append(image_info)
            for ann in annotations:
                ann["id"] = ann_id
                coco_data[split]["annotations"].append(ann)
                ann_id += 1

            image_id += 1
            stats[split] += 1
            stats["total_annotations"] += len(annotations)

    # Save annotations
    for split in ("train", "test"):
        json_path = os.path.join(output_dir, split, "annotations.json")
        with open(json_path, "w") as f:
            json.dump(coco_data[split], f)
        print(f"Saved {split} annotations: {len(coco_data[split]['images'])} images, "
              f"{len(coco_data[split]['annotations'])} instances -> {json_path}")

    print(f"\nDataset stats: train={stats['train']}, test={stats['test']}, "
          f"skipped={stats['skipped']}, total_annotations={stats['total_annotations']}")

    # Create ZIP for Colab upload
    if create_zip:
        print(f"\nCreating ZIP archive: {SAM2_DATASET_ZIP}")
        with zipfile.ZipFile(SAM2_DATASET_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, os.path.dirname(output_dir))
                    zf.write(filepath, arcname)
        zip_size_mb = os.path.getsize(SAM2_DATASET_ZIP) / (1024 * 1024)
        print(f"ZIP created: {zip_size_mb:.1f} MB")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare SAM2 training dataset from main_data.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: data/sam2_dataset/)")
    parser.add_argument("--no_zip", action="store_true",
                        help="Skip creating ZIP archive")
    args = parser.parse_args()

    build_dataset(output_dir=args.output_dir, create_zip=not args.no_zip)


if __name__ == "__main__":
    main()
