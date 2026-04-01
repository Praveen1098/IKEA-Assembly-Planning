"""
Pipeline integration: generate SAM2 mask PNGs in the naming convention
expected by Stage 2 and Stage 3 of the VLM assembly plan pipeline.

Stage 2 loads files matching: step_{N}_no_seg_numbered.png (or step_{N}_no_seg.png)
Stage 3 loads files matching: step_{N}_no_seg_numbered.png

This module wraps predict.py to produce correctly-named outputs.

Usage:
    cd VLM_assembly_plan_gen
    python -m part_segmentation.integrate \
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

from tqdm import tqdm

from part_segmentation.config import (
    INFER_DEFAULTS,
    MAIN_DATA_JSON,
    SAM2_MASK_DIR,
    SAM2_MODELS,
)
from part_segmentation.predict import (
    generate_masks_for_item,
    load_sam2_model,
)


def integrate_all(
    checkpoint: str,
    model_size: str = "base_plus",
    output_dir: str = None,
    start: int = 0,
    end: int = 102,
    device: str = "cuda",
):
    """Generate mask PNGs for all items, ready for pipeline consumption.

    The output directory structure matches data/mask/:
        output_dir/{category}/{name}/step_{N}_no_seg.png
        output_dir/{category}/{name}/step_{N}_no_seg_numbered.png
    """
    output_dir = output_dir or SAM2_MASK_DIR

    with open(MAIN_DATA_JSON) as f:
        data = json.load(f)

    predictor = load_sam2_model(checkpoint, model_size, device)

    for idx in tqdm(range(start, min(end, len(data))), desc="Integrating SAM2 masks"):
        item = data[idx]
        generate_masks_for_item(
            predictor,
            item,
            output_dir,
            use_gt_boxes=False,  # Use auto mask generation for new items
            mask_threshold=INFER_DEFAULTS["mask_threshold"],
            min_mask_area=INFER_DEFAULTS["min_mask_area"],
            nms_iou=INFER_DEFAULTS["nms_iou_threshold"],
        )

    print(f"\nMasks saved to {output_dir}")
    print("Use with: python inference/run.py --use_sam2 ...")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM2 masks for pipeline integration"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default=INFER_DEFAULTS["model_size"],
                        choices=list(SAM2_MODELS.keys()))
    parser.add_argument("--output_dir", type=str, default=SAM2_MASK_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=102)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    integrate_all(
        checkpoint=args.checkpoint,
        model_size=args.model_size,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        device=args.device,
    )


if __name__ == "__main__":
    main()
