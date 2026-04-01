"""
PyTorch Dataset for SAM2 fine-tuning on IKEA manual part segmentation.

Each sample is a single (image, instance_mask, box_prompt) tuple.
Images are resized to 1024x1024 for SAM2 input.
"""

import json
import os
import random

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from torch.utils.data import Dataset


class SAM2FineTuneDataset(Dataset):
    """Dataset that yields (image, mask, box_prompt) tuples for SAM2 training.

    Flattens per-image annotations into per-instance samples so each sample
    has exactly one binary mask and one bounding box prompt.
    """

    def __init__(
        self,
        annotations_json: str,
        images_dir: str = None,
        input_size: int = 1024,
        box_jitter_px: int = 10,
        augment: bool = False,
    ):
        with open(annotations_json) as f:
            coco = json.load(f)

        self.images_dir = images_dir
        if self.images_dir is None:
            self.images_dir = os.path.join(os.path.dirname(annotations_json), "images")

        self.input_size = input_size
        self.box_jitter_px = box_jitter_px
        self.augment = augment

        # Build lookup tables
        self.images = {img["id"]: img for img in coco["images"]}

        # Flatten: one sample per (image_id, annotation)
        self.samples = []
        for ann in coco["annotations"]:
            if ann["area"] < 10:
                continue
            self.samples.append(ann)

    def __len__(self):
        return len(self.samples)

    def _decode_mask(self, ann: dict, img_info: dict) -> np.ndarray:
        """Decode annotation mask to binary array."""
        seg = ann["segmentation"]
        if isinstance(seg, dict) and "counts" in seg:
            rle = {"size": seg["size"], "counts": seg["counts"].encode("utf-8") if isinstance(seg["counts"], str) else seg["counts"]}
            return mask_util.decode(rle)
        # Polygon format
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in seg:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask

    def _jitter_box(self, bbox: list, img_w: int, img_h: int) -> np.ndarray:
        """Add random jitter to bbox [x, y, w, h] and convert to [x1, y1, x2, y2]."""
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h

        if self.box_jitter_px > 0:
            j = self.box_jitter_px
            x1 += random.randint(-j, j)
            y1 += random.randint(-j, j)
            x2 += random.randint(-j, j)
            y2 += random.randint(-j, j)

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def __getitem__(self, idx):
        ann = self.samples[idx]
        img_info = self.images[ann["image_id"]]

        # Load image
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Decode mask
        mask = self._decode_mask(ann, img_info)

        # Augmentations (applied before resize)
        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            # Brightness/contrast jitter
            if random.random() < 0.3:
                alpha = random.uniform(0.8, 1.2)
                beta = random.randint(-20, 20)
                image = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)

        # Get box prompt before resize (in original coords)
        bbox = ann["bbox"]  # [x, y, w, h]
        box = self._jitter_box(bbox, orig_w, orig_h)

        # Resize to input_size x input_size
        scale_x = self.input_size / orig_w
        scale_y = self.input_size / orig_h
        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        # Scale box to new coords
        box[0] *= scale_x
        box[1] *= scale_y
        box[2] *= scale_x
        box[3] *= scale_y

        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
        box_tensor = torch.from_numpy(box)  # (4,)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "box": box_tensor,
            "image_id": ann["image_id"],
            "original_size": (orig_h, orig_w),
        }
