import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MAIN_DATA_JSON = os.path.join(DATA_DIR, "main_data.json")
MASK_DIR = os.path.join(DATA_DIR, "mask")

# Dataset output
SAM2_DATASET_DIR = os.path.join(DATA_DIR, "sam2_dataset")
SAM2_DATASET_ZIP = os.path.join(DATA_DIR, "sam2_dataset.zip")

# SAM2 model configs
SAM2_MODELS = {
    "tiny":      {"config": "configs/sam2.1/sam2.1_hiera_t.yaml",  "checkpoint": "sam2.1_t.pt"},
    "small":     {"config": "configs/sam2.1/sam2.1_hiera_s.yaml",  "checkpoint": "sam2.1_s.pt"},
    "base_plus": {"config": "configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoint": "sam2.1_b.pt"},
    "large":     {"config": "configs/sam2.1/sam2.1_hiera_l.yaml",  "checkpoint": "sam2.1_l.pt"},
}

# Training defaults
TRAIN_DEFAULTS = {
    "mode": "full",           # "decoder" or "full"
    "model_size": "base_plus",
    "epochs": 50,
    "batch_size": 8,
    "backbone_lr": 1e-6,
    "decoder_lr": 1e-4,
    "weight_decay": 0.01,
    "focal_weight": 20.0,
    "dice_weight": 1.0,
    "box_jitter_px": 10,
    "input_size": 1024,
    "warmup_steps": 500,
    "save_every": 5,
}

# Inference defaults
INFER_DEFAULTS = {
    "model_size": "base_plus",
    "checkpoint": "sam2_ikea_best.pt",
    "mask_threshold": 0.5,
    "min_mask_area": 100,
    "nms_iou_threshold": 0.8,
}

# Generated mask output (for pipeline integration)
SAM2_MASK_DIR = os.path.join(DATA_DIR, "sam2_mask")
