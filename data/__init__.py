"""
data/
─────
Dataset utilities for the Steel Pipe Weld Defect dataset.

  dataset.py   — PyTorch Dataset, YOLO annotation parser, bbox→mask converter
  preprocess.py — One-time preprocessing: split into reference/target sets
  augment.py   — X-ray-appropriate image augmentations
"""

from .dataset import (
    WeldDefectSample,
    WeldDefectDataset,
    build_class_samples,
    build_normal_samples,
    yolo_bbox_to_pixel,
    bbox_to_mask,
    gray_to_rgb,
    load_config,
)

from .augment import (
    AugmentForTraining,
    AugmentForDownstream,
    tta_transforms,
    tta_merge_masks,
    brightness_contrast,
    add_radiographic_noise,
    gamma_correction,
    elastic_deform,
    hflip,
    vflip,
)

__all__ = [
    # dataset
    "WeldDefectSample", "WeldDefectDataset",
    "build_class_samples", "build_normal_samples",
    "yolo_bbox_to_pixel", "bbox_to_mask", "gray_to_rgb", "load_config",
    # augment
    "AugmentForTraining", "AugmentForDownstream",
    "tta_transforms", "tta_merge_masks",
    "brightness_contrast", "add_radiographic_noise", "gamma_correction",
    "elastic_deform", "hflip", "vflip",
]
