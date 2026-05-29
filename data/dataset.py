cat > data/dataset.py << 'ENDOFFILE'
"""
data/dataset.py — MVTec AD dataset loader for DefectFill.
"""
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


class MVTecSample:
    __slots__ = ("image_path", "mask_path", "object_name", "defect_type")

    def __init__(self, image_path, mask_path, object_name, defect_type):
        self.image_path  = image_path
        self.mask_path   = mask_path
        self.object_name = object_name
        self.defect_type = defect_type


def build_defect_samples(dataset_root, object_name, defect_type, config):
    root     = Path(dataset_root)
    img_dir  = root / object_name / "test" / defect_type
    mask_dir = root / object_name / "ground_truth" / defect_type

    if not img_dir.exists():
        print(f"  [WARN] Not found: {img_dir}")
        return [], []

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / (img_path.stem + "_mask.png")
        if not mask_path.exists():
            mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f"  [WARN] No mask for: {img_path.name}")
            continue
        samples.append(MVTecSample(
            str(img_path), str(mask_path), object_name, defect_type
        ))

    if not samples:
        return [], []

    random.seed(config["dataset"]["seed"])
    random.shuffle(samples)

    ref_ratio = config["dataset"]["reference_split"]
    split_idx = max(1, int(len(samples) * ref_ratio))
    reference = samples[:split_idx]
    target    = samples[split_idx:]

    print(f"  [{object_name}/{defect_type}] "
          f"Total={len(samples)}, Reference={len(reference)}, Target={len(target)}")
    return reference, target


def get_normal_images(dataset_root, object_name):
    normal_dir = Path(dataset_root) / object_name / "train" / "good"
    if not normal_dir.exists():
        return []
    return [str(p) for p in sorted(normal_dir.glob("*.png"))]


class MVTecDefectDataset(Dataset):
    def __init__(self, samples: List[MVTecSample], config: dict, augment: bool = True):
        self.samples  = samples
        self.img_size = config["dataset"]["img_size"]
        self.augment  = augment
        self.config   = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {sample.image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img_rgb.ndim == 2 or (img_rgb.ndim == 3 and img_rgb.shape[2] == 1):
            img_rgb = gray_to_rgb(img_rgb)

        mask = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {sample.mask_path}")
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)

        if self.augment:
            img_rgb, mask = self._augment(img_rgb, mask)

        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        mask    = cv2.resize(mask, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)

        img_t  = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
        mask_t = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return {
            "pixel_values": img_t,
            "mask":         mask_t,
            "object_name":  sample.object_name,
            "defect_type":  sample.defect_type,
            "image_path":   sample.image_path,
        }

    def _augment(self, img, mask):
        import random as rnd
        if rnd.random() > 0.5:
            img  = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        if rnd.random() > 0.5:
            img  = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)
        if rnd.random() > 0.4:
            alpha = rnd.uniform(0.8, 1.2)
            beta  = rnd.randint(-15, 15)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return img, mask
ENDOFFILE