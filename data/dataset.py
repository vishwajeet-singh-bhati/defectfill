"""
data/dataset.py
───────────────
PyTorch Dataset for the Steel Pipe Weld Defect dataset.

Responsibilities:
  1. Parse YOLO-format annotation files (.txt with normalized bbox coords)
  2. Convert bounding boxes → binary inpainting masks
  3. Return (image, mask, class_id) triplets ready for DefectFill training
  4. Handle grayscale-to-RGB conversion for Stable Diffusion compatibility

Why bounding boxes → masks?
  The dataset provides bbox annotations (YOLO format), but DefectFill needs
  per-pixel binary masks indicating the defect region.  We convert each bbox
  to a filled rectangle mask, optionally dilated to give the inpainting model
  a slightly larger area to work with (controlled by config.mask_mode).

Why grayscale → RGB?
  Stable Diffusion was pre-trained on RGB images.  X-ray weld images are
  single-channel grayscale.  We tile the single channel across R, G, B so
  the pretrained weights remain meaningful while accepting our domain.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import yaml


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def yolo_bbox_to_pixel(
    bbox_norm: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO normalized bbox (cx, cy, w, h) to pixel coords (x1,y1,x2,y2).

    YOLO stores bounding boxes as:
        cx, cy  — centre x, centre y  (normalised 0-1)
        w, h    — width, height       (normalised 0-1)

    Returns pixel-space (x1, y1, x2, y2) — top-left and bottom-right corners.
    """
    cx, cy, bw, bh = bbox_norm
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w - 1, x2), min(img_h - 1, y2)
    return x1, y1, x2, y2


def bbox_to_mask(
    bboxes: List[Tuple[int, int, int, int]],
    img_h: int,
    img_w: int,
    dilation_factor: float = 1.0,
) -> np.ndarray:
    """
    Convert a list of pixel bounding boxes to a single binary mask.

    Args:
        bboxes:           List of (x1,y1,x2,y2) pixel coordinates.
        img_h, img_w:     Image dimensions.
        dilation_factor:  Expand each box by this factor (e.g. 1.15 = 15% larger).
                          Gives the inpainting model room to blend edges.

    Returns:
        mask: uint8 numpy array, shape (H, W), values in {0, 255}.
              255 = defect region to be inpainted.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for (x1, y1, x2, y2) in bboxes:
        if dilation_factor != 1.0:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            hw = (x2 - x1) / 2 * dilation_factor
            hh = (y2 - y1) / 2 * dilation_factor
            x1 = max(0, int(cx - hw))
            y1 = max(0, int(cy - hh))
            x2 = min(img_w - 1, int(cx + hw))
            y2 = min(img_h - 1, int(cy + hh))
        mask[y1:y2, x1:x2] = 255
    return mask


def gray_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image (H,W) or (H,W,1) to (H,W,3) by tiling the channel.
    This preserves intensity information across all three channels so that the
    CLIP text encoder and U-Net can process it without domain mismatch artifacts.
    """
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img  # already RGB


# ─── Core Dataset ─────────────────────────────────────────────────────────────

class WeldDefectSample:
    """Lightweight data container for a single annotated defect sample."""
    __slots__ = ("image_path", "label_path", "class_ids", "bboxes_norm")

    def __init__(self, image_path: str, label_path: str):
        self.image_path = image_path
        self.label_path = label_path
        self.class_ids: List[int] = []
        self.bboxes_norm: List[Tuple[float, float, float, float]] = []
        self._parse_label()

    def _parse_label(self):
        """Read YOLO label file: each line → class_id cx cy w h."""
        if not os.path.exists(self.label_path):
            return
        with open(self.label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                bbox = tuple(float(x) for x in parts[1:])
                self.class_ids.append(cls)
                self.bboxes_norm.append(bbox)


class WeldDefectDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning DefectFill on the steel weld dataset.

    Each item returned is a dict with:
      - "pixel_values": torch.Tensor (3, H, W), normalised to [-1, 1]
      - "mask":         torch.Tensor (1, H, W), float, 1=defect, 0=background
      - "class_id":     int
      - "class_name":   str

    Usage:
        ds = WeldDefectDataset(samples, class_id=3, config=cfg)
        loader = DataLoader(ds, batch_size=1, shuffle=True)
    """

    def __init__(
        self,
        samples: List[WeldDefectSample],
        class_id: int,
        config: dict,
        augment: bool = True,
    ):
        """
        Args:
            samples:   List of WeldDefectSample for this specific defect class.
            class_id:  Defect class index (0-7).
            config:    Full config dict (loaded from config.yaml).
            augment:   Whether to apply data augmentation.
        """
        self.samples = samples
        self.class_id = class_id
        self.class_name = config["dataset"]["class_names"][class_id]
        self.img_size = config["dataset"]["img_size"]
        self.dilation = config["dataset"].get("dilation_factor", 1.15)
        self.augment = augment
        self.config = config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {sample.image_path}")

        img_h, img_w = img_bgr.shape[:2]

        # Convert BGR → grayscale (X-ray) → RGB (SD-compatible)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb = gray_to_rgb(img_gray)  # (H, W, 3) uint8

        # ── Build mask for this class's defects only ──────────────────────────
        target_bboxes_px = []
        for cid, bbox_norm in zip(sample.class_ids, sample.bboxes_norm):
            if cid == self.class_id:
                px = yolo_bbox_to_pixel(bbox_norm, img_w, img_h)
                target_bboxes_px.append(px)

        mask = bbox_to_mask(target_bboxes_px, img_h, img_w, self.dilation)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            img_rgb, mask = self._augment(img_rgb, mask)

        # ── Resize ────────────────────────────────────────────────────────────
        img_rgb = cv2.resize(img_rgb, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # ── Convert to tensors ────────────────────────────────────────────────
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 127.5 - 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return {
            "pixel_values": img_tensor,   # (3, H, W), range [-1, 1]
            "mask": mask_tensor,           # (1, H, W), range [0, 1]
            "class_id": self.class_id,
            "class_name": self.class_name,
            "image_path": sample.image_path,
        }

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        """
        Apply augmentations appropriate for X-ray images.

        X-ray augmentations DIFFER from natural images:
          ✓ Horizontal flip (physically valid — defects can appear anywhere)
          ✓ Vertical flip   (valid for X-ray scans)
          ✓ Brightness/contrast adjustment (simulates different exposure settings)
          ✗ Colour jitter   (irrelevant — grayscale domain)
          ✗ Saturation      (same)
        """
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        if random.random() > 0.3:
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-15, 15)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return img, mask


# ─── Dataset Builder ──────────────────────────────────────────────────────────

def build_class_samples(
    dataset_root: str,
    class_id: int,
    config: dict,
) -> Tuple[List[WeldDefectSample], List[WeldDefectSample]]:
    """
    Collect all image-label pairs that contain at least one annotation of
    class_id, then split them into reference (1/3) and target (2/3) sets.

    Globs across ALL split subdirectories (train2021, val2021, etc.) found
    under images_dir, so the full dataset is pooled before our own split.

    Reference set  → used to LEARN the defect concept (fine-tune DefectFill)
    Target set     → defect-free backgrounds for GENERATING new images

    Args:
        dataset_root: Path to dataset folder (should be .../yolo/).
        class_id:     Defect class index (0-7).
        config:       Config dict.

    Returns:
        (reference_samples, target_samples)
    """
    images_dir = Path(dataset_root) / config["dataset"]["images_dir"]
    labels_dir = Path(dataset_root) / config["dataset"]["labels_dir"]
    ext = config["dataset"]["image_ext"]
    ref_ratio = config["dataset"]["reference_split"]
    seed = config["dataset"]["seed"]

    all_samples: List[WeldDefectSample] = []

    # FIX: glob recursively through subdirectories (train2021, val2021, etc.)
    # The dataset stores images in images/train2021/ and images/val2021/
    # rather than directly under images/, so a flat glob(*ext) finds nothing.
    for img_path in sorted(images_dir.rglob(f"*{ext}")):
        # Mirror the subdirectory structure to find the matching label file.
        # e.g. images/train2021/foo.jpg → labels/train2021/foo.txt
        rel = img_path.relative_to(images_dir)          # FIX: preserve subdir
        label_path = labels_dir / rel.with_suffix(".txt")  # FIX: match subdir
        sample = WeldDefectSample(str(img_path), str(label_path))
        if class_id in sample.class_ids:
            all_samples.append(sample)

    if not all_samples:
        return [], []

    random.seed(seed)
    random.shuffle(all_samples)

    split_idx = max(1, int(len(all_samples) * ref_ratio))
    reference = all_samples[:split_idx]
    target = all_samples[split_idx:]

    print(
        f"[{config['dataset']['class_names'][class_id]}] "
        f"Total={len(all_samples)}, Reference={len(reference)}, Target={len(target)}"
    )
    return reference, target


def build_normal_samples(dataset_root: str, config: dict) -> List[str]:
    """
    Collect paths to all images across all split subdirectories.
    Returns a list of image file paths.
    """
    images_dir = Path(dataset_root) / config["dataset"]["images_dir"]
    ext = config["dataset"]["image_ext"]
    # FIX: rglob to pick up train2021/ and val2021/ subdirectories
    return [str(p) for p in sorted(images_dir.rglob(f"*{ext}"))]