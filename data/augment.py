"""
data/augment.py
───────────────
X-ray weld image augmentation strategies tailored for DefectFill training.

Why a dedicated augment module?
  Standard augmentations (colour jitter, saturation) are designed for natural
  RGB photography.  X-ray weld images have completely different properties:
    • Single-channel grayscale (luminance only — no hue/saturation)
    • Radiographic noise (Poisson + Gaussian, distinct from camera noise)
    • Fixed geometry (the weld seam is always approximately centred)
    • Defects are small and sparse (augmentation must preserve defect region)

  Inappropriate augmentations (aggressive colour jitter, large crops that
  remove the defect) would damage training signal.

Two augmentation contexts used in this project:
  1. AugmentForTraining — applied per sample in WeldDefectDataset during
     DefectFill fine-tuning.  Conservative: flip + brightness/contrast.

  2. AugmentForDownstream — applied during ResNet-34 / UNet training to
     increase variety in the generated+real training set.  More aggressive
     because the downstream model needs stronger regularisation.

All functions operate on numpy arrays (uint8) and return numpy arrays,
matching the cv2 ecosystem used in dataset.py.
"""

import random
from typing import Tuple, Optional

import cv2
import numpy as np


# ─── Primitive Augmentations ─────────────────────────────────────────────────

def hflip(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horizontal flip — physically valid for X-ray scans.
    The pipe weld can appear from either side of the detector.
    """
    return cv2.flip(image, 1), cv2.flip(mask, 1)


def vflip(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vertical flip — valid because the weld scanner can be oriented either way.
    """
    return cv2.flip(image, 0), cv2.flip(mask, 0)


def brightness_contrast(
    image: np.ndarray,
    alpha_range: Tuple[float, float] = (0.75, 1.25),
    beta_range: Tuple[int, int] = (-20, 20),
) -> np.ndarray:
    """
    Random linear brightness/contrast adjustment.

    Models: pixel_out = alpha * pixel_in + beta

    Physical meaning:
      alpha >1 → higher contrast  (stronger X-ray exposure)
      alpha <1 → lower contrast   (weaker exposure or thicker material)
      beta    → global brightness shift (detector offset variation)

    Mask is NOT affected — it's binary and defined by annotation, not pixel values.
    """
    alpha = random.uniform(*alpha_range)
    beta = random.randint(*beta_range)
    adjusted = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return adjusted


def add_radiographic_noise(
    image: np.ndarray,
    intensity: float = 0.02,
) -> np.ndarray:
    """
    Add Gaussian noise to simulate radiographic detector noise.

    Real X-ray detectors produce Poisson-distributed photon noise.  At the
    image-level this approximates Gaussian noise proportional to image intensity.
    We simulate with additive Gaussian noise (intensity controls standard deviation
    as a fraction of the dynamic range).

    intensity=0.02 → std = 0.02 * 255 ≈ 5 grey levels (subtle, realistic)
    """
    sigma = intensity * 255
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def gamma_correction(
    image: np.ndarray,
    gamma_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """
    Random gamma correction — simulates nonlinearity in X-ray detector response.

    pixel_out = (pixel_in / 255)^gamma * 255

    gamma < 1 → brighten dark regions (more sensitive detector)
    gamma > 1 → darken, increase contrast in bright regions
    """
    gamma = random.uniform(*gamma_range)
    lut = np.array(
        [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, lut)


def elastic_deform(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 30.0,
    sigma: float = 6.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elastic deformation — simulates slight geometric variation in weld geometry.

    Models the fact that pipe welds have slight curvature variations and the
    X-ray projection angle is never perfectly reproducible.

    Uses random displacement fields (Simard et al. 2003).
    Applied identically to image and mask to preserve correspondence.

    Args:
        alpha:  Magnitude of displacement (pixels).  30 ≈ subtle warp.
        sigma:  Smoothness of displacement field.    6  ≈ smooth, realistic.
    """
    h, w = image.shape[:2]
    # Random displacement field, smoothed with Gaussian
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma
    ) * alpha

    # Build absolute map
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    img_warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    msk_warped = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return img_warped, msk_warped


def cutout(
    image: np.ndarray,
    mask: np.ndarray,
    n_holes: int = 3,
    max_size: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random rectangular cutout in the BACKGROUND (non-mask) region.

    Forces the downstream model to not rely on local context outside the
    defect region.  Applied only outside the mask to avoid destroying
    the defect signal.

    Inspired by: DeVries & Taylor (2017) "Improved Regularization of
    Convolutional Neural Networks with Cutout".
    """
    h, w = image.shape[:2]
    result = image.copy()
    for _ in range(n_holes):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        hw = random.randint(5, max_size) // 2
        hh = random.randint(5, max_size) // 2
        x1, x2 = max(0, cx - hw), min(w, cx + hw)
        y1, y2 = max(0, cy - hh), min(h, cy + hh)
        # Only cut outside the defect mask
        cut_region = mask[y1:y2, x1:x2]
        no_defect = cut_region < 128    # True where not defect
        result[y1:y2, x1:x2][no_defect] = 0   # zero out background patch
    return result, mask


# ─── Composed Augmentation Pipelines ─────────────────────────────────────────

class AugmentForTraining:
    """
    Conservative augmentation pipeline for DefectFill fine-tuning.

    Used in:  data/dataset.py → WeldDefectDataset (augment=True)

    Philosophy: The reference set is very small (10-17 images per class).
    We must augment to prevent overfitting, but cannot be aggressive because:
      1. The defect must remain visible in the mask region.
      2. The inpainting model already has strong priors — we just need
         enough variation in reference images to avoid memorisation.

    Probability of each augmentation is kept moderate.
    """

    def __call__(
        self,
        image: np.ndarray,   # (H, W) or (H, W, 3), uint8
        mask: np.ndarray,    # (H, W), uint8, 255=defect
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Ensure 3-channel image
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        # Horizontal flip (50%)
        if random.random() > 0.5:
            image, mask = hflip(image, mask)

        # Vertical flip (30%)
        if random.random() > 0.7:
            image, mask = vflip(image, mask)

        # Brightness/contrast (60%)
        if random.random() > 0.4:
            image = brightness_contrast(image, alpha_range=(0.8, 1.2), beta_range=(-15, 15))

        # Gamma correction (30%)
        if random.random() > 0.7:
            image = gamma_correction(image, gamma_range=(0.85, 1.15))

        # Radiographic noise (40%)
        if random.random() > 0.6:
            image = add_radiographic_noise(image, intensity=0.015)

        return image, mask


class AugmentForDownstream:
    """
    Stronger augmentation pipeline for ResNet-34 and UNet training.

    Used in:  inspect/classifier.py, inspect/localizer.py

    The downstream models train on generated images which, while realistic,
    still have lower diversity than real collected data.  Stronger augmentation
    compensates and helps the model generalise to real test images.

    Includes elastic deformation and cutout which are too aggressive for
    the fine-tuning stage but appropriate for downstream supervised training.
    """

    def __init__(self, img_size: int = 512):
        self.img_size = img_size

    def __call__(
        self,
        image: np.ndarray,   # (H, W, 3), uint8
        mask: Optional[np.ndarray] = None,  # (H, W), uint8 — None if classification
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        # Flips (50% each)
        if random.random() > 0.5:
            image, mask = (hflip(image, mask) if mask is not None
                           else (cv2.flip(image, 1), None))
        if random.random() > 0.5:
            image, mask = (vflip(image, mask) if mask is not None
                           else (cv2.flip(image, 0), None))

        # Brightness/contrast (70%)
        if random.random() > 0.3:
            image = brightness_contrast(image, alpha_range=(0.7, 1.3), beta_range=(-25, 25))

        # Gamma correction (50%)
        if random.random() > 0.5:
            image = gamma_correction(image, gamma_range=(0.75, 1.25))

        # Radiographic noise (50%)
        if random.random() > 0.5:
            image = add_radiographic_noise(image, intensity=0.025)

        # Elastic deformation (30%) — only when mask is available
        if random.random() > 0.7 and mask is not None:
            image, mask = elastic_deform(image, mask, alpha=25.0, sigma=5.0)

        # Cutout (40%)
        if random.random() > 0.6:
            if mask is not None:
                image, mask = cutout(image, mask, n_holes=3, max_size=25)
            else:
                # Create dummy mask for cutout (all background)
                dummy_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                image, _ = cutout(image, dummy_mask, n_holes=3, max_size=25)

        # Random rotation (±5°) — small rotation (30%)
        if random.random() > 0.7:
            angle = random.uniform(-5, 5)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT)

        return image, mask


# ─── Test-Time Augmentation (TTA) for evaluation ──────────────────────────────

def tta_transforms(image: np.ndarray) -> list:
    """
    Test-Time Augmentation: generate multiple views of one image at inference.

    For evaluation on real test images, averaging predictions over multiple
    augmented views reduces variance and typically improves localisation AUROC.

    Returns list of 4 augmented variants: original, hflip, vflip, both.
    """
    img3 = image if image.ndim == 3 else np.stack([image]*3, axis=-1)
    variants = [
        img3,
        cv2.flip(img3, 1),
        cv2.flip(img3, 0),
        cv2.flip(cv2.flip(img3, 1), 0),
    ]
    return variants


def tta_merge_masks(masks: list) -> np.ndarray:
    """
    Merge TTA prediction masks back to original orientation and average.

    Args:
        masks: List of 4 prediction arrays (H, W), corresponding to tta_transforms output.
    Returns:
        Averaged mask in original orientation.
    """
    # Undo the flips applied in tta_transforms
    inv_transforms = [
        lambda m: m,
        lambda m: cv2.flip(m, 1),
        lambda m: cv2.flip(m, 0),
        lambda m: cv2.flip(cv2.flip(m, 1), 0),
    ]
    unflipped = [f(m) for f, m in zip(inv_transforms, masks)]
    return np.mean(np.stack(unflipped, axis=0), axis=0)
