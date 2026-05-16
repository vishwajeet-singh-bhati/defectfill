"""
utils/metrics.py
────────────────
All evaluation metrics used in the DefectFill paper, implemented for
the Steel Weld Defect dataset.

Metric overview:

  Generation Quality Metrics:
  ┌──────────────┬──────────────────────────────────────────────────────┐
  │ KID ↓        │ Kernel Inception Distance — measures quality of      │
  │              │ generated distribution vs. real distribution.         │
  │              │ Lower = more realistic. Preferred over FID for small  │
  │              │ datasets (unbiased estimator).                        │
  ├──────────────┼──────────────────────────────────────────────────────┤
  │ IC-LPIPS ↑   │ Intra-Class LPIPS — measures diversity WITHIN the    │
  │              │ generated set. Higher = more diverse. Prevents mode   │
  │              │ collapse: a model that generates one perfect defect   │
  │              │ repeatedly would score 0.                             │
  └──────────────┴──────────────────────────────────────────────────────┘

  Downstream Task Metrics (Localisation):
  ┌──────────────┬──────────────────────────────────────────────────────┐
  │ AUROC ↑      │ Area Under ROC Curve — threshold-free ranking metric  │
  │ AP ↑         │ Average Precision — area under P-R curve             │
  │ F1-max ↑     │ Best F1 across all thresholds                        │
  │ PRO ↑        │ Per-Region Overlap — rewards correctly localising     │
  │              │ even small defect regions proportionally.             │
  └──────────────┴──────────────────────────────────────────────────────┘

  Downstream Task Metrics (Classification):
  ┌──────────────┬──────────────────────────────────────────────────────┐
  │ Accuracy ↑   │ Per-class and overall top-1 accuracy                 │
  └──────────────┴──────────────────────────────────────────────────────┘
"""

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ─── Inception Feature Extractor ──────────────────────────────────────────────

class InceptionFeatureExtractor:
    """
    Extract pool3 features from InceptionV3 for KID and IC-LPIPS computation.

    InceptionV3 pool3 features (2048-dim) are the standard feature space
    for image quality metrics.  They capture both low-level texture and
    high-level semantics, making them sensitive to both realism and diversity.
    """

    def __init__(self, device: str = "cpu"):
        import torchvision.models as tv_models
        self.device = device

        inception = tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.DEFAULT,
            aux_logits=False,
        )
        # Remove classification head — we only want pool3 features
        self.model = torch.nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
        ).to(device).eval()

        # InceptionV3 preprocessing: resize to 299×299, normalise to [-1,1]
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def extract(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Extract 2048-dim InceptionV3 pool3 features for a list of PIL images.

        Args:
            images:     List of PIL RGB images.
            batch_size: Inference batch size.

        Returns:
            features: (N, 2048) float32 numpy array.
        """
        all_features = []
        for i in range(0, len(images), batch_size):
            batch_pil = images[i: i + batch_size]
            batch_t = torch.stack([self.transform(img.convert("RGB")) for img in batch_pil])
            batch_t = batch_t.to(self.device)
            feats = self.model(batch_t).cpu().numpy()
            all_features.append(feats)
        return np.concatenate(all_features, axis=0)  # (N, 2048)


# ─── KID (Kernel Inception Distance) ─────────────────────────────────────────

def compute_kid(
    real_features: np.ndarray,     # (N_real, 2048)
    fake_features: np.ndarray,     # (N_fake, 2048)
    num_subsets: int = 100,
    max_subset_size: int = 1000,
) -> Tuple[float, float]:
    """
    Compute KID using polynomial kernel MMD with subsampling.

    KID = MMD²(real_feats, fake_feats) using polynomial kernel k(x,y)=(x·y/d + 1)³
    where d = feature dimension (2048).

    Advantages over FID:
      • Unbiased estimator — accurate even with few samples (important for
        scarce classes like bite-edge).
      • Returns mean ± std over subsets, giving confidence interval.
      • Does NOT assume Gaussian distribution (FID's key assumption).

    Reference: Binkowski et al. "Demystifying MMD GANs" (2018).

    Returns:
        (kid_mean, kid_std) — both × 10³ for readability (paper convention).
    """
    n = min(min(len(real_features), len(fake_features)), max_subset_size)
    scores = []
    for _ in range(num_subsets):
        real_sub = real_features[np.random.choice(len(real_features), n, replace=False)]
        fake_sub = fake_features[np.random.choice(len(fake_features), n, replace=False)]
        scores.append(_mmd_poly(real_sub, fake_sub))
    scores = np.array(scores) * 1000  # scale for readability
    return float(scores.mean()), float(scores.std())


def _mmd_poly(x: np.ndarray, y: np.ndarray) -> float:
    """
    Polynomial kernel Maximum Mean Discrepancy between sets x and y.
    k(a, b) = (a·b/d + 1)^3
    MMD² = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
    """
    d = x.shape[1]
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    def poly_k(a, b):
        return ((a @ b.T) / d + 1) ** 3

    kxx = poly_k(x, x)
    kyy = poly_k(y, y)
    kxy = poly_k(x, y)

    n = len(x)
    # Unbiased MMD²: exclude diagonal terms in kxx and kyy
    mask = torch.eye(n, dtype=torch.bool)
    mmd = (kxx[~mask].mean() + kyy[~mask].mean() - 2 * kxy.mean()).item()
    return mmd


# ─── IC-LPIPS (Intra-Class LPIPS) ────────────────────────────────────────────

def compute_ic_lpips(
    images: List[torch.Tensor],   # List of (1, 3, H, W) tensors in [-1, 1]
    device: str = "cpu",
    num_pairs: int = 100,
) -> float:
    """
    Compute Intra-Class LPIPS (IC-LPIPS) — measures diversity within generated set.

    IC-LPIPS = mean LPIPS distance over random pairs from the same class.
    Higher IC-LPIPS = more diverse set = harder to mode-collapse.

    Reference: Ojha et al. "Few-Shot Image Generation via Cross-Domain
    Correspondence" (CVPR 2021).

    Args:
        images:    List of generated images as (1,3,H,W) tensors.
        device:    Compute device.
        num_pairs: Number of random pairs to sample for efficiency.

    Returns:
        ic_lpips: Mean pairwise LPIPS distance (float).
    """
    import lpips
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    n = len(images)
    if n < 2:
        return 0.0

    scores = []
    for _ in range(num_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        img_i = images[i].to(device)
        img_j = images[j].to(device)
        with torch.no_grad():
            d = lpips_fn(img_i, img_j).item()
        scores.append(d)
    return float(np.mean(scores))


# ─── Localisation Metrics ─────────────────────────────────────────────────────

def compute_pixel_auroc(
    pred_maps: List[np.ndarray],    # List of (H, W) float probability maps
    gt_masks:  List[np.ndarray],    # List of (H, W) binary ground-truth masks
) -> float:
    """
    Pixel-level AUROC for defect localisation.

    Flattens all pixel predictions and labels, then computes ROC AUC.
    A score of 1.0 means perfect separation of defect vs. background pixels.
    """
    from sklearn.metrics import roc_auc_score
    all_pred = np.concatenate([m.flatten() for m in pred_maps])
    all_gt   = np.concatenate([m.flatten() for m in gt_masks]).astype(int)
    if all_gt.sum() == 0 or all_gt.sum() == len(all_gt):
        return float("nan")
    return float(roc_auc_score(all_gt, all_pred))


def compute_pixel_ap(
    pred_maps: List[np.ndarray],
    gt_masks:  List[np.ndarray],
) -> float:
    """
    Average Precision (area under precision-recall curve) at pixel level.

    More informative than AUROC when class imbalance is extreme (which is
    always the case for defect localisation — most pixels are background).
    """
    from sklearn.metrics import average_precision_score
    all_pred = np.concatenate([m.flatten() for m in pred_maps])
    all_gt   = np.concatenate([m.flatten() for m in gt_masks]).astype(int)
    if all_gt.sum() == 0:
        return float("nan")
    return float(average_precision_score(all_gt, all_pred))


def compute_f1_max(
    pred_maps: List[np.ndarray],
    gt_masks:  List[np.ndarray],
    num_thresholds: int = 100,
) -> Tuple[float, float]:
    """
    Maximum F1 score over all thresholds.

    Sweeps thresholds from 0 to 1 and returns the best F1 and the threshold
    that achieves it.  Unlike a fixed threshold, F1-max reflects the model's
    best-case discriminative ability.

    Returns:
        (f1_max, best_threshold)
    """
    all_pred = np.concatenate([m.flatten() for m in pred_maps])
    all_gt   = np.concatenate([m.flatten() for m in gt_masks]).astype(int)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, num_thresholds):
        pred_bin = (all_pred >= thr).astype(int)
        tp = ((pred_bin == 1) & (all_gt == 1)).sum()
        fp = ((pred_bin == 1) & (all_gt == 0)).sum()
        fn = ((pred_bin == 0) & (all_gt == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_f1), float(best_thr)


def compute_pro(
    pred_maps: List[np.ndarray],
    gt_masks:  List[np.ndarray],
    num_thresholds: int = 100,
    fpr_limit: float = 0.3,
) -> float:
    """
    Per-Region Overlap (PRO) — the localisation metric most sensitive to
    correctly identifying ALL defect regions, including small ones.

    Algorithm:
      1. For each threshold t, binarise pred_maps.
      2. For each connected defect REGION in gt_mask, compute overlap fraction
         = |pred ∩ region| / |region|.  Each region contributes equally,
         regardless of size (unlike pixel-level IoU which is dominated by
         large defects).
      3. Sweep t, compute mean overlap = TPR_region and FPR_pixel.
      4. Integrate TPR_region over FPR_pixel ∈ [0, fpr_limit], normalised by fpr_limit.

    Reference: Bergmann et al. (MVTec AD paper, CVPR 2019).

    Args:
        fpr_limit:  Upper limit on per-pixel FPR for integration (default 0.3).
                    Follows MVTec AD protocol.

    Returns:
        pro_score: Float in [0, 1].  Higher is better.
    """
    import cv2
    thresholds = np.linspace(0, 1, num_thresholds)
    fprs, tprs = [], []

    all_preds = [p.copy() for p in pred_maps]
    all_gts   = [g.copy().astype(np.uint8) for g in gt_masks]

    for thr in thresholds:
        per_region_tpr = []
        fp_pixels, total_neg = 0, 0

        for pred, gt in zip(all_preds, all_gts):
            pred_bin = (pred >= thr).astype(np.uint8)

            # Find connected components in ground-truth mask
            n_comp, labels = cv2.connectedComponents(gt)
            for comp_id in range(1, n_comp):  # skip background (0)
                region = (labels == comp_id)
                overlap = (pred_bin.astype(bool) & region).sum() / (region.sum() + 1e-8)
                per_region_tpr.append(overlap)

            # FPR: false positive pixels over total negative pixels
            neg = (gt == 0)
            fp_pixels   += (pred_bin.astype(bool) & neg).sum()
            total_neg   += neg.sum()

        fpr = fp_pixels / (total_neg + 1e-8)
        tpr = np.mean(per_region_tpr) if per_region_tpr else 0.0
        fprs.append(fpr)
        tprs.append(tpr)

    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # Sort by FPR for integration
    order = np.argsort(fprs)
    fprs, tprs = fprs[order], tprs[order]

    # Clip to [0, fpr_limit] and normalise
    mask = fprs <= fpr_limit
    if mask.sum() < 2:
        return 0.0
    pro = np.trapz(tprs[mask], fprs[mask]) / fpr_limit
    return float(np.clip(pro, 0, 1))


# ─── Classification Accuracy ──────────────────────────────────────────────────

def compute_accuracy(
    all_preds: List[int],
    all_labels: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Overall and per-class classification accuracy.

    Args:
        all_preds:   List of predicted class indices.
        all_labels:  List of ground-truth class indices.
        class_names: Optional list of class name strings for pretty output.

    Returns:
        Dict with 'overall' accuracy and per-class accuracies.
    """
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    overall    = float((all_preds == all_labels).mean() * 100)

    result = {"overall": round(overall, 2)}
    for cls_id in np.unique(all_labels):
        mask = all_labels == cls_id
        cls_acc = float((all_preds[mask] == all_labels[mask]).mean() * 100)
        key = class_names[cls_id] if class_names else str(cls_id)
        result[key] = round(cls_acc, 2)
    return result


# ─── Unified Evaluation Summary ───────────────────────────────────────────────

def summarise_localisation(
    pred_maps: List[np.ndarray],
    gt_masks: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute all localisation metrics in one call and return a summary dict.
    """
    auroc  = compute_pixel_auroc(pred_maps, gt_masks)
    ap     = compute_pixel_ap(pred_maps, gt_masks)
    f1_max, best_thr = compute_f1_max(pred_maps, gt_masks)
    pro    = compute_pro(pred_maps, gt_masks)

    return {
        "AUROC":      round(auroc, 4),
        "AP":         round(ap, 4),
        "F1-max":     round(f1_max, 4),
        "best_threshold": round(best_thr, 4),
        "PRO":        round(pro, 4),
    }


def summarise_generation(
    real_images: List[Image.Image],
    fake_images: List[Image.Image],
    fake_tensors: Optional[List[torch.Tensor]] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Compute KID and IC-LPIPS for generation quality evaluation.

    Args:
        real_images:   List of real defect PIL images (test set).
        fake_images:   List of generated PIL images (same class).
        fake_tensors:  Optional pre-computed tensors for IC-LPIPS (faster).
        device:        Compute device.

    Returns:
        Dict with 'KID_mean', 'KID_std', 'IC_LPIPS'.
    """
    extractor = InceptionFeatureExtractor(device=device)

    print("    Extracting features from real images...")
    real_feats = extractor.extract(real_images)

    print("    Extracting features from generated images...")
    fake_feats = extractor.extract(fake_images)

    print("    Computing KID...")
    kid_mean, kid_std = compute_kid(real_feats, fake_feats)

    print("    Computing IC-LPIPS...")
    if fake_tensors is None:
        import torchvision.transforms as T
        to_t = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                          T.Normalize([0.5]*3, [0.5]*3)])
        fake_tensors = [to_t(img.convert("RGB")).unsqueeze(0) for img in fake_images]

    ic_lpips = compute_ic_lpips(fake_tensors, device=device, num_pairs=200)

    return {
        "KID_mean":  round(kid_mean, 4),
        "KID_std":   round(kid_std, 4),
        "IC_LPIPS":  round(ic_lpips, 4),
    }
