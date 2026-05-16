"""
utils/visualization.py
──────────────────────
All visualisation utilities for DefectFill on the Steel Weld Defect dataset.

Functions:
  1. plot_generated_grid       — Show generated defects in a grid (like paper Fig.4)
  2. plot_comparison_grid      — Side-by-side: input | mask | generated (like Fig.5)
  3. plot_attention_map        — Overlay cross-attention A^[V*] on image (debug/paper fig)
  4. plot_lfs_candidates       — Show N candidates with LPIPS scores, highlight selected
  5. plot_training_curves      — Loss components over training steps
  6. plot_classification_bar   — Per-class accuracy bar chart
  7. plot_localisation_masks   — Show pred vs. gt masks on real test images
  8. plot_metrics_table        — Render an HTML/PNG table of all evaluation numbers
  9. plot_class_distribution   — Visualise severe class imbalance in the raw dataset

All functions save to a specified output directory and return the file path.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


# ─── Utility Helpers ─────────────────────────────────────────────────────────

def _tensor_to_numpy(t) -> np.ndarray:
    """Convert torch tensor (C,H,W) or (1,C,H,W) in [-1,1] to uint8 numpy (H,W,3)."""
    import torch
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        arr = ((t.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return arr
    return t


def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Ensure image is (H,W,3) uint8 RGB."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] == 1:
        return np.concatenate([img, img, img], axis=-1)
    return img


def _mask_overlay(image: np.ndarray, mask: np.ndarray,
                  color=(255, 80, 80), alpha=0.4) -> np.ndarray:
    """Overlay a binary mask on an RGB image with a semi-transparent colour."""
    overlay = image.copy()
    mask_bool = mask > 127
    overlay[mask_bool] = (
        (1 - alpha) * image[mask_bool].astype(float) +
        alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


# ─── 1. Generated Defects Grid ────────────────────────────────────────────────

def plot_generated_grid(
    images: List[np.ndarray],           # List of generated images (H,W,3)
    masks: List[np.ndarray],            # Corresponding masks (H,W)
    class_name: str,
    save_path: str,
    n_cols: int = 5,
    show_mask_overlay: bool = True,
) -> str:
    """
    Plot a grid of generated defect images — replicates paper Figure 4.

    Rows: normal image | generated defect (with mask overlay) | zoomed defect
    Columns: different generated samples

    Args:
        images:   List of generated RGB images (uint8, numpy).
        masks:    Corresponding binary masks.
        save_path: File path to save the figure.
        n_cols:   Number of columns (samples) to display.
        show_mask_overlay: If True, overlay mask on generated images (red).

    Returns:
        save_path
    """
    n = min(len(images), n_cols)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(f"Generated Defects — [{class_name}]", fontsize=14, fontweight="bold")

    for i in range(n):
        img = _ensure_rgb(images[i])
        msk = masks[i]

        # Row 0: generated defect image
        ax0 = axes[0, i] if n > 1 else axes[0]
        if show_mask_overlay:
            ax0.imshow(_mask_overlay(img, msk))
        else:
            ax0.imshow(img)
        ax0.set_title(f"Sample {i+1}", fontsize=8)
        ax0.axis("off")

        # Row 1: zoomed in on defect bbox
        ax1 = axes[1, i] if n > 1 else axes[1]
        ys, xs = np.where(msk > 127)
        if len(ys) > 0:
            pad = 10
            y1, y2 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
            x1, x2 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
            ax1.imshow(img[y1:y2, x1:x2])
            ax1.set_title("Zoomed", fontsize=8)
        else:
            ax1.imshow(img)
            ax1.set_title("(no mask)", fontsize=8)
        ax1.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 2. Comparison Grid (Ours vs Baselines) ───────────────────────────────────

def plot_comparison_grid(
    method_results: Dict[str, List[np.ndarray]],   # {"DFMGAN": [...], "AnoDiff": [...], "Ours": [...]}
    class_names: List[str],
    save_path: str,
) -> str:
    """
    Replicate paper Figure 5 — side-by-side comparison of different methods.

    Rows: one method per row (DFMGAN, AnomalyDiffusion, Ours)
    Cols: one defect class per column

    Args:
        method_results: Dict mapping method name → list of images (one per class).
        class_names:    List of class names for column headers.
    """
    methods = list(method_results.keys())
    n_methods = len(methods)
    n_classes = len(class_names)

    fig, axes = plt.subplots(
        n_methods, n_classes,
        figsize=(2.5 * n_classes, 2.5 * n_methods),
    )
    if n_methods == 1:
        axes = axes[np.newaxis, :]
    if n_classes == 1:
        axes = axes[:, np.newaxis]

    for j, cls_name in enumerate(class_names):
        axes[0, j].set_title(cls_name, fontsize=8, fontweight="bold")

    for i, method in enumerate(methods):
        axes[i, 0].set_ylabel(method, fontsize=9, fontweight="bold", rotation=0,
                              ha="right", labelpad=60)
        imgs = method_results[method]
        for j in range(n_classes):
            ax = axes[i, j]
            if j < len(imgs) and imgs[j] is not None:
                ax.imshow(_ensure_rgb(imgs[j]))
            else:
                ax.set_facecolor("#222222")
            ax.axis("off")

    plt.suptitle("Defect Generation Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 3. Attention Map Overlay ─────────────────────────────────────────────────

def plot_attention_map(
    image: np.ndarray,                # (H, W, 3) RGB uint8
    attn_map: np.ndarray,             # (H, W) float in [0, 1]
    mask: np.ndarray,                 # (H, W) binary uint8
    save_path: str,
    class_name: str = "",
    step: int = 0,
) -> str:
    """
    Visualise the [V*] cross-attention map alongside the defect mask.

    Used to debug whether the attention loss (L_attn) is successfully
    aligning the model's attention with the defect region.

    Shows: original image | attention heatmap | mask | overlay
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"[V*] Attention Map — [{class_name}] step {step}",
        fontsize=12, fontweight="bold"
    )

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    heatmap = axes[1].imshow(attn_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("[V*] Attention")
    axes[1].axis("off")
    plt.colorbar(heatmap, ax=axes[1], fraction=0.046)

    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title("Defect Mask")
    axes[2].axis("off")

    # Overlay: attention on image
    overlay = image.copy().astype(float)
    hm_rgb = plt.cm.jet(attn_map)[:, :, :3] * 255
    alpha = 0.5
    overlay = (overlay * (1 - alpha) + hm_rgb * alpha).clip(0, 255).astype(np.uint8)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ─── 4. LFS Candidate Selection ───────────────────────────────────────────────

def plot_lfs_candidates(
    candidates: List[np.ndarray],     # N generated candidates (H,W,3)
    lpips_scores: List[float],        # LPIPS score per candidate
    selected_idx: int,                # Index of LFS-selected candidate
    save_path: str,
    class_name: str = "",
) -> str:
    """
    Visualise all N LFS candidates with their LPIPS scores.
    Replicates paper Figure 3 — shows why LFS selects a specific candidate.

    The selected candidate (highest LPIPS) is highlighted with a blue border.
    """
    n = len(candidates)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    fig.suptitle(
        f"Low-Fidelity Selection — [{class_name}]\n"
        f"Blue box = selected (highest LPIPS = most defect-like)",
        fontsize=10
    )

    if n == 1:
        axes = [axes]

    for i, (img, score) in enumerate(zip(candidates, lpips_scores)):
        ax = axes[i]
        ax.imshow(_ensure_rgb(img))
        label = f"LPIPS: {score:.4f}"
        title_color = "blue" if i == selected_idx else "black"
        ax.set_title(label, fontsize=8, color=title_color, fontweight="bold" if i == selected_idx else "normal")
        ax.axis("off")
        # Add coloured border for selected
        if i == selected_idx:
            for spine in ax.spines.values():
                spine.set_edgecolor("blue")
                spine.set_linewidth(3)
                spine.set_visible(True)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 5. Training Loss Curves ──────────────────────────────────────────────────

def plot_training_curves(
    log: Dict[str, List],    # {"steps": [...], "loss_total": [...], ...}
    save_path: str,
    class_name: str = "",
) -> str:
    """
    Plot the three DefectFill loss components over training steps.

    Shows L_def, L_obj, L_attn, and their weighted sum L_total.
    Essential for verifying that each loss component is decreasing properly.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"DefectFill Training Loss — [{class_name}]", fontsize=13, fontweight="bold")

    steps = log["steps"]
    loss_map = {
        "loss_total": ("Total Loss", "black", axes[0, 0]),
        "loss_def":   ("L_def (Defect Loss, λ=0.5)", "crimson", axes[0, 1]),
        "loss_obj":   ("L_obj (Object Loss, λ=0.2)", "steelblue", axes[1, 0]),
        "loss_attn":  ("L_attn (Attention Loss, λ=0.05)", "forestgreen", axes[1, 1]),
    }

    for key, (title, color, ax) in loss_map.items():
        if key in log and len(log[key]) > 0:
            ax.plot(steps, log[key], color=color, linewidth=1.5, alpha=0.8)
            # Smoothed line
            if len(log[key]) > 5:
                kernel = np.ones(5) / 5
                smoothed = np.convolve(log[key], kernel, mode="same")
                ax.plot(steps, smoothed, color=color, linewidth=2.5,
                        linestyle="--", label="smoothed")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 6. Classification Accuracy Bar Chart ─────────────────────────────────────

def plot_classification_bar(
    results_by_method: Dict[str, Dict[str, float]],   # {"Ours": {"crack": 87.5, ...}, ...}
    class_names: List[str],
    save_path: str,
) -> str:
    """
    Per-class accuracy bar chart comparing multiple methods.
    Replicates the data in paper Table 3 as a visual.
    """
    n_classes = len(class_names)
    n_methods = len(results_by_method)
    methods = list(results_by_method.keys())
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"][:n_methods]

    x = np.arange(n_classes)
    bar_w = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(10, n_classes * 1.5), 5))

    for i, (method, color) in enumerate(zip(methods, colors)):
        accs = [results_by_method[method].get(c, 0) for c in class_names]
        bars = ax.bar(x + i * bar_w - (n_methods - 1) * bar_w / 2, accs,
                      width=bar_w, label=method, color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xlabel("Defect Class", fontsize=11)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax.set_title("Per-Class Classification Accuracy on Steel Weld Defects", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 7. Localisation Prediction Overlay ───────────────────────────────────────

def plot_localisation_masks(
    images: List[np.ndarray],       # (H, W, 3) RGB images
    pred_maps: List[np.ndarray],    # (H, W) float probability maps
    gt_masks: List[np.ndarray],     # (H, W) binary ground-truth
    save_path: str,
    n_show: int = 6,
    threshold: float = 0.5,
    class_name: str = "",
) -> str:
    """
    Visualise UNet localisation predictions vs. ground truth.

    Columns: input | gt mask | predicted prob map | thresholded pred | overlay
    """
    n = min(len(images), n_show)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "GT Mask", "Pred Prob Map", "Pred (thresh)", "Overlay"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    for i in range(n):
        img  = _ensure_rgb(images[i])
        gt   = gt_masks[i]
        pred = pred_maps[i]
        pred_bin = (pred >= threshold).astype(np.uint8) * 255

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 2].imshow(pred, cmap="jet", vmin=0, vmax=1)
        axes[i, 3].imshow(pred_bin, cmap="gray")
        axes[i, 4].imshow(_mask_overlay(img, pred_bin, color=(255, 80, 80)))

        for j in range(5):
            axes[i, j].axis("off")

    fig.suptitle(f"Localisation Results — [{class_name}]  (thresh={threshold})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 8. Dataset Class Distribution ────────────────────────────────────────────

def plot_class_distribution(
    class_names: List[str],
    counts: List[int],
    save_path: str,
) -> str:
    """
    Visualise the severe class imbalance in the raw Steel Weld dataset.

    air-hole (5191) vs bite-edge (35) — 148× imbalance!
    This motivates why DefectFill is essential for this dataset.
    """
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(class_names)))
    sorted_idx = np.argsort(counts)[::-1]
    sorted_names  = [class_names[i] for i in sorted_idx]
    sorted_counts = [counts[i] for i in sorted_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart (log scale to show imbalance clearly)
    bars = axes[0].bar(range(len(sorted_names)), sorted_counts,
                       color=colors, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, sorted_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 30, str(count),
                     ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(range(len(sorted_names)))
    axes[0].set_xticklabels(sorted_names, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("Number of Annotations")
    axes[0].set_title("Class Distribution (log scale)", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].axhline(100, color="red", linestyle="--", linewidth=1, label="100 samples")
    axes[0].legend(fontsize=8)

    # Pie chart
    axes[1].pie(sorted_counts, labels=sorted_names, autopct="%1.1f%%",
                colors=colors, startangle=140, textprops={"fontsize": 8})
    axes[1].set_title("Class Share (%)", fontsize=11)

    plt.suptitle(
        "Steel Weld Defect Dataset — Class Imbalance\n"
        f"Total annotations: {sum(counts):,}  |  "
        f"Imbalance ratio: {max(counts)/min(counts):.0f}×",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path


# ─── 9. Metrics Summary Table ─────────────────────────────────────────────────

def plot_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],   # {class_name: {metric: value}}
    title: str,
    save_path: str,
    highlight_best_col: bool = True,
) -> str:
    """
    Render an evaluation metrics table as a matplotlib figure.

    Args:
        metrics_dict:       {row_label: {col_label: value}}.
        title:              Figure title.
        highlight_best_col: Bold the best value in each column.
    """
    rows = list(metrics_dict.keys())
    cols = list(next(iter(metrics_dict.values())).keys())
    data = [[metrics_dict[r].get(c, float("nan")) for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.5), max(3, len(rows) * 0.5 + 1)))
    ax.axis("off")

    cell_text  = [[f"{v:.4f}" if not np.isnan(v) else "—" for v in row] for row in data]
    col_labels = cols
    row_labels = rows

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(cols)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style row labels
    for i in range(len(rows)):
        table[(i + 1, -1)].set_facecolor("#ecf0f1")
        table[(i + 1, -1)].set_text_props(fontweight="bold")

    # Highlight best per column
    if highlight_best_col:
        for j, col in enumerate(cols):
            vals = [data[i][j] for i in range(len(rows)) if not np.isnan(data[i][j])]
            if not vals:
                continue
            best_val = max(vals) if "AUROC" in col or "AP" in col or "F1" in col or "PRO" in col or "IC" in col or "Acc" in col else min(vals)
            for i in range(len(rows)):
                if abs(data[i][j] - best_val) < 1e-6:
                    table[(i + 1, j)].set_facecolor("#d5f5e3")
                    table[(i + 1, j)].set_text_props(fontweight="bold", color="#1e8449")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return save_path
