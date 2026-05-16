"""
utils/
──────
Shared utilities: metrics computation and visualisation.

  metrics.py       — KID, IC-LPIPS, AUROC, AP, F1-max, PRO, Accuracy
  visualization.py — All matplotlib plots for paper-style figures
"""

from .metrics import (
    InceptionFeatureExtractor,
    compute_kid,
    compute_ic_lpips,
    compute_pixel_auroc,
    compute_pixel_ap,
    compute_f1_max,
    compute_pro,
    compute_accuracy,
    summarise_localisation,
    summarise_generation,
)

from .visualization import (
    plot_generated_grid,
    plot_comparison_grid,
    plot_attention_map,
    plot_lfs_candidates,
    plot_training_curves,
    plot_classification_bar,
    plot_localisation_masks,
    plot_class_distribution,
    plot_metrics_table,
)

__all__ = [
    # metrics
    "InceptionFeatureExtractor",
    "compute_kid", "compute_ic_lpips",
    "compute_pixel_auroc", "compute_pixel_ap", "compute_f1_max", "compute_pro",
    "compute_accuracy", "summarise_localisation", "summarise_generation",
    # visualization
    "plot_generated_grid", "plot_comparison_grid", "plot_attention_map",
    "plot_lfs_candidates", "plot_training_curves", "plot_classification_bar",
    "plot_localisation_masks", "plot_class_distribution", "plot_metrics_table",
]
