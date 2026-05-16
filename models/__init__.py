"""
models/
───────
DefectFill model components.

  defectfill.py  — SD2-inpainting + LoRA + learned [V*] token + attention hook
  lfs.py         — Low-Fidelity Selection for post-generation quality filtering
"""

from .defectfill import DefectFillModel, CrossAttentionHook, VStarAttentionProcessor
from .lfs import LowFidelitySelector, compute_lfs_scores_for_set

__all__ = [
    "DefectFillModel",
    "CrossAttentionHook",
    "VStarAttentionProcessor",
    "LowFidelitySelector",
    "compute_lfs_scores_for_set",
]
