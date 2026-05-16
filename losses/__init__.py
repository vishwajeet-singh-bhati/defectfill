"""
losses/
───────
Custom loss functions implementing the DefectFill training objective.

  defectfill_loss.py — L_def + L_obj + L_attn → L_ours (weighted combination)
"""

from .defectfill_loss import (
    DefectFillLoss,
    DefectLoss,
    ObjectLoss,
    AttentionLoss,
    make_random_box_mask,
)

__all__ = [
    "DefectFillLoss",
    "DefectLoss",
    "ObjectLoss",
    "AttentionLoss",
    "make_random_box_mask",
]
