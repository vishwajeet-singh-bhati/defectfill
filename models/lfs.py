"""
models/lfs.py
─────────────
Low-Fidelity Selection (LFS) — the quality filtering post-process from
Section 3.3 of the DefectFill paper.

Problem it solves
─────────────────
Diffusion models are stochastic — each generation from the same (image, mask)
pair produces a different result.  Sometimes the inpainting model:
  • Over-reconstructs the masked area (fills it with normal weld texture),
    resulting in NO visible defect — useless for training.
  • Generates a very mild defect that's barely distinguishable from normal.
  • Generates a perfect, pronounced defect — this is what we want!

LFS identifies the "best" sample from N candidates automatically, without
human effort.

How it works
────────────
Given N generated candidates from the same (image, mask):
  1. Compute a reconstruction metric between each candidate and the ORIGINAL
     image, measured ONLY within the masked region.
  2. Pick the candidate with the HIGHEST LPIPS score (= most perceptually
     different from the original = most defect-like appearance).

Why LPIPS (not SSIM or PSNR)?
  LPIPS (Learned Perceptual Image Patch Similarity) measures perceptual
  distance using deep features, not pixel-level statistics.  A subtle
  texture change (like a crack) may have high pixel similarity (SSIM≈1)
  but be very perceptually different (LPIPS↑).  LPIPS is more sensitive
  to the kind of localised texture changes that characterise weld defects.

  From the paper (Figure 3): LFS selects the sample with the highest LPIPS
  score within the masked region, ensuring the defect is most pronounced.

Paper citation: "we select the least reconstructed image from the eight
samples generated... based on a reconstruction metric (e.g. PSNR, SSIM,
LPIPS) measured only within the masked region."
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class LowFidelitySelector:
    """
    Selects the highest-quality defect image from a set of candidates.

    Usage:
        selector = LowFidelitySelector(metric="lpips", device="cuda")
        best_idx = selector.select(candidates, original_image, mask)
        best_image = candidates[best_idx]
    """

    SUPPORTED_METRICS = ["lpips", "ssim", "psnr"]

    def __init__(self, metric: str = "lpips", device: str = "cpu"):
        """
        Args:
            metric: Which reconstruction metric to use.
                    "lpips" → pick HIGHEST score (most different from original).
                    "ssim"  → pick LOWEST score (least similar to original).
                    "psnr"  → pick LOWEST score (lowest signal fidelity).
            device: "cpu" or "cuda".
        """
        assert metric in self.SUPPORTED_METRICS, \
            f"metric must be one of {self.SUPPORTED_METRICS}"
        self.metric = metric
        self.device = device
        self._lpips_fn = None   # Lazy-loaded to avoid import cost

    def _get_lpips(self):
        """Lazy-load LPIPS model (AlexNet backbone, faster than VGG)."""
        if self._lpips_fn is None:
            import lpips
            self._lpips_fn = lpips.LPIPS(net="alex").to(self.device)
            self._lpips_fn.eval()
        return self._lpips_fn

    def compute_score(
        self,
        generated: torch.Tensor,   # (1, 3, H, W), range [-1, 1]
        original: torch.Tensor,    # (1, 3, H, W), range [-1, 1]
        mask: torch.Tensor,        # (1, 1, H, W), range [0, 1]
    ) -> float:
        """
        Compute reconstruction score ONLY within the masked region.

        Args:
            generated:  One generated candidate image.
            original:   The original defect-free background image.
            mask:       Binary mask (1 = defect region).

        Returns:
            Scalar score.  Higher = more defect-like (for lpips).
        """
        if self.metric == "lpips":
            return self._lpips_score(generated, original, mask)
        elif self.metric == "ssim":
            return self._ssim_score(generated, original, mask)
        elif self.metric == "psnr":
            return self._psnr_score(generated, original, mask)

    def _lpips_score(self, gen, orig, mask):
        """
        LPIPS between generated and original, masked to defect region.
        Higher LPIPS = more perceptually different = better defect.
        """
        lpips_fn = self._get_lpips()
        with torch.no_grad():
            # Mask both images to isolate the defect region
            mask_b = mask.expand_as(gen)
            gen_masked = gen * mask_b
            orig_masked = orig * mask_b
            score = lpips_fn(gen_masked, orig_masked).item()
        return score   # Higher is better (we want most different)

    def _ssim_score(self, gen, orig, mask):
        """SSIM within masked region. Lower SSIM = more different = better."""
        from skimage.metrics import structural_similarity as ssim
        gen_np = gen.squeeze(0).permute(1, 2, 0).cpu().numpy()
        orig_np = orig.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy().astype(bool)

        # Crop to bounding box of mask for meaningful SSIM
        rows, cols = np.where(mask_np)
        if len(rows) == 0:
            return 0.0
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        g_crop = gen_np[r0:r1, c0:c1]
        o_crop = orig_np[r0:r1, c0:c1]

        score = ssim(g_crop, o_crop, channel_axis=-1, data_range=2.0)
        return -score   # Negate so "higher is better" logic applies

    def _psnr_score(self, gen, orig, mask):
        """PSNR within masked region. Lower PSNR = more different = better."""
        mask_b = mask.expand_as(gen)
        gen_masked = gen * mask_b
        orig_masked = orig * mask_b
        mse = F.mse_loss(gen_masked, orig_masked).item()
        if mse == 0:
            return 0.0
        psnr = 10 * np.log10(4.0 / mse)   # data range is 2 → max signal = 4
        return -psnr   # Negate so "higher is better" logic applies

    def select(
        self,
        candidates: List[torch.Tensor],    # List of (1, 3, H, W) tensors
        original: torch.Tensor,             # (1, 3, H, W) original image
        mask: torch.Tensor,                 # (1, 1, H, W) defect mask
    ) -> Tuple[int, float]:
        """
        Select the best candidate from N generated images.

        Args:
            candidates:  List of N generated image tensors, range [-1, 1].
            original:    Original (defect-free or defective-background) image.
            mask:        Defect region mask.

        Returns:
            (best_idx, best_score): Index of best candidate and its score.
        """
        if not candidates:
            raise ValueError("Empty candidate list")
        if len(candidates) == 1:
            return 0, 0.0

        scores = []
        for cand in candidates:
            score = self.compute_score(
                cand.to(self.device),
                original.to(self.device),
                mask.to(self.device),
            )
            scores.append(score)

        # Always pick HIGHEST score (LPIPS: naturally higher = better;
        # SSIM/PSNR: negated above so highest = most different)
        best_idx = int(np.argmax(scores))
        return best_idx, scores[best_idx]

    def select_batch(
        self,
        candidates_list: List[List[torch.Tensor]],
        originals: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Apply LFS to multiple (original, mask) groups simultaneously.
        Useful for batch generation where each image gets N candidates.

        Args:
            candidates_list: List of N-candidate groups. candidates_list[i]
                             is the list of candidates for originals[i].
            originals:       Corresponding original images.
            masks:           Corresponding masks.

        Returns:
            List of best-selected tensors (one per group).
        """
        selected = []
        for candidates, orig, mask in zip(candidates_list, originals, masks):
            best_idx, score = self.select(candidates, orig, mask)
            selected.append(candidates[best_idx])
            print(f"    LFS selected candidate {best_idx} (score={score:.4f})")
        return selected


# ─── Score computation for evaluation / logging ────────────────────────────────

def compute_lfs_scores_for_set(
    candidates: List[torch.Tensor],
    original: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> List[float]:
    """
    Compute LFS scores for all candidates (for logging/analysis).

    Returns list of LPIPS scores, one per candidate.
    """
    selector = LowFidelitySelector(metric="lpips", device=device)
    return [
        selector.compute_score(c.to(device), original.to(device), mask.to(device))
        for c in candidates
    ]
