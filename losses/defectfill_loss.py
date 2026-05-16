"""
losses/defectfill_loss.py
─────────────────────────
The three custom loss functions that make DefectFill work.

Why three losses?
─────────────────
Standard diffusion fine-tuning (e.g., DreamBooth, Textual Inversion) learns
a concept by minimising a single denoising loss across the whole image.  This
works well for large, dominant objects.

Defects are different:
  • They are LOCAL — small regions that depend on the surrounding object.
  • They are ANOMALOUS — features the base model has never seen (cracks, blow-
    holes in steel welds do not appear in LAION-5B training data).
  • They must BLEND — a defect rendered in the wrong colour or texture looks
    fake instantly.

The three losses address each of these challenges:

  L_def  (Defect Loss)
    → Forces the model to learn the INTRINSIC appearance of the defect itself,
      ignoring all background context.  Uses defect prompt "An X-ray showing [V*]".
      Loss is masked to only back-propagate through the defect region M.

  L_obj  (Object Loss)
    → Forces the model to understand the SEMANTIC RELATIONSHIP between the
      defect and its host object.  Uses object prompt "A steel weld with [V*]"
      and a random box mask Mrand to inpaint arbitrary regions — so the model
      sees the whole image context.  Loss weight on defect area = 1.0,
      on background = α (small), so defect details still dominate.

  L_attn (Attention Loss)
    → Forces the [V*] attention map to ALIGN with the defect mask M.
      Without this, the word token can diffuse attention over the whole image,
      causing the model to generate defects in the wrong location or shape.
      Uses decoder-layer cross-attention maps only (encoder maps are noisy).

Final loss:
    L = λ_def · L_def + λ_obj · L_obj + λ_attn · L_attn
    Weights: (0.5, 0.2, 0.05)  — paper Table in Sec. 3.2

Reference: Song et al. "DefectFill" CVPR 2025, Sec. 3.2
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Defect Loss ──────────────────────────────────────────────────────────────

class DefectLoss(nn.Module):
    """
    L_def — Masked denoising loss focused exclusively on the defect region.

    Equation (5) from the paper:
        L_def = E[ || M ⊙ (ε - ε_θ(x_t^def, t, c^def)) ||₂² ]

    Where:
        M         = binary defect mask (1 inside defect region)
        ε         = ground-truth Gaussian noise added to x0
        ε_θ(...)  = model's noise prediction given defect-prompt embedding c^def
        x_t^def   = concat(x_t, b_def, M) — inpainting model input
                    b_def = background with defect masked out

    The ⊙ masking ensures gradient only flows through defect pixels.
    Background pixels contribute ZERO gradient, so the model is not
    confused by what the background "should" look like.
    """

    def forward(
        self,
        noise_pred: torch.Tensor,      # (B, C, H, W) — model noise prediction
        noise_target: torch.Tensor,    # (B, C, H, W) — ground-truth noise ε
        mask: torch.Tensor,            # (B, 1, H, W) — binary mask [0,1]
    ) -> torch.Tensor:
        """
        Args:
            noise_pred:   Output of UNet ε_θ(x_t^def, t, c^def).
            noise_target: The noise ε that was added to x0 (regression target).
            mask:         Defect binary mask M, float, [0,1], shape (B,1,H,W).
                          Broadcast over channel dim automatically.

        Returns:
            Scalar loss tensor.
        """
        # Expand mask from (B,1,H,W) to (B,C,H,W) to match noise tensors
        mask_expanded = mask.expand_as(noise_pred)

        # Masked L2 between predicted and ground-truth noise
        # Only defect pixels (mask=1) contribute
        diff = (noise_target - noise_pred) * mask_expanded
        loss = (diff ** 2).sum() / (mask_expanded.sum() + 1e-8)
        return loss


# ─── Object Loss ──────────────────────────────────────────────────────────────

class ObjectLoss(nn.Module):
    """
    L_obj — Contextual denoising loss with object-awareness.

    Equation (7) from the paper:
        L_obj = E[ || M' ⊙ (ε - ε_θ(x_t^obj, t, c^obj)) ||₂² ]
        M' = M + α · (1 - M)

    Where:
        c^obj  = embedding of "A steel weld with [V*]" (object-context prompt)
        M_rand = random box mask (30 boxes) covering arbitrary image regions
        x_t^obj = concat(x_t, b_rand, M_rand) — random inpainting input
        M'      = adjusted weight mask: defect pixels weight=1, bg pixels weight=α

    The random boxes teach the model how the ENTIRE weld image looks, so it
    understands what normal weld texture looks like and can blend the defect
    seamlessly.  The adjusted mask M' ensures the defect area still gets
    priority (weight=1) even though we are filling arbitrary random regions.

    α (alpha_bg) is a small value like 0.1 — background still contributes
    gradient but much less than the defect region.
    """

    def __init__(self, alpha_bg: float = 0.1):
        """
        Args:
            alpha_bg: Weight for background (non-defect) pixels in M'.
                      Paper uses a value < 1 to prioritise defect region.
        """
        super().__init__()
        self.alpha_bg = alpha_bg

    def forward(
        self,
        noise_pred: torch.Tensor,      # (B, C, H, W)
        noise_target: torch.Tensor,    # (B, C, H, W)
        defect_mask: torch.Tensor,     # (B, 1, H, W) original defect mask M
    ) -> torch.Tensor:
        """
        Args:
            noise_pred:    Output of UNet using object prompt + random-box mask.
            noise_target:  Ground-truth noise ε.
            defect_mask:   Original defect binary mask M (not the random mask).
                           Used to compute M' = M + α(1-M).
        """
        # Adjusted mask M' = defect area fully weighted, bg area α-weighted
        M_prime = defect_mask + self.alpha_bg * (1.0 - defect_mask)
        M_prime_expanded = M_prime.expand_as(noise_pred)

        diff = (noise_target - noise_pred) * M_prime_expanded
        loss = (diff ** 2).sum() / (M_prime_expanded.sum() + 1e-8)
        return loss


# ─── Attention Loss ───────────────────────────────────────────────────────────

class AttentionLoss(nn.Module):
    """
    L_attn — Forces [V*] cross-attention maps to align with the defect mask.

    Equation (8) from the paper:
        L_attn = E[ || A_t^[V*] - M ||₂² ]

    Where:
        A_t^[V*] = averaged decoder cross-attention maps for the [V*] token,
                   resized to match the latent spatial resolution.
        M        = binary defect mask (0 or 1, latent-space resolution).

    Why only decoder layers?
        The paper cites MasaCtrl (Cao et al. ICCV 2023) showing that UNet
        ENCODER attention maps do not accurately represent spatial layout of
        tokens.  Only DECODER attention maps align with semantic regions.
        Using encoder maps would provide noisy, misleading supervision.

    How the attention maps are extracted:
        We register forward hooks on the decoder cross-attention layers during
        the forward pass of the object-prompt pipeline.  The hook caches
        attn_weights for the specific [V*] token position in the text sequence.

    Normalisation:
        Raw attention maps are averaged over heads and decoder layers,
        then resized to match M's spatial dimensions.  Values are naturally
        in [0, 1] after softmax inside the attention blocks.
    """

    def forward(
        self,
        attn_map: torch.Tensor,    # (B, H, W) — averaged decoder attn for [V*]
        mask: torch.Tensor,        # (B, 1, H, W) — binary defect mask M
    ) -> torch.Tensor:
        """
        Args:
            attn_map:  Cross-attention activations for [V*] token, decoder only.
                       Shape: (B, H, W) or (B, 1, H, W), values in [0, 1].
            mask:      Binary defect mask M, float [0, 1], shape (B, 1, H, W).

        Returns:
            Scalar L2 loss that pushes attn_map → mask.
        """
        # Ensure attn_map is (B, 1, H, W)
        if attn_map.dim() == 3:
            attn_map = attn_map.unsqueeze(1)   # (B, 1, H, W)

        # Resize attn_map to match mask spatial dimensions if needed
        if attn_map.shape[-2:] != mask.shape[-2:]:
            attn_map = F.interpolate(
                attn_map,
                size=mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # L2 between attention map and defect mask
        loss = F.mse_loss(attn_map, mask)
        return loss


# ─── Combined DefectFill Loss ─────────────────────────────────────────────────

class DefectFillLoss(nn.Module):
    """
    Combined loss: L = λ_def·L_def + λ_obj·L_obj + λ_attn·L_attn

    This class orchestrates all three loss components and returns both the
    combined scalar and the individual components (useful for logging).

    Default weights from paper:
        λ_def  = 0.5  (defect loss dominates — most important signal)
        λ_obj  = 0.2  (object context is secondary)
        λ_attn = 0.05 (attention regularisation, smallest weight)
    """

    def __init__(
        self,
        lambda_def: float = 0.5,
        lambda_obj: float = 0.2,
        lambda_attn: float = 0.05,
        alpha_bg: float = 0.1,
    ):
        super().__init__()
        self.lambda_def = lambda_def
        self.lambda_obj = lambda_obj
        self.lambda_attn = lambda_attn

        self.def_loss = DefectLoss()
        self.obj_loss = ObjectLoss(alpha_bg=alpha_bg)
        self.attn_loss = AttentionLoss()

    def forward(
        self,
        # Defect-prompt forward pass outputs
        noise_pred_def: torch.Tensor,
        noise_target: torch.Tensor,
        mask: torch.Tensor,
        # Object-prompt forward pass outputs
        noise_pred_obj: torch.Tensor,
        # Attention map for [V*] from decoder layers
        attn_map_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            noise_pred_def:  Noise predicted using defect prompt (L_def).
            noise_target:    Ground-truth noise ε (shared between both prompts
                             since they share the same (t, ε) sample).
            mask:            Defect binary mask M, shape (B,1,H,W).
            noise_pred_obj:  Noise predicted using object prompt (L_obj).
            attn_map_v:      Decoder attention map for [V*] token (L_attn).
                             If None, L_attn is skipped.

        Returns:
            total_loss:  Weighted scalar loss for backprop.
            components:  Dict with individual loss values for logging.
        """
        l_def = self.def_loss(noise_pred_def, noise_target, mask)
        l_obj = self.obj_loss(noise_pred_obj, noise_target, mask)

        total = self.lambda_def * l_def + self.lambda_obj * l_obj

        l_attn_val = torch.tensor(0.0, device=mask.device)
        if attn_map_v is not None:
            l_attn_val = self.attn_loss(attn_map_v, mask)
            total = total + self.lambda_attn * l_attn_val

        components = {
            "loss_def":  l_def.item(),
            "loss_obj":  l_obj.item(),
            "loss_attn": l_attn_val.item(),
            "loss_total": total.item(),
        }
        return total, components


# ─── Random Box Mask Generator ────────────────────────────────────────────────

def make_random_box_mask(
    batch_size: int,
    img_size: int,
    num_boxes: int = 30,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate random rectangular box masks for the object loss.

    The paper uses 30 random boxes covering arbitrary image regions.
    This forces the object-prompt forward pass to reconstruct diverse
    local regions, teaching the model the full semantic context of
    the weld image (what normal texture/structure looks like everywhere).

    Args:
        batch_size:  Number of masks in the batch.
        img_size:    Spatial dimension (assumes square: img_size × img_size).
        num_boxes:   Number of random boxes per mask (paper: 30).
        device:      Target device.

    Returns:
        Mask tensor of shape (B, 1, img_size, img_size), float [0, 1].
        1 = masked region (to be inpainted), 0 = keep region.
    """
    masks = torch.zeros(batch_size, 1, img_size, img_size, device=device)
    for b in range(batch_size):
        for _ in range(num_boxes):
            # Random box: x1,y1 in [0, img_size-1], w,h in [5, img_size//4]
            x1 = torch.randint(0, img_size, (1,)).item()
            y1 = torch.randint(0, img_size, (1,)).item()
            w = torch.randint(5, max(6, img_size // 4), (1,)).item()
            h = torch.randint(5, max(6, img_size // 4), (1,)).item()
            x2 = min(img_size, x1 + w)
            y2 = min(img_size, y1 + h)
            masks[b, 0, y1:y2, x1:x2] = 1.0
    return masks
