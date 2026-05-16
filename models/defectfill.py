"""
models/defectfill.py
────────────────────
Core DefectFill model — wraps Stable Diffusion 2 Inpainting with LoRA
fine-tuning and attention-map extraction hooks.

Architecture Overview
─────────────────────
                         ┌──────────────────────────────┐
  text prompt ───────►   │   CLIP Text Encoder + LoRA   │  ─── c (text embed)
                         └──────────────────────────────┘
                                       │
  image + mask ──────►  VAE encoder   │
        │               └── x₀ ──►    ▼
        │             forward noise   ┌──────────────────────────────┐
        │             ε, t ──────►    │  SD2 Inpainting UNet + LoRA │  ─► ε_θ(...)
        └── concat(x_t, b, M) ──►    └──────────────────────────────┘
                                                    │
                                        Decoder cross-attention hooks
                                        extract A_t^[V*] for L_attn

Components modified vs base SD:
  1. Text encoder (CLIP) — LoRA adapters added to attention projections
  2. UNet attention layers — LoRA adapters added
  3. [V*] token — new learned embedding added to tokenizer/text-encoder vocab
  4. Cross-attention hook — captures decoder-layer attn maps for [V*] position

What is LoRA?
  Low-Rank Adaptation (Hu et al. 2021).  Instead of fine-tuning all W params,
  we inject small matrices A (m×r) and B (r×n) where r << min(m,n).
  During forward: W_eff = W_pretrained + (B @ A) * (α/r)
  Only A and B are trained.  This:
    • Reduces trainable params by ~100×
    • Prevents catastrophic forgetting of SD's general capabilities
    • Allows fast training (<10 min per defect class on a single GPU)

Why inpainting model specifically?
  The base SD2 model (text-to-image) would require generating the ENTIRE image
  from scratch to include a defect.  The inpainting model (SD2-inpainting) is
  trained to fill ONLY the masked region while preserving the background.
  This gives us:
    • Free background preservation (no need to re-generate the normal weld)
    • Natural blending at mask boundaries (the model was trained for this)
    • Ability to specify EXACT defect location via mask shape
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


# ─── Attention Hook for extracting cross-attention maps ───────────────────────

class CrossAttentionHook:
    """
    Forward hook registered on UNet decoder cross-attention layers.

    After each forward pass through a hooked layer, it caches the attention
    weights corresponding to the [V*] token position in the text sequence.

    Usage:
        hook = CrossAttentionHook(v_star_position=5)
        handles = hook.register(unet)
        _ = unet(...)      # forward pass
        attn_map = hook.get_averaged_map(latent_h, latent_w)
        hook.clear()
    """

    def __init__(self, v_star_position: int = 1):
        """
        Args:
            v_star_position: Index of [V*] token in the text sequence.
                             Position 1 is typical for "A photo of [V*]"
                             (index 0 = <BOS>, 1 = "A", ... depends on tokenizer).
                             We search for it dynamically in forward().
        """
        self.v_star_position = v_star_position
        self._maps: List[torch.Tensor] = []

    def register(self, unet: UNet2DConditionModel) -> List:
        """
        Register hooks on all decoder cross-attention blocks.

        The paper specifies DECODER layers only (not encoder layers), because
        encoder attention maps do not reliably represent spatial token layout.

        Returns list of hook handles (call handle.remove() to clean up).
        """
        handles = []
        # Iterate over UNet up-blocks (decoder portion)
        for block in unet.up_blocks:
            for layer in block.modules():
                if hasattr(layer, "processor") and hasattr(layer, "heads"):
                    # This is an attention block
                    handle = layer.register_forward_hook(self._hook_fn)
                    handles.append(handle)
        return handles

    def _hook_fn(self, module, input, output):
        """
        Hook called after each cross-attention forward pass.

        Cross-attention output in diffusers includes attention_probs when
        we temporarily swap the processor to capture them.
        We store the [V*] attention slice.
        """
        # output is the attention output tensor; we need to capture probs
        # This is handled via a custom processor pattern below
        pass

    def clear(self):
        self._maps.clear()

    def get_averaged_map(self, h: int, w: int) -> Optional[torch.Tensor]:
        """
        Average collected attention maps and resize to (h, w).

        Returns:
            Averaged attention tensor (1, 1, h, w) or None if no maps collected.
        """
        if not self._maps:
            return None
        # Stack: (N_layers, B, H_attn, W_attn)
        stacked = torch.stack(self._maps, dim=0).mean(dim=0)  # (B, H, W)
        # Resize to latent spatial dims
        stacked = stacked.unsqueeze(1)  # (B, 1, H, W)
        stacked = torch.nn.functional.interpolate(
            stacked.float(), size=(h, w), mode="bilinear", align_corners=False
        )
        return stacked


class VStarAttentionProcessor:
    """
    Custom attention processor that captures cross-attention maps for [V*].

    Replaces the default AttnProcessor2_0 on decoder layers during training.
    After forward, the attention probabilities for the [V*] token position
    are averaged over heads and stored in the hook container.

    This is the standard way to extract attention maps in diffusers.
    """

    def __init__(self, hook: CrossAttentionHook, v_star_pos: int, is_decoder: bool = True):
        self.hook = hook
        self.v_star_pos = v_star_pos
        self.is_decoder = is_decoder

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        """Standard attention forward with attention-map capture."""
        batch_size, seq_len, _ = hidden_states.shape
        is_cross = encoder_hidden_states is not None

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        value = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Compute attention scores
        scale = query.shape[-1] ** -0.5
        attn_probs = torch.bmm(query * scale, key.transpose(1, 2)).softmax(dim=-1)

        # Capture [V*] attention map from cross-attention decoder layers
        if is_cross and self.is_decoder and self.v_star_pos < attn_probs.shape[-1]:
            # attn_probs: (B*heads, H*W, text_seq_len)
            h = w = int(attn_probs.shape[1] ** 0.5)
            n_heads = attn.heads
            v_map = attn_probs[..., self.v_star_pos]   # (B*heads, H*W)
            v_map = v_map.reshape(batch_size, n_heads, h, w)
            v_map = v_map.mean(dim=1)                   # (B, H, W) — average over heads
            self.hook._maps.append(v_map.detach())

        out = torch.bmm(attn_probs, value)
        out = attn.batch_to_head_dim(out)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ─── DefectFill Model Wrapper ─────────────────────────────────────────────────

class DefectFillModel(nn.Module):
    """
    Wraps Stable Diffusion 2 Inpainting with:
      1. LoRA adapters on UNet attention + text encoder
      2. Custom [V*] learned token in tokenizer/text encoder
      3. Cross-attention hook for attention loss

    This is the per-class model — one instance is created and trained for
    each defect class (bite-edge, crack, etc.).

    Parameters learned:
      - LoRA matrices in UNet attention layers
      - LoRA matrices in text encoder attention
      - [V*] token embedding vector
    """

    def __init__(
        self,
        config: dict,
        device: torch.device,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.model_id = config["model"]["base_model_id"]
        self.placeholder = config["model"]["placeholder_token"]
        self.lora_cfg = config["model"]["lora"]
        self.img_size = config["dataset"]["img_size"]

        # Load all components
        self._load_components()
        self._add_placeholder_token()
        self._apply_lora()
        self.attn_hook = CrossAttentionHook(v_star_position=self._get_v_star_position())

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_components(self):
        """Load tokenizer, text encoder, VAE, and UNet from HuggingFace."""
        print(f"  Loading SD2-inpainting from: {self.model_id}")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        ).to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae"
        ).to(self.device)
        self.vae.requires_grad_(False)   # VAE is FROZEN — never fine-tuned

        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)

        # Noise scheduler (DDPM for training, DDIM for inference)
        from diffusers import DDPMScheduler, DDIMScheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

    def _add_placeholder_token(self):
        """
        Add the learnable [V*] token to the tokenizer and text encoder.

        This token starts with a random embedding and is optimised during
        fine-tuning to represent the defect concept.  The text encoder's
        embedding table grows by 1 row to accommodate it.
        """
        # Add token to tokenizer
        num_added = self.tokenizer.add_tokens([self.placeholder])
        if num_added == 0:
            print(f"  [V*] token '{self.placeholder}' already in tokenizer.")

        # Resize embedding layer to accommodate the new token
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))

        # Get the ID of the new token
        self.v_star_token_id = self.tokenizer.convert_tokens_to_ids(self.placeholder)
        print(f"  [V*] token ID: {self.v_star_token_id}")

    def _get_v_star_position(self) -> int:
        """
        Find the position of [V*] in a typical defect prompt.
        e.g. "An X-ray image showing [V*] defect" → tokenise → find [V*] index.
        """
        prompt = self.config["prompts"]["defect_template"].format(
            placeholder=self.placeholder
        )
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        positions = (tokens == self.v_star_token_id).nonzero(as_tuple=True)[0]
        if len(positions) > 0:
            return positions[0].item()
        return 4   # fallback

    def _apply_lora(self):
        """
        Apply LoRA to UNet and text encoder attention layers via PEFT.

        LoRA keeps all original weights frozen and only trains the small
        rank-decomposed matrices A, B.  This is why DefectFill can fine-tune
        in 400 steps without destroying the pretrained knowledge.
        """
        lora_r = self.lora_cfg["rank"]
        lora_alpha = self.lora_cfg["alpha"]
        target_modules = self.lora_cfg["target_modules"]

        # UNet LoRA
        unet_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, unet_lora_config)

        # Text encoder LoRA (applied to its attention projections)
        te_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        self.text_encoder = get_peft_model(self.text_encoder, te_lora_config)

        print(f"  LoRA applied (rank={lora_r}, alpha={lora_alpha})")

    # ── Encoding Helpers ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to VAE latent space.
        image: (B, 3, H, W), range [-1, 1]
        Returns x0: (B, 4, H//8, W//8)
        """
        return self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent back to pixel space. Returns (B,3,H,W) in [-1,1]."""
        return self.vae.decode(latent / self.vae.config.scaling_factor).sample

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """
        Tokenise and encode a text prompt to CLIP embeddings.
        Returns: (1, seq_len, hidden_dim)
        """
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(**tokens).last_hidden_state

    def get_inpaint_input(
        self,
        x_t: torch.Tensor,          # Noisy latent (B, 4, h, w)
        image: torch.Tensor,         # Original image (B, 3, H, W)
        mask: torch.Tensor,          # Binary mask (B, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the inpainting model input by concatenating:
            x_t  (noisy latent)
            b    (background latent = encode(image * (1-mask)))
            M    (downsampled mask)

        This 9-channel concatenation is the standard SD-inpainting input format.

        Returns:
            x_inpaint: (B, 9, h, w) — ready to pass into UNet
            b_latent:  (B, 4, h, w) — background latent (for reference)
        """
        # Mask out defect region in image to get background
        mask_resized = torch.nn.functional.interpolate(
            mask, size=image.shape[-2:], mode="nearest"
        )
        background = image * (1.0 - mask_resized)
        b_latent = self.encode_image(background)

        # Downsample mask to latent resolution
        latent_h, latent_w = x_t.shape[-2], x_t.shape[-1]
        mask_latent = torch.nn.functional.interpolate(
            mask, size=(latent_h, latent_w), mode="nearest"
        )

        # Concatenate: 4 (x_t) + 4 (b_latent) + 1 (mask) = 9 channels
        x_inpaint = torch.cat([x_t, b_latent, mask_latent], dim=1)
        return x_inpaint, b_latent

    # ── Forward Pass ─────────────────────────────────────────────────────────

    def forward_with_loss_inputs(
        self,
        image: torch.Tensor,       # (B, 3, H, W)
        mask: torch.Tensor,        # (B, 1, H, W)
        defect_prompt: str,
        object_prompt: str,
        rand_mask: torch.Tensor,   # (B, 1, H, W) — random box mask for L_obj
    ) -> Dict[str, torch.Tensor]:
        """
        Run both forward passes (defect-prompt and object-prompt) needed
        to compute all three loss terms.

        This implements Figure 2 from the paper:
          Upper pipeline (Ldef):  image + defect_mask + defect_prompt
          Lower pipeline (Lobj):  image + rand_mask   + object_prompt
        Both share the same noise sample (ε, t).

        Returns dict with all tensors needed by DefectFillLoss.forward().
        """
        batch_size = image.shape[0]

        # 1. Encode clean image to latent x0
        x0 = self.encode_image(image)
        latent_h, latent_w = x0.shape[-2], x0.shape[-1]

        # 2. Sample noise and timestep (SHARED between both pipelines)
        eps = torch.randn_like(x0)
        t = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()

        # 3. Add noise to x0 → x_t  (forward diffusion process)
        x_t = self.noise_scheduler.add_noise(x0, eps, t)

        # ── Defect-prompt pipeline (L_def) ─────────────────────────────────
        x_def_input, _ = self.get_inpaint_input(x_t, image, mask)
        c_def = self.encode_prompt(defect_prompt)
        noise_pred_def = self.unet(x_def_input, t, encoder_hidden_states=c_def).sample

        # ── Object-prompt pipeline (L_obj + L_attn) ────────────────────────
        # Uses RANDOM mask (not the defect mask) to teach object context
        x_obj_input, _ = self.get_inpaint_input(x_t, image, rand_mask)
        c_obj = self.encode_prompt(object_prompt)

        # Clear hook before this forward pass
        self.attn_hook.clear()

        noise_pred_obj = self.unet(x_obj_input, t, encoder_hidden_states=c_obj).sample

        # Extract averaged decoder attention map for [V*]
        attn_map = self.attn_hook.get_averaged_map(latent_h, latent_w)

        return {
            "noise_pred_def": noise_pred_def,
            "noise_pred_obj": noise_pred_obj,
            "noise_target": eps,
            "mask": torch.nn.functional.interpolate(
                mask, size=(latent_h, latent_w), mode="nearest"
            ),
            "attn_map": attn_map,
        }

    # ── Saving / Loading ─────────────────────────────────────────────────────

    def save_lora_weights(self, save_dir: str, class_name: str):
        """Save LoRA weights + [V*] embedding for a specific defect class."""
        import os, torch
        os.makedirs(save_dir, exist_ok=True)
        # Save UNet LoRA
        self.unet.save_pretrained(f"{save_dir}/{class_name}_unet_lora")
        # Save text encoder LoRA
        self.text_encoder.save_pretrained(f"{save_dir}/{class_name}_te_lora")
        # Save [V*] token embedding
        emb = self.text_encoder.get_input_embeddings().weight[self.v_star_token_id]
        torch.save(emb.cpu(), f"{save_dir}/{class_name}_v_star_embedding.pt")
        print(f"  Saved LoRA weights: {save_dir}/{class_name}_*")

    def load_lora_weights(self, save_dir: str, class_name: str):
        """Restore LoRA weights + [V*] embedding for a specific defect class."""
        from peft import PeftModel
        import torch
        self.unet = PeftModel.from_pretrained(
            self.unet, f"{save_dir}/{class_name}_unet_lora"
        )
        self.text_encoder = PeftModel.from_pretrained(
            self.text_encoder, f"{save_dir}/{class_name}_te_lora"
        )
        emb = torch.load(f"{save_dir}/{class_name}_v_star_embedding.pt")
        self.text_encoder.get_input_embeddings().weight.data[self.v_star_token_id] = emb.to(self.device)
        print(f"  Loaded LoRA weights for: {class_name}")
