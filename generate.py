"""
generate.py
───────────
Generate synthetic weld defect images using fine-tuned DefectFill models.

This script implements Section 3.3 of the paper: "Generating Defect".

Pipeline per defect class:
  1. Load fine-tuned LoRA weights for the class.
  2. For each "target" image (the 2/3 split not used for training):
       a. Choose a mask — either from the existing YOLO annotation (same location)
          or draw a custom shape mask to test generalisation.
       b. Generate N candidates using DDIM (50 steps).
       c. Apply Low-Fidelity Selection → pick the most defect-like candidate.
       d. Save the selected image + mask as a training pair.
  3. Continue until num_samples_per_class images are generated.

The generated dataset augments the original, enabling downstream classifiers
and localisers to be trained on far more defect examples.

Run:
    python generate.py --config configs/config.yaml
    python generate.py --config configs/config.yaml --class_id 1 --num_samples 200
    python generate.py --config configs/config.yaml --custom_masks  # use star/square masks
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import (
    build_class_samples,
    yolo_bbox_to_pixel,
    bbox_to_mask,
    gray_to_rgb,
)
from models.defectfill import DefectFillModel
from models.lfs import LowFidelitySelector


# ─── Custom Mask Shapes (for generalisation testing) ──────────────────────────

def make_shape_mask(img_size: int, shape: str, cx: int, cy: int, size: int = 60) -> np.ndarray:
    """
    Create a mask with a geometric shape (star, square, circle, ellipse).
    Used to test if DefectFill can generalise to unseen mask shapes at inference.

    The paper demonstrates this in Figure 1 — defects are generated in star
    and square shapes even though training masks were all irregular blobs.

    Args:
        img_size:  Output mask size (square: img_size × img_size).
        shape:     One of "square", "circle", "ellipse", "star".
        cx, cy:    Centre of the shape.
        size:      Approximate radius / half-width.

    Returns:
        mask: uint8 (img_size, img_size), 255=defect.
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    if shape == "square":
        cv2.rectangle(mask, (cx - size, cy - size), (cx + size, cy + size), 255, -1)
    elif shape == "circle":
        cv2.circle(mask, (cx, cy), size, 255, -1)
    elif shape == "ellipse":
        cv2.ellipse(mask, (cx, cy), (size, size // 2), 0, 0, 360, 255, -1)
    elif shape == "star":
        pts = []
        import math
        for i in range(5):
            outer_angle = math.pi / 2 + i * 2 * math.pi / 5
            inner_angle = outer_angle + math.pi / 5
            pts.append((int(cx + size * math.cos(outer_angle)),
                         int(cy - size * math.sin(outer_angle))))
            pts.append((int(cx + size * 0.4 * math.cos(inner_angle)),
                         int(cy - size * 0.4 * math.sin(inner_angle))))
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


SHAPE_CYCLE = ["square", "circle", "ellipse", "star"]


# ─── Generator ────────────────────────────────────────────────────────────────

class DefectGenerator:
    """
    Wraps the fine-tuned SD2-inpainting pipeline for inference-time
    defect generation with LFS quality filtering.
    """

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.gen_cfg = config["generation"]
        self.img_size = config["dataset"]["img_size"]
        self.placeholder = config["model"]["placeholder_token"]

        # Build the full pipeline (VAE + UNet + text encoder + scheduler)
        print("  Loading SD2-inpainting pipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config["model"]["base_model_id"],
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=None,
        ).to(device)

        # Replace scheduler with DDIM for faster, deterministic inference
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # LFS quality selector
        self.selector = LowFidelitySelector(
            metric=self.gen_cfg["lfs_metric"],
            device=str(device),
        )

    def load_class_weights(self, class_name: str, ckpt_dir: str):
        """
        Hot-swap the LoRA weights for a given defect class into the pipeline.
        This avoids reloading the full ~4GB model for each class.
        """
        from peft import PeftModel
        # Load LoRA-wrapped UNet
        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            f"{ckpt_dir}/{class_name}_unet_lora",
        ).to(self.device)
        # Load LoRA-wrapped text encoder + [V*] embedding
        self.pipe.text_encoder = PeftModel.from_pretrained(
            self.pipe.text_encoder,
            f"{ckpt_dir}/{class_name}_te_lora",
        ).to(self.device)
        # Restore [V*] token embedding
        v_star_id = self.pipe.tokenizer.convert_tokens_to_ids(self.placeholder)
        emb = torch.load(f"{ckpt_dir}/{class_name}_v_star_embedding.pt")
        self.pipe.text_encoder.get_input_embeddings().weight.data[v_star_id] = \
            emb.to(self.device)
        print(f"  Loaded weights for: {class_name}")

    def generate_candidates(
        self,
        image_pil: Image.Image,     # RGB PIL image
        mask_pil: Image.Image,      # Grayscale mask PIL (255=defect)
        prompt: str,
        n_candidates: int,
        seed_base: int = 0,
    ) -> List[Image.Image]:
        """
        Generate N candidate defect images using DDIM inpainting.

        Each candidate uses a different seed so we get diverse results.
        LFS will then pick the best one.

        Args:
            image_pil:    Background (defect-free) image.
            mask_pil:     Mask indicating where to generate the defect.
            prompt:       Object-context prompt for inference.
            n_candidates: How many samples to generate.
            seed_base:    Starting seed (each candidate gets seed_base + i).

        Returns:
            List of N PIL images (generated).
        """
        candidates = []
        for i in range(n_candidates):
            generator = torch.Generator(device=self.device).manual_seed(seed_base + i)
            result = self.pipe(
                prompt=prompt,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=self.gen_cfg["ddim_steps"],
                guidance_scale=self.gen_cfg["guidance_scale"],
                generator=generator,
            )
            candidates.append(result.images[0])
        return candidates

    def pil_to_tensor(self, img_pil: Image.Image) -> torch.Tensor:
        """PIL (H,W,3) → torch (1,3,H,W) in [-1,1]."""
        arr = np.array(img_pil.convert("RGB")).astype(np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t

    def mask_pil_to_tensor(self, mask_pil: Image.Image) -> torch.Tensor:
        """Grayscale mask PIL → torch (1,1,H,W) in [0,1]."""
        arr = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    def generate_for_class(
        self,
        class_id: int,
        ckpt_dir: str,
        num_samples: int,
        use_custom_masks: bool = False,
    ):
        """
        Full generation pipeline for one defect class.

        Saves generated images and masks to:
            outputs/generated/{class_name}/images/
            outputs/generated/{class_name}/masks/
        """
        class_name = self.config["dataset"]["class_names"][class_id]
        dataset_root = self.config["dataset"]["root"]
        out_dir = Path(self.gen_cfg["output_dir"]) / class_name
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

        print(f"\n  Generating class [{class_name}]  target={num_samples} images")

        # Load class-specific LoRA weights
        self.load_class_weights(class_name, ckpt_dir)

        # Get target images (2/3 split — these are used as backgrounds)
        _, target_samples = build_class_samples(dataset_root, class_id, self.config)
        if not target_samples:
            print(f"  [WARN] No target samples for {class_name}, skipping.")
            return

        # Object prompt for inference (same as training)
        object_prompt = self.config["prompts"]["object_template"].format(
            placeholder=self.placeholder
        )

        n_candidates = self.gen_cfg["num_candidates"]
        generated_count = 0
        img_index = 0
        pbar = tqdm(total=num_samples, desc=f"  [{class_name}]", unit="img")

        while generated_count < num_samples:
            # Cycle through target samples
            sample = target_samples[img_index % len(target_samples)]
            img_index += 1

            # Load image
            img_bgr = cv2.imread(sample.image_path)
            if img_bgr is None:
                continue
            img_h, img_w = img_bgr.shape[:2]
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            img_rgb = gray_to_rgb(img_gray)

            # ── Choose mask ────────────────────────────────────────────────
            if use_custom_masks:
                # Generate a random geometric mask shape
                shape = SHAPE_CYCLE[generated_count % len(SHAPE_CYCLE)]
                cx = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
                cy = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
                size = np.random.randint(20, 60)
                mask_arr = make_shape_mask(self.img_size, shape, cx, cy, size)
            else:
                # Use annotation-based mask (dilated bbox)
                bboxes_px = [
                    yolo_bbox_to_pixel(bn, img_w, img_h)
                    for cid, bn in zip(sample.class_ids, sample.bboxes_norm)
                    if cid == class_id
                ]
                if not bboxes_px:
                    continue
                mask_arr = bbox_to_mask(
                    bboxes_px, img_h, img_w,
                    self.config["dataset"].get("dilation_factor", 1.15)
                )

            # Resize
            img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))
            mask_resized = cv2.resize(mask_arr, (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_NEAREST)

            # Convert to PIL for diffusers pipeline
            img_pil = Image.fromarray(img_resized.astype(np.uint8))
            mask_pil = Image.fromarray(mask_resized)

            # ── Generate N candidates ─────────────────────────────────────
            candidates_pil = self.generate_candidates(
                image_pil=img_pil,
                mask_pil=mask_pil,
                prompt=object_prompt,
                n_candidates=n_candidates,
                seed_base=generated_count * n_candidates,
            )

            # ── LFS: pick the best candidate ──────────────────────────────
            # Convert candidates to tensors for LPIPS comparison
            orig_tensor = self.pil_to_tensor(img_pil)
            mask_tensor = self.mask_pil_to_tensor(mask_pil)
            cand_tensors = [self.pil_to_tensor(c) for c in candidates_pil]

            best_idx, best_score = self.selector.select(cand_tensors, orig_tensor, mask_tensor)
            best_img_pil = candidates_pil[best_idx]

            # ── Save ──────────────────────────────────────────────────────
            stem = f"{class_name}_{generated_count:05d}"
            best_img_pil.save(str(out_dir / "images" / f"{stem}.png"))
            mask_pil.save(str(out_dir / "masks" / f"{stem}.png"))

            generated_count += 1
            pbar.update(1)

        pbar.close()
        print(f"  ✓ Generated {generated_count} images → {out_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate defect images with DefectFill")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--class_id", type=int, default=None,
                        help="Generate for only this class ID. Default: all.")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override num_samples_per_class from config.")
    parser.add_argument("--custom_masks", action="store_true",
                        help="Use geometric masks (star/square) to test generalisation.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    num_samples = args.num_samples or config["generation"]["num_samples_per_class"]
    ckpt_dir = config["training"]["output_dir"]
    class_names = config["dataset"]["class_names"]

    # Initialise generator (loads base SD pipeline once)
    generator = DefectGenerator(config=config, device=device)

    class_ids = [args.class_id] if args.class_id is not None else list(range(len(class_names)))

    for class_id in class_ids:
        ckpt_path = Path(ckpt_dir) / f"{class_names[class_id]}_unet_lora"
        if not ckpt_path.exists():
            print(f"  [SKIP] No checkpoint found for: {class_names[class_id]}")
            print(f"         Run train.py first.")
            continue

        generator.generate_for_class(
            class_id=class_id,
            ckpt_dir=ckpt_dir,
            num_samples=num_samples,
            use_custom_masks=args.custom_masks,
        )

    print(f"\n{'='*60}")
    print("  Generation complete!  Run evaluate.py next.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
