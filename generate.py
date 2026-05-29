"""
generate.py
───────────
Generate synthetic defect images using fine-tuned DefectFill models
on the MVTec AD dataset.

Pipeline per (object, defect_type):
1. Load fine-tuned LoRA weights for the run.
2. For each mask in the "target" split (2/3 not used for training):
   a. Pick a random NORMAL image as the background (paper setup).
   b. Load the pixel-perfect ground-truth mask from MVTec.
   c. Generate N candidates using DDIM (50 steps).
   d. Apply Low-Fidelity Selection → pick the most defect-like candidate.
   e. Save the selected image + mask.
3. Continue until num_samples images are generated.

Run:
    python generate.py --config configs/config.yaml
    python generate.py --config configs/config.yaml --object hazelnut --defect_type crack --num_samples 200
    python generate.py --config configs/config.yaml --custom_masks   # use star/square masks
"""

import argparse
import os
import sys
import json
import random
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (
    build_defect_samples,
    get_normal_images,
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
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    if shape == "square":
        cv2.rectangle(mask, (cx - size, cy - size), (cx + size, cy + size), 255, -1)
    elif shape == "circle":
        cv2.circle(mask, (cx, cy), size, 255, -1)
    elif shape == "ellipse":
        cv2.ellipse(mask, (cx, cy), (size, size // 2), 0, 0, 360, 255, -1)
    elif shape == "star":
        import math
        pts = []
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
        self.config      = config
        self.device      = device
        self.gen_cfg     = config["generation"]
        self.img_size    = config["dataset"]["img_size"]
        self.placeholder = config["model"]["placeholder_token"]

        print("  Loading SD2-inpainting pipeline...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            config["model"]["base_model_id"],
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            safety_checker=None,
        ).to(device)

        # Replace scheduler with DDIM
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # LFS quality selector
        self.selector = LowFidelitySelector(
            metric=self.gen_cfg["lfs_metric"],
            device=str(device),
        )

    def load_class_weights(self, run_name: str, ckpt_dir: str):
        """
        Hot-swap LoRA weights for a given (object, defect_type) run into the pipeline.
        run_name: e.g. "hazelnut_crack"
        """
        from peft import PeftModel

        self.pipe.unet = PeftModel.from_pretrained(
            self.pipe.unet,
            f"{ckpt_dir}/{run_name}_unet_lora",
        ).to(self.device)

        self.pipe.text_encoder = PeftModel.from_pretrained(
            self.pipe.text_encoder,
            f"{ckpt_dir}/{run_name}_te_lora",
        ).to(self.device)

        v_star_id = self.pipe.tokenizer.convert_tokens_to_ids(self.placeholder)
        emb = torch.load(f"{ckpt_dir}/{run_name}_v_star_embedding.pt")
        self.pipe.text_encoder.get_input_embeddings().weight.data[v_star_id] = \
            emb.to(self.device)

        print(f"  Loaded weights for: {run_name}")

    def generate_candidates(
        self,
        image_pil: Image.Image,
        mask_pil: Image.Image,
        prompt: str,
        n_candidates: int,
        seed_base: int = 0,
    ) -> List[Image.Image]:
        """Generate N candidate images with different seeds."""
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
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t

    def mask_pil_to_tensor(self, mask_pil: Image.Image) -> torch.Tensor:
        """Grayscale mask PIL → torch (1,1,H,W) in [0,1]."""
        arr = np.array(mask_pil.convert("L")).astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    def generate_for_defect(
        self,
        object_name: str,
        defect_type: str,
        ckpt_dir: str,
        num_samples: int,
        use_custom_masks: bool = False,
    ):
        """
        Full generation pipeline for one (object, defect_type) pair.

        Paper setup:
          - Use TARGET split masks (pixel-perfect ground-truth)
          - Apply masks onto NORMAL background images (from train/good/)
          - Generate 1000 images per defect category

        Saves to:
          outputs/generated/{object_name}/{defect_type}/images/
          outputs/generated/{object_name}/{defect_type}/masks/
        """
        run_name     = f"{object_name}_{defect_type}"
        dataset_root = self.config["dataset"]["root"]

        # Output directories
        out_dir = Path(self.gen_cfg["output_dir"]) / object_name / defect_type
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)

        print(f"\n  Generating [{run_name}] — target={num_samples} images")

        # Load LoRA weights for this run
        self.load_class_weights(run_name, ckpt_dir)

        # Get target split: these provide the MASKS to use
        _, target_samples = build_defect_samples(
            dataset_root, object_name, defect_type, self.config
        )
        if not target_samples:
            print(f"  [WARN] No target samples for {run_name}, skipping.")
            return

        # Get normal (defect-free) images to use as backgrounds — paper setup
        normal_images = get_normal_images(dataset_root, object_name)
        if not normal_images:
            print(f"  [WARN] No normal images found for {object_name}, skipping.")
            return

        # Object-context prompt: "A hazelnut with [V*]"
        object_prefix = self.config["dataset"]["object_prompts"][object_name]
        object_prompt = f"{object_prefix} {self.placeholder}"

        n_candidates    = self.gen_cfg["num_candidates"]
        generated_count = 0
        img_index       = 0

        pbar = tqdm(total=num_samples, desc=f"  [{run_name}]", unit="img")

        while generated_count < num_samples:
            # Cycle through target samples
            sample = target_samples[img_index % len(target_samples)]
            img_index += 1

            # ── Choose mask ────────────────────────────────────────────────
            if use_custom_masks:
                shape = SHAPE_CYCLE[generated_count % len(SHAPE_CYCLE)]
                cx    = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
                cy    = np.random.randint(self.img_size // 4, 3 * self.img_size // 4)
                size  = np.random.randint(20, 60)
                mask_arr = make_shape_mask(self.img_size, shape, cx, cy, size)
            else:
                # Load pixel-perfect ground-truth mask from MVTec
                mask_arr = cv2.imread(sample.mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_arr is None:
                    continue
                # Normalise to 0/255
                if mask_arr.max() <= 1:
                    mask_arr = (mask_arr * 255).astype(np.uint8)

            # ── Use a random NORMAL image as background (paper setup) ──────
            normal_path = random.choice(normal_images)
            img_bgr     = cv2.imread(normal_path)
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Handle grayscale MVTec objects (zipper, screw, grid)
            grayscale_objects = self.config["dataset"].get("grayscale_objects", [])
            if object_name in grayscale_objects:
                if img_rgb.ndim == 2 or (img_rgb.ndim == 3 and img_rgb.shape[2] == 1):
                    img_rgb = gray_to_rgb(img_rgb)

            # ── Resize ────────────────────────────────────────────────────
            img_resized  = cv2.resize(img_rgb,  (self.img_size, self.img_size))
            mask_resized = cv2.resize(mask_arr, (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_NEAREST)

            # ── Convert to PIL ────────────────────────────────────────────
            img_pil  = Image.fromarray(img_resized.astype(np.uint8))
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
            orig_tensor  = self.pil_to_tensor(img_pil)
            mask_tensor  = self.mask_pil_to_tensor(mask_pil)
            cand_tensors = [self.pil_to_tensor(c) for c in candidates_pil]

            best_idx, best_score = self.selector.select(cand_tensors, orig_tensor, mask_tensor)
            best_img_pil = candidates_pil[best_idx]

            # ── Save ──────────────────────────────────────────────────────
            stem = f"{object_name}_{defect_type}_{generated_count:05d}"
            best_img_pil.save(str(out_dir / "images" / f"{stem}.png"))
            mask_pil.save(str(out_dir / "masks" / f"{stem}.png"))

            generated_count += 1
            pbar.update(1)

        pbar.close()
        print(f"  ✓ Generated {generated_count} images → {out_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate defect images with DefectFill (MVTec AD)")
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--object",       type=str, default=None,
                        help="Generate for only this object, e.g. 'hazelnut'. Default: all.")
    parser.add_argument("--defect_type",  type=str, default=None,
                        help="Generate for only this defect type. Default: all.")
    parser.add_argument("--num_samples",  type=int, default=None,
                        help="Override num_samples_per_class from config.")
    parser.add_argument("--custom_masks", action="store_true",
                        help="Use geometric masks (star/square) to test generalisation.")
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    num_samples  = args.num_samples or config["generation"]["num_samples_per_class"]
    ckpt_dir     = config["training"]["output_dir"]
    objects      = config["dataset"]["objects"]
    defect_types = config["dataset"]["defect_types"]

    # Filter by --object / --defect_type flags
    if args.object:
        if args.object not in objects:
            raise ValueError(f"Unknown object '{args.object}'. Valid: {objects}")
        objects = [args.object]
    if args.defect_type:
        defect_types = {obj: [args.defect_type] for obj in objects}

    # Initialise generator (loads base SD pipeline once, then hot-swaps LoRA per run)
    generator = DefectGenerator(config=config, device=device)

    for object_name in objects:
        for defect_type in defect_types[object_name]:
            run_name  = f"{object_name}_{defect_type}"
            ckpt_path = Path(ckpt_dir) / f"{run_name}_unet_lora"
            if not ckpt_path.exists():
                print(f"  [SKIP] No checkpoint found for: {run_name}  (run train.py first)")
                continue

            generator.generate_for_defect(
                object_name=object_name,
                defect_type=defect_type,
                ckpt_dir=ckpt_dir,
                num_samples=num_samples,
                use_custom_masks=args.custom_masks,
            )

    print(f"\n{'='*60}")
    print("  Generation complete! Run evaluate.py next.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()