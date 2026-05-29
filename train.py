"""
train.py
────────
Fine-tune DefectFill (SD2-inpainting + LoRA) for each (object, defect_type)
pair on the MVTec AD dataset.

What happens during training
─────────────────────────────
For each of the 10 objects × their defect types (~40 combinations):
1. Load the reference samples (1/3 of images) as a Dataset.
2. Initialise a DefectFillModel (SD2-inpainting + LoRA + [V*] token).
3. Run N training steps (default 400) with the combined DefectFill loss.
4. Save LoRA weights + [V*] embedding to disk.

Training is extremely lightweight:
• Only LoRA adapters + [V*] embedding are learned (~1M params vs SD's 865M).
• 400 steps × 1 image/step takes ~8 min on a single A100 GPU.

Run:
    python train.py --config configs/config.yaml
    python train.py --config configs/config.yaml --object hazelnut                # all defects for one object
    python train.py --config configs/config.yaml --object hazelnut --defect_type crack  # single run
    python train.py --config configs/config.yaml --resume
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import build_defect_samples, MVTecDefectDataset
from models.defectfill import DefectFillModel
from losses.defectfill_loss import DefectFillLoss, make_random_box_mask


def get_prompts(object_name: str, config: dict) -> tuple:
    """
    Build the two text prompts for a given object.

    Defect prompt : "A photo of [V*]"  — paper exact wording
    Object prompt : "A hazelnut with [V*]"  — per-object context prompt
    """
    placeholder   = config["model"]["placeholder_token"]
    defect_prompt = f"A photo of {placeholder}"
    object_prefix = config["dataset"]["object_prompts"][object_name]
    object_prompt = f"{object_prefix} {placeholder}"
    return defect_prompt, object_prompt


def train_one_defect(
    object_name: str,
    defect_type: str,
    config: dict,
    device: torch.device,
    resume: bool = False,
):
    """
    Fine-tune DefectFill for a single (object, defect_type) pair.

    Args:
        object_name  : e.g. "hazelnut"
        defect_type  : e.g. "crack"
        config       : Full config dict.
        device       : Torch device.
        resume       : If True, load existing LoRA weights and continue.
    """
    run_name     = f"{object_name}_{defect_type}"   # e.g. "hazelnut_crack"
    train_cfg    = config["training"]
    ckpt_dir     = train_cfg["output_dir"]
    dataset_root = config["dataset"]["root"]
    img_size     = config["dataset"]["img_size"]

    print(f"\n{'─'*60}")
    print(f" Training DefectFill for [{run_name.upper()}]")
    print(f"{'─'*60}")

    # ── Check if already trained ──────────────────────────────────────────────
    ckpt_path = Path(ckpt_dir) / f"{run_name}_unet_lora"
    if ckpt_path.exists() and not resume:
        print(f"  [SKIP] Checkpoint exists: {ckpt_path}")
        print(f"  Use --resume to continue training or delete to retrain.")
        return

    # ── Build dataset ─────────────────────────────────────────────────────────
    reference_samples, _ = build_defect_samples(dataset_root, object_name, defect_type, config)
    if not reference_samples:
        print(f"  [SKIP] No reference samples found for: {run_name}")
        return

    dataset = MVTecDefectDataset(
        samples=reference_samples,
        config=config,
        augment=True,
    )

    loader      = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                             pin_memory=False, drop_last=False)
    loader_iter = iter(loader)

    print(f"  Reference samples : {len(dataset)}")
    print(f"  Training steps    : {train_cfg['num_train_steps']}")
    print(f"  Effective batch   : {train_cfg['batch_size'] * train_cfg['gradient_accumulation']}")

    # ── Initialise model ──────────────────────────────────────────────────────
    model = DefectFillModel(config=config, device=device)
    if resume and ckpt_path.exists():
        model.load_lora_weights(ckpt_dir, run_name)
        print("  Resumed from existing checkpoint.")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable_params  = [p for p in model.unet.parameters()         if p.requires_grad]
    trainable_params += [p for p in model.text_encoder.parameters() if p.requires_grad]

    print(f"  Trainable params  : {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(
        [
            {"params": [p for p in model.unet.parameters()         if p.requires_grad],
             "lr": train_cfg["lr_unet"]},
            {"params": [p for p in model.text_encoder.parameters() if p.requires_grad],
             "lr": train_cfg["lr_text_encoder"]},
        ],
        weight_decay=1e-2,
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = DefectFillLoss(
        lambda_def=train_cfg["lambda_def"],
        lambda_obj=train_cfg["lambda_obj"],
        lambda_attn=train_cfg["lambda_attn"],
        alpha_bg=train_cfg["alpha_bg"],
    )

    # ── Mixed precision ───────────────────────────────────────────────────────
    use_fp16 = (train_cfg["mixed_precision"] == "fp16") and device.type == "cuda"
    scaler   = torch.cuda.amp.GradScaler() if use_fp16 else None

    # ── Build prompts ─────────────────────────────────────────────────────────
    defect_prompt, object_prompt = get_prompts(object_name, config)
    print(f"  Defect prompt     : '{defect_prompt}'")
    print(f"  Object prompt     : '{object_prompt}'")

    # ── Training loop ─────────────────────────────────────────────────────────
    model.unet.train()
    model.text_encoder.train()

    log_data   = {"steps": [], "loss_total": [], "loss_def": [], "loss_obj": [], "loss_attn": []}
    num_steps  = train_cfg["num_train_steps"]
    grad_accum = train_cfg["gradient_accumulation"]
    log_every  = train_cfg["log_steps"]
    save_every = train_cfg["save_steps"]

    pbar = tqdm(range(num_steps), desc=f"[{object_name}/{defect_type}]", unit="step")
    optimizer.zero_grad()
    t0 = time.time()

    for step in pbar:

        # ── Get next batch (cycle through small reference set) ────────────────
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch       = next(loader_iter)

        image = batch["pixel_values"].to(device)   # (1, 3, H, W)
        mask  = batch["mask"].to(device)            # (1, 1, H, W)

        # Random box mask for object loss
        rand_mask = make_random_box_mask(
            batch_size=1,
            img_size=img_size,
            num_boxes=config["training"].get("num_random_boxes", 30),
            device=device,
        )

        # ── Forward + Loss ────────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=use_fp16):
            fwd_out = model.forward_with_loss_inputs(
                image=image,
                mask=mask,
                defect_prompt=defect_prompt,
                object_prompt=object_prompt,
                rand_mask=rand_mask,
            )

            total_loss, components = criterion(
                noise_pred_def=fwd_out["noise_pred_def"],
                noise_target=fwd_out["noise_target"],
                mask=fwd_out["mask"],
                noise_pred_obj=fwd_out["noise_pred_obj"],
                attn_map_v=fwd_out.get("attn_map"),
            )

        loss_scaled = total_loss / grad_accum

        # ── Backward ─────────────────────────────────────────────────────────
        if use_fp16:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        # ── Optimiser step ────────────────────────────────────────────────────
        if (step + 1) % grad_accum == 0:
            if use_fp16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────────────────
        if step % log_every == 0:
            pbar.set_postfix(
                loss=f"{components['loss_total']:.4f}",
                def_l=f"{components['loss_def']:.4f}",
                obj_l=f"{components['loss_obj']:.4f}",
                attn_l=f"{components['loss_attn']:.4f}",
            )
            log_data["steps"].append(step)
            for k in ["loss_total", "loss_def", "loss_obj", "loss_attn"]:
                log_data[k].append(components[k])

        # ── Intermediate checkpoint ───────────────────────────────────────────
        if (step + 1) % save_every == 0:
            model.save_lora_weights(ckpt_dir, run_name)

    # ── Final save ────────────────────────────────────────────────────────────
    model.save_lora_weights(ckpt_dir, run_name)
    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed/60:.1f} min")
    print(f"  Final loss: {log_data['loss_total'][-1]:.4f}")

    log_path = Path(ckpt_dir) / f"{run_name}_train_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"  Training log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train DefectFill on MVTec AD Dataset")
    parser.add_argument("--config",      default="configs/config.yaml")
    parser.add_argument("--object",      type=str, default=None,
                        help="Train only this object, e.g. 'hazelnut'. Default: all objects.")
    parser.add_argument("--defect_type", type=str, default=None,
                        help="Train only this defect type, e.g. 'crack'. Default: all types.")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume training from existing checkpoints.")
    parser.add_argument("--device",      default="auto",
                        help="Device: 'cpu', 'cuda', 'cuda:0', or 'auto'.")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")

    # ── Determine which objects/defect_types to train ─────────────────────────
    objects      = config["dataset"]["objects"]
    defect_types = config["dataset"]["defect_types"]

    if args.object:
        if args.object not in objects:
            raise ValueError(f"Unknown object '{args.object}'. Valid: {objects}")
        objects = [args.object]

    if args.defect_type:
        defect_types = {obj: [args.defect_type] for obj in objects}

    total_runs = sum(len(defect_types[obj]) for obj in objects)
    print(f"\nObjects to train  : {objects}")
    print(f"Total training runs: {total_runs}")

    # ── Train each (object, defect_type) ─────────────────────────────────────
    for object_name in objects:
        for defect_type in defect_types[object_name]:
            train_one_defect(
                object_name=object_name,
                defect_type=defect_type,
                config=config,
                device=device,
                resume=args.resume,
            )

    print(f"\n{'='*60}")
    print("  All training runs complete! Run generate.py next.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()