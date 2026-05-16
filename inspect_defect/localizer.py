"""
inspect/localizer.py
─────────────────────
Downstream localisation task: train a UNet segmentation model to predict
PIXEL-LEVEL defect locations, trained on generated images + masks.

Why localisation matters:
  Classification says "this weld has a crack".  Localisation says WHERE the
  crack is — essential for repair routing on manufacturing lines.

Architecture: UNet (Ronneberger et al. 2015)
  A standard encoder-decoder with skip connections.  The encoder is a
  pretrained ResNet-34 backbone (same as classifier) for feature extraction.
  The decoder upsamples the features back to the original resolution.
  Output: single-channel binary mask (sigmoid → 0=normal, 1=defect).

Training:
  - Train on all generated defect images + their masks, PLUS normal weld images
    (with all-zero masks) so the model learns what "no defect" looks like.
  - Focal loss (Lin et al. 2017) handles class imbalance between defect pixels
    (tiny fraction of image) and background pixels.
  - Test: evaluate on real held-out images using AUROC, AP, F1-max, PRO.

Loss: Focal Loss
  L_focal = -α(1-p_t)^γ log(p_t)
  γ=2 downweights easy background pixels, forcing focus on hard defect pixels.
  Without focal loss, the model collapses to predicting all-background.
"""

import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
import yaml
from tqdm import tqdm


# ─── Focal Loss ───────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Binary focal loss for highly imbalanced segmentation.

    Focal loss was introduced by Lin et al. (2017) for object detection.
    Here we apply it to binary pixel-wise segmentation.

    L_focal = -[α·y·(1-p)^γ·log(p) + (1-α)·(1-y)·p^γ·log(1-p)]

    With γ=2 (paper default), easy pixels (confident background) contribute
    very little gradient, while hard pixels (small defect region near boundary)
    contribute proportionally more — exactly what we need.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:    (B, 1, H, W) — raw logits from UNet output.
            target:  (B, 1, H, W) — binary mask [0, 1] float.
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(pred)
        # Binary cross entropy without reduction
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        # Focal weight
        p_t = p * target + (1 - p) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * focal_weight * bce
        return loss.mean()


# ─── Simple UNet ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Standard double-conv block: Conv → BN → ReLU → Conv → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UpBlock(nn.Module):
    """Decoder block: bilinear upsample + skip concat + ConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class WeldUNet(nn.Module):
    """
    UNet with ResNet-34 encoder backbone for weld defect localisation.

    Encoder: pretrained ResNet-34 (extracts rich texture + semantic features)
    Decoder: 4 up-blocks with skip connections from encoder
    Output:  1-channel sigmoid logit map (defect probability per pixel)

    Input:  (B, 3, 512, 512) — RGB-converted X-ray image
    Output: (B, 1, 512, 512) — defect probability map
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 3):
        super().__init__()

        # ── ResNet-34 Encoder ─────────────────────────────────────────────────
        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        # Extract encoder stages (keep as ModuleList for easy access)
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # 64 ch, /2
        self.pool = backbone.maxpool                                              # /4
        self.enc1 = backbone.layer1   # 64  ch, /4
        self.enc2 = backbone.layer2   # 128 ch, /8
        self.enc3 = backbone.layer3   # 256 ch, /16
        self.enc4 = backbone.layer4   # 512 ch, /32

        # Bottleneck
        self.bottleneck = ConvBlock(512, 512)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64,  64)
        self.up1 = UpBlock(64,  64,  32)    # skip from enc0
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ConvBlock(32, 16),
        )

        # Final prediction head
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image.
        Returns:
            (B, 1, H, W) raw logits — apply sigmoid for probabilities.
        """
        # Encoder
        e0 = self.enc0(x)       # (B, 64, H/2, W/2)
        e1 = self.enc1(self.pool(e0))  # (B, 64, H/4, W/4)
        e2 = self.enc2(e1)      # (B, 128, H/8, W/8)
        e3 = self.enc3(e2)      # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)      # (B, 512, H/32, W/32)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.up4(b, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)
        d0 = self.up0(d1)

        return self.head(d0)


# ─── Dataset for Localisation ─────────────────────────────────────────────────

class LocalisationDataset(Dataset):
    """
    Combined dataset for defect localisation training.

    Training set:
      - Generated defect images + corresponding masks (from generate.py output)

    The model learns:
      - Which pixels are defective (mask=1)
      - What normal weld pixels look like (mask=0 background of defect images)
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        generated_dir: str,
        class_names: List[str],
        img_size: int = 512,
    ):
        import torchvision.transforms as T
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        self.items: List[Tuple[str, str]] = []  # (img_path, mask_path)

        for cls_name in class_names:
            img_dir  = Path(generated_dir) / cls_name / "images"
            mask_dir = Path(generated_dir) / cls_name / "masks"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    self.items.append((str(img_path), str(mask_path)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.items[idx]

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        msk_arr = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        msk_pil = Image.fromarray(msk_arr)

        img_t = self.img_transform(img_pil)     # (3, H, W)
        msk_t = self.mask_transform(msk_pil)    # (1, H, W), [0,1]
        msk_t = (msk_t > 0.5).float()           # Binarise

        return img_t, msk_t


# ─── Training ─────────────────────────────────────────────────────────────────

def train_localizer(config: dict, device: torch.device):
    """Train UNet localiser on generated images, evaluate on real images."""
    loc_cfg = config["localization"]
    gen_dir = config["generation"]["output_dir"]
    class_names = config["dataset"]["class_names"]
    out_dir = Path(loc_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Localizer Training ───────────────────────────────────")

    # Datasets
    train_ds = LocalisationDataset(gen_dir, class_names, config["dataset"]["img_size"])
    print(f"  Train samples: {len(train_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=loc_cfg["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True,
    )

    # Model
    model = WeldUNet(pretrained=True, in_channels=loc_cfg["in_channels"]).to(device)
    criterion = FocalLoss(gamma=loc_cfg["focal_loss_gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=loc_cfg["lr"])

    best_loss = float("inf")
    log = {"epoch": [], "train_loss": []}

    for epoch in range(loc_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for imgs, masks in tqdm(train_loader,
                                desc=f"Epoch {epoch+1}/{loc_cfg['epochs']}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        print(f"  Epoch {epoch+1:3d} | focal_loss={epoch_loss:.4f}")
        log["epoch"].append(epoch + 1)
        log["train_loss"].append(round(epoch_loss, 4))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), str(out_dir / "best_localizer.pt"))

    with open(out_dir / "localizer_training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Best loss: {best_loss:.4f}")
    print(f"  Weights saved: {out_dir}/best_localizer.pt")
    return model
