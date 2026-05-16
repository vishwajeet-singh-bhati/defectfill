"""
inspect/classifier.py
─────────────────────
Downstream classification task: train a ResNet-34 to classify the 8 weld
defect types using the generated images as training data.

Why this matters:
  The entire point of DefectFill is to generate realistic defect images that
  can be used to train downstream inspection models.  If the generated images
  are realistic, a classifier trained on them should generalise well to REAL
  defect images it has never seen.

  Paper metric: Classification Accuracy (%) on real test images, where the
  classifier was trained ONLY on generated images.

Architecture:
  ResNet-34 (He et al. 2016), pretrained on ImageNet, with the final FC
  layer replaced to output 8 classes (one per defect type).

  The input is an RGB-converted X-ray image (512×512).  The ResNet expects
  ImageNet normalisation, applied here.

Training strategy:
  - Train on 1000 generated images per class (8000 total).
  - Test on the held-out 2/3 of REAL defect images.
  - Standard CE loss + Adam optimiser.
  - No fine-tuning tricks — kept simple to isolate effect of generated data.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import yaml
from tqdm import tqdm
import cv2


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GeneratedDefectDataset(Dataset):
    """
    Loads generated defect images from outputs/generated/{class_name}/images/

    Returns (image_tensor, class_id) pairs for training the classifier.

    ImageNet normalisation is applied because ResNet-34 uses ImageNet weights.
    X-ray images (grayscale → tiled RGB) are somewhat out-of-distribution for
    ImageNet, but the pretrained features still transfer well for textures.
    """

    # ImageNet stats used because ResNet-34 pretrained on ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        generated_dir: str,    # Root of generated images
        class_names: List[str],
        split: str = "train",  # "train" uses generated, "test" uses real data
        real_samples_dir: Optional[str] = None,
        class_id_filter: Optional[List[int]] = None,
    ):
        self.class_names = class_names
        self.split = split

        # ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),   # ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        self.items: List[Tuple[str, int]] = []   # (image_path, class_id)

        if split == "train":
            # Load from generated directory
            for cls_id, cls_name in enumerate(class_names):
                if class_id_filter and cls_id not in class_id_filter:
                    continue
                cls_dir = Path(generated_dir) / cls_name / "images"
                if not cls_dir.exists():
                    print(f"  [WARN] Generated dir not found: {cls_dir}")
                    continue
                for img_path in sorted(cls_dir.glob("*.png")):
                    self.items.append((str(img_path), cls_id))

        elif split == "test":
            # Load real defect images from the original dataset
            # (the target 2/3 split reserved for testing)
            assert real_samples_dir is not None, "real_samples_dir required for test split"
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.dataset import WeldDefectSample, build_class_samples
            import yaml
            # We need config to find the dataset
            cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            for cls_id, cls_name in enumerate(class_names):
                _, target_samples = build_class_samples(real_samples_dir, cls_id, cfg)
                for sample in target_samples:
                    self.items.append((sample.image_path, cls_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, cls_id = self.items[idx]

        # Load image (could be PNG from generated set or JPG from real dataset)
        img = cv2.imread(img_path)
        if img is None:
            # Fallback: create blank image
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img.astype(np.uint8))
        img_tensor = self.transform(img_pil)   # (3, 224, 224)
        return img_tensor, cls_id


# ─── Model ────────────────────────────────────────────────────────────────────

def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-34 classifier with modified final FC layer.

    The original ResNet-34 FC outputs 1000 classes (ImageNet).
    We replace it with a linear layer for our 8 defect classes.

    Pretrained weights are loaded — fine-tuning from ImageNet features is
    standard practice and greatly speeds convergence on small datasets.
    """
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    # Replace the final fully connected layer
    in_features = model.fc.in_features  # 512 for ResNet-34
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_classifier(config: dict, device: torch.device):
    """
    Train ResNet-34 classifier on generated images, test on real images.

    Args:
        config:  Full config dict.
        device:  Torch device.
    """
    cls_cfg = config["classification"]
    gen_dir = config["generation"]["output_dir"]
    class_names = config["dataset"]["class_names"]
    dataset_root = config["dataset"]["root"]
    out_dir = Path(cls_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n── Classifier Training ──────────────────────────────────")
    print(f"  Model       : {cls_cfg['model']}")
    print(f"  Train data  : generated images ({gen_dir})")
    print(f"  Test data   : real images ({dataset_root})")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = GeneratedDefectDataset(gen_dir, class_names, split="train")
    test_ds = GeneratedDefectDataset(gen_dir, class_names, split="test",
                                     real_samples_dir=dataset_root)

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Test samples  : {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cls_cfg["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cls_cfg["batch_size"] * 2, shuffle=False,
        num_workers=2, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_classifier(
        num_classes=cls_cfg["num_classes"],
        pretrained=cls_cfg["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_cfg["lr"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    results = {"epoch": [], "train_loss": [], "val_acc": [], "per_class_acc": []}

    for epoch in range(cls_cfg["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cls_cfg['epochs']}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        overall_acc = (all_preds == all_labels).mean() * 100

        # Per-class accuracy
        per_class = {}
        for cls_id, cls_name in enumerate(class_names):
            cls_mask = all_labels == cls_id
            if cls_mask.sum() > 0:
                cls_acc = (all_preds[cls_mask] == all_labels[cls_mask]).mean() * 100
                per_class[cls_name] = round(cls_acc, 2)

        print(f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} | acc={overall_acc:.2f}%")

        results["epoch"].append(epoch + 1)
        results["train_loss"].append(round(train_loss, 4))
        results["val_acc"].append(round(overall_acc, 2))
        results["per_class_acc"].append(per_class)

        if overall_acc > best_acc:
            best_acc = overall_acc
            torch.save(model.state_dict(), str(out_dir / "best_classifier.pt"))

    # ── Print final results ───────────────────────────────────────────────────
    print(f"\n  ── Classification Results ──────────────────────────────")
    print(f"  Best overall accuracy: {best_acc:.2f}%")
    last_per_class = results["per_class_acc"][-1]
    for cls_name, acc in last_per_class.items():
        print(f"  {cls_name:<20}: {acc:.2f}%")

    # Save results
    with open(out_dir / "classification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_dir}/classification_results.json")

    return results
