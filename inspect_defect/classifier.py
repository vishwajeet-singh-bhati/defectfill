"""
inspect_defect/classifier.py
─────────────────────────────
Downstream classification task: train a ResNet-34 to classify defect types
within each MVTec AD object category using generated images as training data.

Why this matters:
  The entire point of DefectFill is to generate realistic defect images that
  can be used to train downstream inspection models. If the generated images
  are realistic, a classifier trained on them should generalise well to REAL
  defect images it has never seen.

  Paper metric (Table 3): Classification Accuracy (%) on real test images,
  where the classifier was trained ONLY on generated images.
  One classifier per object — classifying defect_type within that object.
  e.g. for hazelnut: classify {crack, cut, hole, print}

Architecture:
  ResNet-34 (He et al. 2016), pretrained on ImageNet, with the final FC
  layer replaced to output N classes (one per defect_type for the object).

Training strategy:
  - Train on generated images from outputs/generated/{object}/{defect_type}/
  - Test on the held-out 2/3 of REAL MVTec defect images.
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
    Loads generated defect images from:
      outputs/generated/{object_name}/{defect_type}/images/

    Returns (image_tensor, class_id) pairs for training the classifier.
    class_id is the index into defect_types list for this object.

    ImageNet normalisation is applied because ResNet-34 uses ImageNet weights.
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        generated_dir: str,
        object_name: str,
        defect_types: List[str],
        split: str = "train",           # "train" = generated, "test" = real MVTec
        real_dataset_root: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.object_name  = object_name
        self.defect_types = defect_types
        self.split        = split

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

        self.items: List[Tuple[str, int]] = []   # (image_path, class_id)

        if split == "train":
            # Load from generated directory: one class_id per defect_type
            for cls_id, defect_type in enumerate(defect_types):
                cls_dir = Path(generated_dir) / object_name / defect_type / "images"
                if not cls_dir.exists():
                    print(f"  [WARN] Generated dir not found: {cls_dir}")
                    continue
                for img_path in sorted(cls_dir.glob("*.png")):
                    self.items.append((str(img_path), cls_id))

        elif split == "test":
            # Load real MVTec defect images (the target 2/3 split)
            assert real_dataset_root is not None, "real_dataset_root required for test split"
            assert config is not None, "config required for test split"

            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data.dataset import build_defect_samples

            for cls_id, defect_type in enumerate(defect_types):
                _, target_samples = build_defect_samples(
                    real_dataset_root, object_name, defect_type, config
                )
                for sample in target_samples:
                    self.items.append((sample.image_path, cls_id))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, cls_id = self.items[idx]

        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil    = Image.fromarray(img.astype(np.uint8))
        img_tensor = self.transform(img_pil)   # (3, 224, 224)
        return img_tensor, cls_id


# ─── Model ────────────────────────────────────────────────────────────────────

def build_classifier(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build ResNet-34 classifier with modified final FC layer.

    num_classes: number of defect_types for this object
                 e.g. hazelnut → 4 (crack, cut, hole, print)
    """
    model       = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features   # 512 for ResNet-34
    model.fc    = nn.Linear(in_features, num_classes)
    return model


# ─── Training ─────────────────────────────────────────────────────────────────

def train_classifier(
    config: dict,
    device: torch.device,
    object_name: str,
    defect_types: List[str],
) -> Dict:
    """
    Train ResNet-34 classifier on generated images for one object,
    evaluate accuracy on real MVTec test images.

    Args:
        config       : Full config dict.
        device       : Torch device.
        object_name  : e.g. "hazelnut"
        defect_types : e.g. ["crack", "cut", "hole", "print"]

    Returns:
        results dict with epoch losses, val accuracies, per-class accuracies.
    """
    cls_cfg      = config["classification"]
    gen_dir      = config["generation"]["output_dir"]
    dataset_root = config["dataset"]["root"]
    out_dir      = Path(cls_cfg["output_dir"]) / object_name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = len(defect_types)

    print(f"\n── Classifier Training [{object_name}] ──────────────────────────")
    print(f"  Defect types  : {defect_types}")
    print(f"  Num classes   : {num_classes}")
    print(f"  Train data    : generated images ({gen_dir})")
    print(f"  Test data     : real MVTec images ({dataset_root})")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = GeneratedDefectDataset(
        generated_dir=gen_dir,
        object_name=object_name,
        defect_types=defect_types,
        split="train",
    )
    test_ds = GeneratedDefectDataset(
        generated_dir=gen_dir,
        object_name=object_name,
        defect_types=defect_types,
        split="test",
        real_dataset_root=dataset_root,
        config=config,
    )

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Test samples  : {len(test_ds)}")

    if len(train_ds) == 0:
        print(f"  [SKIP] No generated training data for {object_name}")
        return {}
    if len(test_ds) == 0:
        print(f"  [SKIP] No real test data for {object_name}")
        return {}

    train_loader = DataLoader(
        train_ds, batch_size=cls_cfg["batch_size"], shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cls_cfg["batch_size"] * 2, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_classifier(
        num_classes=num_classes,
        pretrained=cls_cfg.get("pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cls_cfg["lr"])

    # ── Training loop ─────────────────────────────────────────────────────────
    best_acc = 0.0
    results  = {
        "object_name":    object_name,
        "defect_types":   defect_types,
        "epoch":          [],
        "train_loss":     [],
        "val_acc":        [],
        "per_class_acc":  [],
    }

    for epoch in range(cls_cfg["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader,
                                  desc=f"  Epoch {epoch+1}/{cls_cfg['epochs']}",
                                  leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Evaluate on real MVTec test images
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs  = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)
        overall_acc = (all_preds == all_labels).mean() * 100

        # Per-class (per-defect_type) accuracy
        per_class = {}
        for cls_id, defect_type in enumerate(defect_types):
            cls_mask = all_labels == cls_id
            if cls_mask.sum() > 0:
                cls_acc = (all_preds[cls_mask] == all_labels[cls_mask]).mean() * 100
                per_class[defect_type] = round(float(cls_acc), 2)

        print(f"  Epoch {epoch+1:3d} | loss={train_loss:.4f} | acc={overall_acc:.2f}%")

        results["epoch"].append(epoch + 1)
        results["train_loss"].append(round(train_loss, 4))
        results["val_acc"].append(round(float(overall_acc), 2))
        results["per_class_acc"].append(per_class)

        if overall_acc > best_acc:
            best_acc = overall_acc
            torch.save(model.state_dict(),
                       str(out_dir / f"best_classifier_{object_name}.pt"))

    # ── Print final results ───────────────────────────────────────────────────
    print(f"\n  ── Classification Results [{object_name}] ──────────────────────")
    print(f"  Best overall accuracy: {best_acc:.2f}%")
    last_per_class = results["per_class_acc"][-1] if results["per_class_acc"] else {}
    for defect_type, acc in last_per_class.items():
        print(f"  {defect_type:<25}: {acc:.2f}%")

    out_json = out_dir / f"classification_results_{object_name}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_json}")

    return results