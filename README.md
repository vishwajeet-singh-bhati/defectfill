# DefectFill on Steel Pipe Weld Defect Dataset

Implementation of **DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection** (CVPR 2025) adapted for the Steel Pipe Weld Defect dataset.

---

## Dataset Overview

**Source:** https://github.com/huangyebiaoke/steel-pipe-weld-defect-detection  
**Paper:** "Deep Learning Based Steel Pipe Weld Defect Detection" (Yang et al., 2021)

### What the Dataset Contains
X-ray radiographic images of steel pipe weld seams. X-rays pass through the weld and are captured by a flat-panel detector, revealing internal structural defects invisible to the naked eye. Images are **grayscale** (single-channel), typically around 800×800 pixels.

### Annotation Format
The dataset ships in **two formats** simultaneously:
- **YOLO format** (`.txt` files with normalized `class cx cy w h` per line)
- **PASCAL VOC 2007 format** (`.xml` files with `<xmin><ymin><xmax><ymax>` bounding boxes)

Our pipeline uses YOLO format for simplicity; bounding boxes are converted to **binary masks** for DefectFill.

### 8 Defect Classes

| Label | English Name     | Chinese | Count | Severity |
|-------|-----------------|---------|-------|----------|
| 0     | air-hole        | 气孔    | 5,191 | Small circular pores, abundant but tiny |
| 1     | bite-edge       | 咬边    | 35    | **Very scarce** — groove cut into base metal |
| 2     | broken-arc      | 断弧    | 458   | Discontinuity in weld arc |
| 3     | crack           | 裂缝    | 119   | Linear fractures, narrow morphology |
| 4     | hollow-bead     | 夹珠    | 229   | Hollow spherical inclusions |
| 5     | overlap         | 焊瘤    | 223   | Metal overflow on weld surface |
| 6     | slag-inclusion  | 夹渣    | 120   | Non-metallic slag trapped in weld |
| 7     | unfused         | 未融合  | 408   | Lack of fusion between layers |

**Total annotations:** ~6,783  
**Class imbalance:** air-hole (5191) vs bite-edge (35) — 148× imbalance. DefectFill is specifically designed to address this.

### Why DefectFill is Perfect for This Dataset
- `bite-edge` has only **35 samples** — almost impossible to train a classifier
- `crack` and `slag-inclusion` have <120 samples
- DefectFill needs only **a few reference images** (uses 1/3 of available samples to learn, generates for the other 2/3)
- The inpainting approach naturally handles the elongated, irregular shapes of weld defects

---

## Project Structure

```
defectfill_weld/
│
├── README.md                   ← This file
├── requirements.txt            ← All Python dependencies
├── configs/
│   └── config.yaml             ← Central config (paths, hyperparameters, class names)
│
├── data/
│   ├── dataset.py              ← PyTorch Dataset; loads images + converts bbox→mask
│   ├── preprocess.py           ← Splits data into reference/target sets; prepares pairs
│   └── augment.py              ← Augmentation helpers (flip, contrast for X-ray images)
│
├── models/
│   ├── defectfill.py           ← Core DefectFill: wraps SD-inpainting + LoRA fine-tuning
│   └── lfs.py                  ← Low-Fidelity Selection (LPIPS-based quality filter)
│
├── losses/
│   └── defectfill_loss.py      ← Three custom losses: L_def, L_obj, L_attn
│
├── inspect/
│   ├── classifier.py           ← ResNet-34 downstream classification model
│   └── localizer.py            ← UNet downstream defect localization model
│
├── utils/
│   ├── metrics.py              ← KID, IC-LPIPS, AUROC, AP, F1, PRO
│   └── visualization.py        ← Plot generated defects, attention maps, metrics
│
├── scripts/
│   ├── download_dataset.sh     ← One-command dataset download + unzip
│   ├── prepare_data.sh         ← Calls preprocess.py to set up reference/target splits
│   └── run_all.sh              ← End-to-end pipeline script
│
├── train.py                    ← Fine-tune SD-inpainting with DefectFill losses
├── generate.py                 ← Generate defect images using trained model
├── evaluate.py                 ← Evaluate generation quality + downstream tasks
│
└── outputs/
    ├── generated/              ← Generated defect images (created at runtime)
    ├── checkpoints/            ← Saved LoRA weights per defect class
    └── results/                ← Evaluation metrics, plots
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
bash scripts/download_dataset.sh

# 3. Prepare reference/target splits
bash scripts/prepare_data.sh

# 4. Train DefectFill for all defect classes
python train.py --config configs/config.yaml

# 5. Generate defect images
python generate.py --config configs/config.yaml --num_samples 1000

# 6. Evaluate (generation quality + downstream tasks)
python evaluate.py --config configs/config.yaml
```

---

## Key Adaptations from the Original Paper

| Aspect | MVTec AD (Paper) | This Implementation |
|--------|-----------------|---------------------|
| Image type | RGB texture photos | Grayscale X-ray → converted to 3-channel |
| Annotations | Segmentation masks | Bounding boxes → binary masks |
| Object prompt | "A hazelnut with [V*]" | "An X-ray of a steel weld with [V*]" |
| Defect prompt | "A photo of [V*]" | "An X-ray showing [V*] defect" |
| Downstream | ResNet-34 + UNet | Same (ResNet-34 + UNet) |
| Base model | SD-2-inpainting | SD-2-inpainting (same) |
