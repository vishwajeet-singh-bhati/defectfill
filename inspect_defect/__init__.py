"""
inspect_defect/
────────────────
Downstream visual inspection models trained on DefectFill-generated data.
  classifier.py — ResNet-34 defect type classifier (8 classes)
  localizer.py  — UNet pixel-level defect localiser (binary segmentation)
"""
# FIX: package renamed from 'inspect' to 'inspect_defect' to avoid shadowing
# Python's standard library 'inspect' module, which torch and numpy depend on.
from .classifier import (
    GeneratedDefectDataset,
    build_classifier,
    train_classifier,
)
from .localizer import (
    FocalLoss,
    ConvBlock,
    UpBlock,
    WeldUNet,
    LocalisationDataset,
    train_localizer,
)
__all__ = [
    # classifier
    "GeneratedDefectDataset", "build_classifier", "train_classifier",
    # localizer
    "FocalLoss", "ConvBlock", "UpBlock", "WeldUNet",
    "LocalisationDataset", "train_localizer",
]