# src/models/model.py
# model architecture — EfficientNetB3 with custom classification head

import torch
import torch.nn as nn
from torchvision import models
from src.utils.logger import get_logger

logger = get_logger("model")


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Build model from torchvision.
    Supported: efficientnet_b3, convnext_small, resnet50
    """
    logger.info(f"Building model: {model_name} | pretrained={pretrained} | classes={num_classes}")

    weights = "DEFAULT" if pretrained else None

    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name == "convnext_small":
        model = models.convnext_small(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


def load_class_weights(baseline_path: str, class_names: list, device: torch.device) -> torch.Tensor:
    """Load precomputed class weights from baseline_stats.json."""
    import json
    with open(baseline_path) as f:
        baseline = json.load(f)

    weights = [baseline["class_weights"][cls] for cls in class_names]
    tensor  = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights loaded: { {c: round(w,3) for c,w in zip(class_names, weights)} }")
    return tensor