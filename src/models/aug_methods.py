import torch 
import numpy as np


def mixup_data(images: torch.Tensor, labels: torch.Tensor, alpha: float, device):
    """
    Mixup — blends two images and their labels.
    alpha controls interpolation strength. Higher = more mixing.
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)

    # random permutation to get mixing pairs
    idx = torch.randperm(batch_size).to(device)

    mixed_images = lam * images + (1 - lam) * images[idx]
    labels_a     = labels
    labels_b     = labels[idx]

    return mixed_images, labels_a, labels_b, lam


def cutmix_data(images: torch.Tensor, labels: torch.Tensor, alpha: float, device):
    """
    CutMix — cuts a patch from one image and pastes into another.
    More aggressive than mixup — forces model to use full image context.
    """
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = images.size()

    idx = torch.randperm(batch_size).to(device)

    # compute cut box dimensions
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h     = int(H * cut_ratio)
    cut_w     = int(W * cut_ratio)

    # random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # clamp box to image boundaries
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    # paste patch
    mixed_images          = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]

    # recalculate lam based on actual box size
    lam      = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    labels_a = labels
    labels_b = labels[idx]

    return mixed_images, labels_a, labels_b, lam


def mixup_cutmix_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Mixed loss — weighted combination of two label losses."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)