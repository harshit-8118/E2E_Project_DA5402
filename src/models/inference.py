# src/models/evaluate.py
# evaluation on test set + gradcam + mlflow logging + model registry

import os
import json
import yaml
import torch
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models.model import build_model
from src.models.train import SkinDataset, get_transforms
from src.utils.logger import get_logger
from src.utils.mlflow_utils import setup_mlflow, log_tags, log_per_class_metrics
from src.utils.metrics import (
    compute_metrics, compute_per_class_f1,
    compute_per_class_mistake_pct, compute_confusion_matrix,
    get_classification_report, CLASS_NAMES
)

logger = get_logger("evaluate")


def plot_confusion_matrix(cm: np.ndarray, classes: list, save_path: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def generate_gradcam_samples(model, dataset, label_to_idx, device, save_dir: str, n_per_class: int = 2):
    """Generate and save GradCAM heatmap samples for each class."""
    os.makedirs(save_dir, exist_ok=True)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # get target layer for efficientnet
    target_layer = model.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    # collect sample indices per class
    df = dataset.df
    for cls_name, cls_idx in label_to_idx.items():
        samples = df[df["label"] == cls_name].head(n_per_class)
        for i, (_, row) in enumerate(samples.iterrows()):
            try:
                # load original image for overlay
                orig_img = np.array(Image.open(row["image_path"]).convert("RGB").resize((224, 224)))
                orig_img = orig_img.astype(np.float32) / 255.0

                # prepare tensor
                img_tensor = dataset.transform(
                    Image.open(row["image_path"]).convert("RGB")
                ).unsqueeze(0).to(device)

                # generate cam
                targets   = [ClassifierOutputTarget(cls_idx)]
                grayscale = cam(input_tensor=img_tensor, targets=targets)
                cam_img   = show_cam_on_image(orig_img, grayscale[0], use_rgb=True)

                # save
                save_path = os.path.join(save_dir, f"{cls_name}_sample_{i+1}.png")
                Image.fromarray(cam_img).save(save_path)

            except Exception as e:
                logger.warning(f"GradCAM failed for {cls_name} sample {i}: {e}")

    logger.info(f"GradCAM samples saved to {save_dir}")


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)

    tp = p["train"]
    ep = p["evaluate"]

    device       = torch.device(tp["device"] if torch.cuda.is_available() else "cpu")
    label_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    # ── load best model ────────────────────────────────────────────────────────
    model = build_model(tp["model_name"], tp["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(ep["model_path"], map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model from {ep['model_path']}")

    # ── test dataloader ────────────────────────────────────────────────────────
    _, val_tf = get_transforms(tp["image_size"])
    test_ds   = SkinDataset("data/processed/test.csv", val_tf, label_to_idx)
    test_loader = DataLoader(test_ds, batch_size=tp["batch_size"],
                             shuffle=False, num_workers=tp["num_workers"])

    # ── inference ──────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # convert indices back to class names
    idx_to_label  = {v: k for k, v in label_to_idx.items()}
    pred_names    = [idx_to_label[p] for p in all_preds]
    true_names    = [idx_to_label[l] for l in all_labels]

    # ── compute all metrics ────────────────────────────────────────────────────
    overall       = compute_metrics(true_names, pred_names)
    per_class_f1  = compute_per_class_f1(true_names, pred_names, CLASS_NAMES)
    mistake_pct   = compute_per_class_mistake_pct(true_names, pred_names, CLASS_NAMES)
    cm            = compute_confusion_matrix(true_names, pred_names, CLASS_NAMES)
    report        = get_classification_report(true_names, pred_names, CLASS_NAMES)

    logger.info(f"\n{report}")
    logger.info(f"Test macro F1: {overall['macro_f1']}")

    # ── save metrics json — used by Prometheus later ───────────────────────────
    os.makedirs("outputs/metrics", exist_ok=True)
    eval_metrics = {
        **overall,
        "per_class_f1"        : per_class_f1,
        "per_class_mistake_pct": mistake_pct,
    }
    with open("outputs/metrics/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    # ── confusion matrix plot ──────────────────────────────────────────────────
    os.makedirs("outputs/plots", exist_ok=True)
    plot_confusion_matrix(cm, CLASS_NAMES, "outputs/plots/confusion_matrix.png")

    # ── gradcam samples ────────────────────────────────────────────────────────
    generate_gradcam_samples(
        model, test_ds, label_to_idx, device,
        save_dir="outputs/plots/gradcam"
    )

    # ── mlflow logging ─────────────────────────────────────────────────────────
    setup_mlflow("skin-disease-detection")

    # resume the same run from train.py
    run_id_path = "outputs/metrics/mlflow_run_id.txt"
    run_id      = open(run_id_path).read().strip() if os.path.exists(run_id_path) else None

    with mlflow.start_run(run_id=run_id):
        log_tags(tp["model_name"], stage="evaluation")

        # overall metrics
        mlflow.log_metrics({
            "test_accuracy"    : overall["accuracy"],
            "test_macro_f1"    : overall["macro_f1"],
            "test_weighted_f1" : overall["weighted_f1"],
            "test_macro_precision": overall["macro_precision"],
            "test_macro_recall": overall["macro_recall"],
        })

        # per class f1
        log_per_class_metrics(per_class_f1,  prefix="test_f1")

        # per class mistake %
        log_per_class_metrics(mistake_pct, prefix="test_mistake_pct")

        # artifacts
        mlflow.log_artifact("outputs/plots/confusion_matrix.png")
        mlflow.log_artifact("outputs/metrics/eval_metrics.json")
        mlflow.log_artifacts("outputs/plots/gradcam", artifact_path="gradcam")

        # ── register model if acceptance criterion met ─────────────────────────
        if overall["macro_f1"] >= ep["acceptance_macro_f1"]:
            logger.info(f"F1 {overall['macro_f1']} >= {ep['acceptance_macro_f1']} — registering model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

            # log model properly for registry
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name="skin-disease-classifier"
            )
            logger.info("Model registered: skin-disease-classifier")
        else:
            logger.warning(
                f"F1 {overall['macro_f1']} < {ep['acceptance_macro_f1']} — model NOT registered"
            )

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()