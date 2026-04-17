import os
import json
import yaml
import torch
import mlflow
import argparse
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
from src.utils.reproducibility import set_seed
import warnings
warnings.filterwarnings("ignore")

logger = get_logger("evaluate")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.float()  
        return self.model(x)

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

def get_target_layer(model, model_name: str):
    if model_name == "efficientnet_b3":
        return model.features[-1]
    elif model_name == "convnext_small":
        return model.features[-1]
    elif model_name == "resnet50":
        return model.layer4[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def generate_gradcam_samples(model, dataset, label_to_idx, device, save_dir: str, n_per_class: int = 2, tp: dict = None):
    """Generate and save GradCAM heatmap samples for each class."""
    os.makedirs(save_dir, exist_ok=True)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # get target layer for efficientnet
    target_layer = get_target_layer(model, tp["model_name"])
    cam = GradCAM(model=model, target_layers=[target_layer])

    img_size = tp['image_size']
    # collect sample indices per class
    df = dataset.df
    for cls_name, cls_idx in label_to_idx.items():
        samples = df[df["label"] == cls_name].head(n_per_class)
        for i, (_, row) in enumerate(samples.iterrows()):
            try:
                # load original image for overlay
                orig_img = np.array(Image.open(row["image_path"]).convert("RGB").resize((img_size, img_size)))
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


def parse_arguments(tp, ep):
    parser = argparse.ArgumentParser(description="Train skin disease detection model")

    parser.add_argument("--model_name", type=str, default=tp["model_name"])
    parser.add_argument("--epochs", type=int, default=int(tp["epochs"]))
    parser.add_argument("--batch_size", type=int, default=int(tp["batch_size"]))
    parser.add_argument("--learning_rate", type=float, default=float(tp["learning_rate"]))
    parser.add_argument("--image_size", type=int, default=int(tp["image_size"]))
    
    parser.add_argument("--mixup_alpha", type=float, default=float(tp["mixup_alpha"]))
    parser.add_argument("--cutmix_alpha", type=float, default=float(tp["cutmix_alpha"]))
    parser.add_argument("--label_smoothing", type=float, default=float(tp["label_smoothing"]))
    parser.add_argument("--weight_decay", type=float, default=float(tp["weight_decay"]))
    parser.add_argument("--random_seed", type=int, default=int(tp["random_seed"]))
    
    parser.add_argument("--early_stopping_patience", type=int, default=int(tp["early_stopping_patience"]))
    parser.add_argument("--scheduler", type=str, default=str(tp["scheduler"]))

    parser.add_argument("--use_weighted_loss", type=str, default=str(tp["use_weighted_loss"]))
    parser.add_argument("--use_amp", type=str, default=str(tp["use_amp"]))

    parser.add_argument("--model_path", type=str, default=str(ep["model_path"]))
    parser.add_argument("--acceptance_macro_f1", type=float, default=float(ep["acceptance_macro_f1"]))

    return parser.parse_args()


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)

    tp = p["train"]
    ep = p["evaluate"]

    args = parse_arguments(tp, ep)

    if args.model_name is not None: tp["model_name"] = args.model_name
    if args.epochs is not None: tp["epochs"] = args.epochs
    if args.batch_size is not None: tp["batch_size"] = args.batch_size
    if args.learning_rate is not None: tp["learning_rate"] = args.learning_rate
    if args.image_size is not None: tp["image_size"] = args.image_size
    if args.random_seed is not None: tp["random_seed"] = args.random_seed
    if args.use_weighted_loss is not None: tp["use_weighted_loss"] = str(args.use_weighted_loss).lower() == "true"
    if args.use_amp is not None: tp["use_amp"] = str(args.use_amp).lower() == "true"
    if args.weight_decay is not None: tp["weight_decay"] = args.weight_decay
    if args.mixup_alpha is not None: tp["mixup_alpha"] = args.mixup_alpha
    if args.cutmix_alpha is not None: tp["cutmix_alpha"] = args.cutmix_alpha
    if args.label_smoothing is not None: tp["label_smoothing"] = args.label_smoothing
    if args.early_stopping_patience is not None: tp["early_stopping_patience"] = args.early_stopping_patience
    if args.scheduler is not None: tp["scheduler"] = args.scheduler
    if args.model_path is not None: ep["model_path"] = args.model_path
    if args.acceptance_macro_f1 is not None: ep["acceptance_macro_f1"] = args.acceptance_macro_f1

    set_seed(tp["random_seed"])

    device       = torch.device(tp["device"] if torch.cuda.is_available() else "cpu")
    label_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}

    # ── load best model ────────────────────────────────────────────────────────
    model = build_model(tp["model_name"], tp["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(ep["model_path"], map_location=device, weights_only=True))
    model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory before: {allocated_before:.2f} GB")

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
        save_dir="outputs/plots/gradcam", tp=tp
    )

    # ── mlflow logging ─────────────────────────────────────────────────────────

    env_run_id  = os.environ.get("MLFLOW_RUN_ID", None)
    file_run_id = None
    run_id_path = "outputs/metrics/mlflow_run_id.txt"
    if os.path.exists(run_id_path):
        file_run_id = open(run_id_path).read().strip()

    # env var takes priority (set by mlflow run), then saved file
    active_run_id = env_run_id or file_run_id

    if active_run_id == env_run_id and env_run_id is not None:
        print(f"Resuming MLflow Run: {active_run_id}")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow"))
    else:
        print(f"Starting a new MLflow Run")
        setup_mlflow("skin-disease-detection")

    with mlflow.start_run(run_id=active_run_id):
        log_tags(tp["model_name"], stage="evaluation")

        # ── overall metrics — all of them ──────────────────────────────────────
        mlflow.log_metrics({
            "test_accuracy"          : overall["accuracy"],
            "test_macro_f1"          : overall["macro_f1"],
            "test_micro_f1"          : overall["micro_f1"],
            "test_weighted_f1"       : overall["weighted_f1"],
            "test_macro_precision"   : overall["macro_precision"],
            "test_micro_precision"   : overall["micro_precision"],
            "test_weighted_precision": overall["weighted_precision"],
            "test_macro_recall"      : overall["macro_recall"],
            "test_micro_recall"      : overall["micro_recall"],
            "test_weighted_recall"   : overall["weighted_recall"],
        })

        # per class f1
        log_per_class_metrics(per_class_f1, prefix="test_f1")

        # per class mistake %
        log_per_class_metrics(mistake_pct, prefix="test_mistake_pct")

        # artifacts
        mlflow.log_artifact("outputs/plots/confusion_matrix.png")
        mlflow.log_artifact("outputs/metrics/eval_metrics.json")
        mlflow.log_artifacts("outputs/plots/gradcam", artifact_path="gradcam")

        # log classification report as text artifact
        report_path = "outputs/metrics/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        wrapped_model = ModelWrapper(model)
        if overall["macro_f1"] >= ep["acceptance_macro_f1"]:
            logger.info(f"F1 {overall['macro_f1']} >= {ep['acceptance_macro_f1']} — registering model")

            # 'artifact_path' was renamed to 'artifact_path' in generic log_model, 
            mlflow.pytorch.log_model(
                pytorch_model=wrapped_model,
                artifact_path="model",
                registered_model_name="skin-disease-classifier",
                code_paths=[__file__],
                pip_requirements=[
                    "torch", #  replace for cuda: f"torch=={torch.__version__}" for cpu: "torch",
                    "torchvision",
                    "Pillow",
                    "numpy",
                ]
            )

            # Transition to production
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            # 1. Search for versions correctly
            versions = client.search_model_versions("name='skin-disease-classifier'")
            if versions:
                latest_version = sorted(versions, key=lambda v: int(v.version))[-1].version
                client.set_registered_model_alias(
                    name="skin-disease-classifier",
                    alias="production",
                    version=latest_version
                )
                logger.info(f"Model v{latest_version} tagged as 'production' alias")
        else:
            logger.warning(
                f"F1 {overall['macro_f1']} < {ep['acceptance_macro_f1']} — model NOT registered"
            )

    logger.info("Evaluation complete")


if __name__ == "__main__":
    main()