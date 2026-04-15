# src/models/train.py
# training loop with mlflow logging, early stopping, weighted loss

import os
import json
import yaml
import torch
import torch.nn as nn
import random
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import argparse

from src.models.model import build_model, load_class_weights
from src.models.aug_methods import mixup_data, cutmix_data, mixup_cutmix_criterion
from src.utils.logger import get_logger
from src.utils.mlflow_utils import setup_mlflow, log_tags, log_params_from_dict, log_per_class_metrics
from src.utils.metrics import compute_metrics, compute_per_class_f1, CLASS_NAMES
from src.utils.reproducibility import set_seed

import warnings
warnings.filterwarnings("ignore")

logger = get_logger("train")

def seed_worker(worker_id):
    # each dataloader worker needs its own seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# dataset
class SkinDataset(Dataset):
    def __init__(self, csv_path: str, transform, label_to_idx: dict):
        self.df          = pd.read_csv(csv_path)
        self.transform   = transform
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row["label"]]
        return image, label


# transforms
def get_transforms(image_size: int):
    # HAM10000 mean and std (precomputed)
    mean = [0.7635, 0.5461, 0.5705]
    std  = [0.1409, 0.1526, 0.1692]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, val_tf


# early stopping
class EarlyStopping:
    def __init__(self, patience: int, path: str):
        self.patience  = patience
        self.path      = path
        self.counter   = 0
        self.best_f1   = 0.0
        self.triggered = False

    def __call__(self, val_f1: float, model: nn.Module) -> bool:
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            logger.info(f"Best model saved — val_macro_f1: {val_f1:.4f}")
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# one epoch
def run_epoch(model, loader, criterion, optimizer, device, is_train: bool, scaler: GradScaler, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            use_mixup  = is_train and mixup_alpha > 0 and np.random.rand() < 0.3
            use_cutmix = is_train and cutmix_alpha > 0 and not use_mixup

            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha, device)
            elif use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha, device)

            # mixed precision forward pass
            with autocast(device_type="cuda", enabled=(scaler is not None)):
                outputs = model(images)
                if use_cutmix or use_mixup:
                    loss = mixup_cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader.dataset), all_preds, all_labels

def parse_arguments(tp):
    parser = argparse.ArgumentParser(description="Train skin disease detection model")

    parser.add_argument("--model_name", type=str, default=tp["model_name"])
    parser.add_argument("--epochs", type=int, default=int(tp["epochs"]))
    parser.add_argument("--batch_size", type=int, default=int(tp["batch_size"]))
    parser.add_argument("--learning_rate", type=float, default=float(tp["learning_rate"]))
    parser.add_argument("--image_size", type=int, default=int(tp["image_size"]))
    parser.add_argument("--random_seed", type=int, default=int(tp["random_seed"]))
    
    parser.add_argument("--mixup_alpha", type=float, default=float(tp["mixup_alpha"]))
    parser.add_argument("--cutmix_alpha", type=float, default=float(tp["cutmix_alpha"]))
    parser.add_argument("--label_smoothing", type=float, default=float(tp["label_smoothing"]))
    parser.add_argument("--weight_decay", type=float, default=float(tp["weight_decay"]))
    
    parser.add_argument("--early_stopping_patience", type=int, default=int(tp["early_stopping_patience"]))
    parser.add_argument("--scheduler", type=str, default=str(tp["scheduler"]))

    parser.add_argument("--use_weighted_loss", type=str, default=str(tp["use_weighted_loss"]))
    parser.add_argument("--use_amp", type=str, default=str(tp["use_amp"]))

    return parser.parse_args()

# main
def main():
    # load params
    with open("params.yaml") as f:
        p = yaml.safe_load(f)

    tp = p["train"]
    pp = p["prepare"]

    args = parse_arguments(tp)

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

    set_seed(tp["random_seed"])

    device = torch.device(tp["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    label_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    idx_to_label = {i: cls for cls, i in label_to_idx.items()}

    # data
    train_tf, val_tf = get_transforms(tp["image_size"])

    train_ds = SkinDataset("data/processed/train.csv", train_tf, label_to_idx)
    val_ds   = SkinDataset("data/processed/val.csv",   val_tf,   label_to_idx)

    g = torch.Generator()
    g.manual_seed(tp["random_seed"])

    train_loader = DataLoader(train_ds, batch_size=tp["batch_size"],
                              shuffle=True,  num_workers=tp["num_workers"], pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=tp["batch_size"],
                              shuffle=False, num_workers=tp["num_workers"], pin_memory=True, worker_init_fn=seed_worker, generator=g)

    # model
    model = build_model(tp["model_name"], tp["num_classes"], tp["pretrained"])
    model = model.to(device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    allocated_before = torch.cuda.memory_allocated() / 1024**3
    print(f"Before training: {allocated_before:.2f} GB allocated")

    # weighted loss — handles class imbalance
    class_weights = load_class_weights("data/reports/baseline_stats.json", CLASS_NAMES, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights if tp["use_weighted_loss"] else None, label_smoothing=tp["label_smoothing"])

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=tp["learning_rate"],
                                  weight_decay=tp["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tp["epochs"]
    ) if tp["scheduler"] == "cosine" else None

    os.makedirs("outputs/models", exist_ok=True)
    early_stop = EarlyStopping(
        patience=tp["early_stopping_patience"],
        path="outputs/models/best_model.pth"
    )

    # mlflow run
    scaler = GradScaler(device="cuda", enabled=tp["use_amp"])

    active_run_id = os.environ.get("MLFLOW_RUN_ID", None)
    if active_run_id:
        print('-'*30)
        print(f"mlflow run injected run id: {active_run_id}")
        print('-'*30)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/da25s003/E2E_Project_DA5402.mlflow"))
    else:
        setup_mlflow("skin-disease-detection")

    with mlflow.start_run(run_id=active_run_id) as run:
        logger.info(f"MLflow run ID: {run.info.run_id}")
        mlflow.log_param("random_seed", tp["random_seed"])
        mlflow.set_tag("random_seed", tp["random_seed"])
        # tags
        log_tags(tp["model_name"], stage="training")

        # params
        log_params_from_dict({**tp, "val_size": pp["val_size"], "prepare_random_seed": pp["random_seed"]})

        best_epoch   = 0
        best_val_f1  = 0.0
        train_history = []

        # training loop
        for epoch in range(1, tp["epochs"] + 1):
            train_loss, train_preds, train_labels = run_epoch(
                model, train_loader, criterion, optimizer, device, is_train=True, scaler=scaler, mixup_alpha=tp["mixup_alpha"],cutmix_alpha=tp["cutmix_alpha"]
            )
            val_loss, val_preds, val_labels = run_epoch(
                model, val_loader, criterion, optimizer, device, is_train=False, scaler=scaler, mixup_alpha=0.0, cutmix_alpha=0.0
            )

            if scheduler:
                scheduler.step()

            # compute metrics
            train_m    = compute_metrics(train_labels, train_preds)
            val_m      = compute_metrics(val_labels, val_preds)
            val_f1_cls = compute_per_class_f1(val_labels, val_preds,
                                              list(range(len(CLASS_NAMES))))

            current_lr = optimizer.param_groups[0]["lr"]

            # log to mlflow per epoch
            mlflow.log_metrics({
                "train_loss"      : round(train_loss, 4),
                "train_acc"       : train_m["accuracy"],
                "train_macro_f1"  : train_m["macro_f1"],
                "train_micro_f1"  : train_m["micro_f1"],
                "train_weighted_f1" : train_m["weighted_f1"],
                'train_weighted_precision': train_m["weighted_precision"],
                'train_macro_precision': train_m["macro_precision"],
                'train_micro_precision': train_m["micro_precision"],
                'train_weighted_recall': train_m["weighted_recall"],
                'train_macro_recall': train_m["macro_recall"],
                'train_micro_recall': train_m["micro_recall"],
                "val_loss"        : round(val_loss, 4),
                "val_acc"         : val_m["accuracy"],
                "val_macro_f1"    : val_m["macro_f1"],
                "val_micro_f1"    : val_m["micro_f1"],
                "val_weighted_f1" : val_m["weighted_f1"],
                "val_weighted_precision": val_m["weighted_precision"],
                "val_macro_precision": val_m["macro_precision"],
                "val_micro_precision": val_m["micro_precision"],
                "val_weighted_recall": val_m["weighted_recall"],
                "val_macro_recall": val_m["macro_recall"],
                "val_micro_recall": val_m["micro_recall"],
                "learning_rate"   : current_lr,
            }, step=epoch)

            # per class f1 per epoch
            log_per_class_metrics(
                {CLASS_NAMES[i]: v for i, v in val_f1_cls.items()},
                prefix="val_f1", step=epoch
            )

            logger.info(
                f"Epoch {epoch:02d} | "
                f"train_loss={train_loss:.4f} train_f1={train_m['macro_f1']:.4f} | "
                f"val_loss={val_loss:.4f} val_f1={val_m['macro_f1']:.4f}"
            )

            train_history.append({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_macro_f1": val_m["macro_f1"]
            })

            # early stopping + best model save
            if early_stop(val_m["macro_f1"], model):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            if val_m["macro_f1"] > best_val_f1:
                best_val_f1 = val_m["macro_f1"]
                best_epoch  = epoch

        # log final summary metrics
        mlflow.log_metrics({
            "best_val_macro_f1": best_val_f1,
            "best_epoch"       : best_epoch,
        })

        # save and log training history
        os.makedirs("outputs/metrics", exist_ok=True)
        history_path = "outputs/metrics/train_history.json"
        with open(history_path, "w") as f:
            json.dump(train_history, f, indent=2)
        mlflow.log_artifact(history_path)

        # log best model as artifact
        mlflow.log_artifact("outputs/models/best_model.pth")

        # save run_id for evaluate.py to pick up
        with open("outputs/metrics/mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)

        logger.info(f"Training complete | best_val_macro_f1={best_val_f1:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()