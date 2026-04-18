# src/data/prepare.py
# splits HAM10000 into train/val by lesion_id (stratified), test set used as-is from ISIC

import os
import json
import argparse
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger

logger = get_logger("prepare")


def load_params() -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["prepare"]


def resolve_image_path(image_id: str, part1_dir: str, part2_dir: str) -> str | None:
    for folder in [part1_dir, part2_dir]:
        path = os.path.join(folder, image_id + ".jpg")
        if os.path.exists(path):
            return path
    return None


def split_by_lesion(df: pd.DataFrame, val_size: float, seed: int):
    """
    Stratified split at lesion level — prevents same lesion leaking into both splits.
    Each lesion gets one representative label (mode of label across its images).
    """
    lesion_labels = (
        df.groupby("lesion_id")["label"]        # ← was "dx", now "label"
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )

    train_lesions, val_lesions = train_test_split(
        lesion_labels["lesion_id"],
        test_size=val_size,
        stratify=lesion_labels["label"],        # ← was "label" already, fine
        random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_lesions)].copy()
    val_df   = df[df["lesion_id"].isin(val_lesions)].copy()

    return train_df, val_df

def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame):
    overlap = set(train_df["lesion_id"]) & set(val_df["lesion_id"])
    if overlap:
        raise ValueError(f"Lesion leakage detected — {len(overlap)} lesions in both splits")
    logger.info("Leakage check passed — no lesion overlap between train and val")


def class_counts(df: pd.DataFrame) -> dict:
    return df["label"].value_counts().sort_index().to_dict()


def compute_class_weights(counts: dict) -> dict:
    # inverse frequency weighting — passed directly to CrossEntropyLoss in train.py
    total = sum(counts.values())
    n_classes = len(counts)
    return {
        cls: round(total / (n_classes * count), 4)
        for cls, count in counts.items()
    }


def parse_arguments(p):
    parser = argparse.ArgumentParser(description="Train skin disease detection model")

    parser.add_argument("--val_size", type=float, default=p["val_size"])
    parser.add_argument("--random_seed", type=int, default=int(p["random_seed"]))
    
    return parser.parse_args()

def main():
    params = load_params()
    logger.info("Starting prepare stage")

    args = parse_arguments(params)
    if args.val_size is not None: params["val_size"] = args.val_size
    if args.random_seed is not None: params["random_seed"] = args.random_seed

    # ── load train metadata ────────────────────────────────────────────────────
    df = pd.read_csv(params["metadata_path"])
    df = df.rename(columns={"dx": "label"})
    logger.info(f"Loaded train metadata: {len(df)} rows, {df['lesion_id'].nunique()} unique lesions")

    # ── resolve image paths ────────────────────────────────────────────────────
    df["image_path"] = df["image_id"].apply(
        lambda x: resolve_image_path(x, params["image_dir_part1"], params["image_dir_part2"])
    )

    unresolved = df["image_path"].isnull().sum()
    if unresolved > 0:
        logger.warning(f"{unresolved} image paths unresolved — dropping those rows")
        df = df.dropna(subset=["image_path"])


    # ── stratified train / val split by lesion_id ──────────────────────────────
    train_df, val_df = split_by_lesion(df, params["val_size"], params["random_seed"])
    verify_no_leakage(train_df, val_df)
    logger.info(f"Train: {len(train_df)} images | Val: {len(val_df)} images")

    # ── load test set as-is (ISIC 2018 holdout) ───────────────────────────────
    # ── load test set as-is ────────────────────────────────────────────────────
    df_test = pd.read_csv(params["test_metadata_path"])

    # drop images missing from disk
    test_imgs_on_disk = set(
        f.replace(".jpg", "")
        for f in os.listdir(params["test_images_dir"])
        if f.endswith(".jpg")
    )
    known_missing = set(df_test["image_id"]) - test_imgs_on_disk
    if known_missing:
        logger.warning(f"Dropping {len(known_missing)} test images not found on disk: {known_missing}")
        df_test = df_test[~df_test["image_id"].isin(known_missing)]

    # dx column already has label strings — same format as train
    df_test = df_test.rename(columns={"dx": "label"})
    df_test["image_path"] = df_test["image_id"].apply(
        lambda x: os.path.join(params["test_images_dir"], x + ".jpg")
    )

    logger.info(f"Test: {len(df_test)} images | Classes: {df_test['label'].unique()}")

    # ── keep consistent columns across all splits ──────────────────────────────
    cols = ["image_id", "image_path", "label", "lesion_id"]
    train_df = train_df[cols].reset_index(drop=True)
    val_df   = val_df[cols].reset_index(drop=True)
    test_df  = df_test[cols].reset_index(drop=True)

    # ── save splits ────────────────────────────────────────────────────────────
    os.makedirs(params["processed_dir"], exist_ok=True)
    train_df.to_csv(os.path.join(params["processed_dir"], params["processed_train_csv"]), index=False)
    val_df.to_csv(os.path.join(params["processed_dir"], params["processed_val_csv"]),     index=False)
    test_df.to_csv(os.path.join(params["processed_dir"], params["processed_test_csv"]),   index=False)
    logger.info("Saved splits to data/processed/")

    # ── prepare summary — tracked as DVC metric ────────────────────────────────
    train_counts = class_counts(train_df)
    summary = {
        "train_size"         : len(train_df),
        "val_size"           : len(val_df),
        "test_size"          : len(test_df),
        "train_distribution" : train_counts,
        "val_distribution"   : class_counts(val_df),
        "test_distribution"  : class_counts(test_df),
        "unique_train_lesions": train_df["lesion_id"].nunique(),
        "unique_val_lesions"  : val_df["lesion_id"].nunique(),
    }

    os.makedirs(params["reports_dir"], exist_ok=True)
    with open(params["prepare_summary_report"], "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Train dist : {train_counts}")

    # ── baseline stats — used later by Prometheus for drift detection ──────────
    baseline = {
        "class_distribution": train_counts,
        "total_train_samples": len(train_df),
        "class_weights" : compute_class_weights(train_counts),
    }

    baseline_stats_path = params["baseline_stats_report"]
    with open(baseline_stats_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Saved {baseline_stats_path}")


if __name__ == "__main__":
    main()