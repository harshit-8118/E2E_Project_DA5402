# src/utils/metrics.py
# metric computation helpers used by train.py and evaluate.py

import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report
)


CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def compute_metrics(y_true: list, y_pred: list) -> dict:
    return {
        "accuracy"           : round(accuracy_score(y_true, y_pred), 4),
        "macro_f1"           : round(f1_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "micro_f1"           : round(f1_score(y_true, y_pred, average="micro",    zero_division=0), 4),
        "weighted_f1"        : round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "macro_precision"    : round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "micro_precision"    : round(precision_score(y_true, y_pred, average="micro",    zero_division=0), 4),
        "weighted_precision" : round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "macro_recall"       : round(recall_score(y_true, y_pred, average="macro",    zero_division=0), 4),
        "micro_recall"       : round(recall_score(y_true, y_pred, average="micro",    zero_division=0), 4),
        "weighted_recall"    : round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }

def compute_per_class_f1(y_true: list, y_pred: list, classes: list) -> dict:
    """Per class F1 score — logged individually to MLflow."""
    scores = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {cls: round(float(scores[i]), 4) for i, cls in enumerate(classes)}


def compute_per_class_mistake_pct(y_true: list, y_pred: list, classes: list) -> dict:
    """
    Per class mistake percentage — used later by Prometheus/Grafana.
    mistake% = wrong predictions / total true samples for that class
    """
    result = {}
    for cls in classes:
        indices      = [i for i, t in enumerate(y_true) if t == cls]
        if not indices:
            result[cls] = 0.0
            continue
        wrong        = sum(1 for i in indices if y_pred[i] != cls)
        result[cls]  = round(wrong / len(indices) * 100, 2)
    return result


def compute_confusion_matrix(y_true: list, y_pred: list, classes: list) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=classes)


def get_classification_report(y_true: list, y_pred: list, classes: list) -> str:
    return classification_report(y_true, y_pred, target_names=classes, zero_division=0)