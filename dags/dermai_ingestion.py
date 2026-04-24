import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.utils import formatdate
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago


logger = logging.getLogger(__name__)

PROJECT_ROOT = os.getenv("AIRFLOW_PROJECT_ROOT", "/opt/airflow/project")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import ensure_parent, load_params, resolve_path  # noqa: E402


PARAMS = load_params()
PREPARE_PARAMS = PARAMS["prepare"]
AIRFLOW_PARAMS = PARAMS["airflow"]

# Resolve all project paths through params/config so the DAG stays portable.
METADATA_PATH = resolve_path(PREPARE_PARAMS["metadata_path"])
DATA_RAW = os.path.dirname(METADATA_PATH)
DATA_PROCESSED = resolve_path(PREPARE_PARAMS["processed_dir"])
DATA_REPORTS = resolve_path(AIRFLOW_PARAMS["reports_dir"])
BASELINE_STATS_PATH = resolve_path(AIRFLOW_PARAMS["baseline_stats_report"])
DRIFT_REPORT_PATH = resolve_path(AIRFLOW_PARAMS["drift_report"])
INGESTION_SUMMARY_PATH = resolve_path(AIRFLOW_PARAMS["ingestion_summary_report"])
RETRAINING_REQUEST_PATH = resolve_path(AIRFLOW_PARAMS["retraining_request_report"])
DRIFT_THRESHOLD = float(AIRFLOW_PARAMS["drift_threshold"])
DRIFT_TOTAL_PSI_THRESHOLD = float(AIRFLOW_PARAMS.get("drift_total_psi_threshold", 0.1))
DRIFT_JS_THRESHOLD = float(AIRFLOW_PARAMS.get("drift_js_divergence_threshold", 0.02))
DRIFT_TV_THRESHOLD = float(AIRFLOW_PARAMS.get("drift_total_variation_threshold", 0.12))
DRIFT_SAMPLE_RATIO_THRESHOLD = float(AIRFLOW_PARAMS.get("drift_sample_ratio_threshold", 0.2))

DEFAULT_ARGS = {
    "owner": "dermai",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}


def task_check_data_integrity(**ctx):
    import pandas as pd

    results = {}
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)
    results["metadata_rows"] = len(df)
    results["unique_lesions"] = df["lesion_id"].nunique()
    results["class_count"] = df["dx"].nunique()
    results["missing_values"] = int(df.isnull().sum().sum())
    results["expected_classes"] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    results["found_classes"] = sorted(df["dx"].unique().tolist())
    results["classes_match"] = results["found_classes"] == results["expected_classes"]

    folder_checks = {
        "images_part_1": resolve_path(PREPARE_PARAMS["image_dir_part1"]),
        "images_part_2": resolve_path(PREPARE_PARAMS["image_dir_part2"]),
        "test_images": resolve_path(PREPARE_PARAMS["test_images_dir"]),
    }
    for key, path in folder_checks.items():
        exists = os.path.exists(path)
        count = len(list(Path(path).glob("*.jpg"))) if exists else 0
        results[f"{key}_exists"] = exists
        results[f"{key}_count"] = count

    if not results["classes_match"]:
        raise ValueError(
            f"Class mismatch: expected {results['expected_classes']}, found {results['found_classes']}"
        )

    logger.info("Data integrity OK: %s", results)
    ctx["ti"].xcom_push(key="integrity", value=results)
    return results


def task_compute_drift(**ctx):
    import numpy as np
    import pandas as pd

    if not os.path.exists(BASELINE_STATS_PATH):
        logger.warning("No baseline found; skipping drift detection on first run")
        report = {"drift_detected": False, "reason": "no_baseline"}
        ctx["ti"].xcom_push(key="drift_detected", value=False)
        ctx["ti"].xcom_push(key="drift_report", value=report)
        return report

    with open(BASELINE_STATS_PATH, encoding="utf-8") as f:
        baseline = json.load(f)

    df = pd.read_csv(METADATA_PATH)
    baseline_dist = baseline.get("class_distribution", {})
    current_dist = df["dx"].value_counts().to_dict()
    n_baseline = sum(baseline_dist.values())
    n_current = sum(current_dist.values())

    baseline_classes = sorted(baseline_dist.keys())
    current_classes = sorted(current_dist.keys())
    missing_classes = sorted(set(baseline_classes) - set(current_classes))
    unseen_classes = sorted(set(current_classes) - set(baseline_classes))

    # Compare current metadata distribution against baseline training distribution.
    psi_scores = {}
    drift_flags = {}
    relative_shift = {}
    baseline_probs = []
    current_probs = []
    for cls in baseline_dist:
        b_pct = baseline_dist.get(cls, 0) / max(n_baseline, 1)
        c_pct = current_dist.get(cls, 0) / max(n_current, 1)
        b_pct = max(b_pct, 1e-6)
        c_pct = max(c_pct, 1e-6)
        psi = (c_pct - b_pct) * np.log(c_pct / b_pct)
        psi_scores[cls] = round(float(psi), 5)
        rel_change = abs(c_pct - b_pct) / b_pct
        relative_shift[cls] = round(float(rel_change), 5)
        drift_flags[cls] = rel_change > DRIFT_THRESHOLD
        baseline_probs.append(b_pct)
        current_probs.append(c_pct)

    baseline_arr = np.array(baseline_probs, dtype=float)
    current_arr = np.array(current_probs, dtype=float)
    mix_arr = 0.5 * (baseline_arr + current_arr)

    total_psi = float(sum(abs(v) for v in psi_scores.values()))
    total_variation = float(0.5 * np.sum(np.abs(current_arr - baseline_arr)))
    js_divergence = float(
        0.5 * np.sum(baseline_arr * np.log(baseline_arr / mix_arr))
        + 0.5 * np.sum(current_arr * np.log(current_arr / mix_arr))
    )

    sample_ratio_change = abs(n_current - n_baseline) / max(n_baseline, 1)

    # Raise a drift signal when any configured distribution-shift condition is breached.
    any_drift = any(
        [
            any(drift_flags.values()),
            total_psi > DRIFT_TOTAL_PSI_THRESHOLD,
            js_divergence > DRIFT_JS_THRESHOLD,
            total_variation > DRIFT_TV_THRESHOLD,
            sample_ratio_change > DRIFT_SAMPLE_RATIO_THRESHOLD,
            bool(missing_classes),
            bool(unseen_classes),
        ]
    )

    triggers = {
        "per_class_relative_shift": {cls: flag for cls, flag in drift_flags.items() if flag},
        "psi_threshold_breached": total_psi > DRIFT_TOTAL_PSI_THRESHOLD,
        "js_divergence_threshold_breached": js_divergence > DRIFT_JS_THRESHOLD,
        "total_variation_threshold_breached": total_variation > DRIFT_TV_THRESHOLD,
        "sample_ratio_threshold_breached": sample_ratio_change > DRIFT_SAMPLE_RATIO_THRESHOLD,
        "missing_classes": missing_classes,
        "unseen_classes": unseen_classes,
    }

    report = {
        "drift_detected": any_drift,
        "total_psi": round(total_psi, 5),
        "jensen_shannon_divergence": round(js_divergence, 5),
        "total_variation_distance": round(total_variation, 5),
        "psi_per_class": psi_scores,
        "relative_shift_per_class": relative_shift,
        "class_drift_flags": drift_flags,
        "missing_classes": missing_classes,
        "unseen_classes": unseen_classes,
        "current_dist": {k: int(v) for k, v in current_dist.items()},
        "baseline_dist": baseline_dist,
        "n_current": int(n_current),
        "n_baseline": int(n_baseline),
        "sample_ratio_change": round(sample_ratio_change, 5),
        "thresholds": {
            "per_class_relative_shift": DRIFT_THRESHOLD,
            "total_psi": DRIFT_TOTAL_PSI_THRESHOLD,
            "jensen_shannon_divergence": DRIFT_JS_THRESHOLD,
            "total_variation_distance": DRIFT_TV_THRESHOLD,
            "sample_ratio_change": DRIFT_SAMPLE_RATIO_THRESHOLD,
        },
        "triggers": triggers,
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(ensure_parent(DRIFT_REPORT_PATH), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Drift check complete: detected=%s total_psi=%.4f js=%.4f tv=%.4f sample_ratio=%.4f",
        any_drift,
        total_psi,
        js_divergence,
        total_variation,
        sample_ratio_change,
    )
    ctx["ti"].xcom_push(key="drift_detected", value=any_drift)
    ctx["ti"].xcom_push(key="drift_report", value=report)
    return report


def task_validate_splits(**ctx):
    import pandas as pd

    expected_cols = {"image_id", "image_path", "label", "lesion_id"}
    results = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(DATA_PROCESSED, f"{split}.csv")
        if not os.path.exists(path):
            results[split] = {"exists": False, "rows": 0, "valid": False}
            continue

        df = pd.read_csv(path)
        missing_cols = expected_cols - set(df.columns)
        results[split] = {
            "exists": True,
            "rows": len(df),
            "classes": df["label"].nunique() if "label" in df.columns else 0,
            "missing_cols": list(missing_cols),
            "valid": len(missing_cols) == 0,
        }

    ctx["ti"].xcom_push(key="split_validation", value=results)
    logger.info("Split validation: %s", results)
    return results


def task_save_summary(**ctx):
    ti = ctx["ti"]
    summary = {
        "run_id": ctx["run_id"],
        "execution_date": ctx["ds"],
        "integrity": ti.xcom_pull(key="integrity", task_ids="check_data_integrity"),
        "drift": ti.xcom_pull(key="drift_report", task_ids="compute_drift"),
        "splits": ti.xcom_pull(key="split_validation", task_ids="validate_splits"),
        "timestamp": datetime.utcnow().isoformat(),
    }

    with open(ensure_parent(INGESTION_SUMMARY_PATH), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Ingestion summary saved: %s", INGESTION_SUMMARY_PATH)
    return summary


def task_branch_on_drift(**ctx):
    drift_detected = ctx["ti"].xcom_pull(key="drift_detected", task_ids="compute_drift")
    return "notify_drift_detected" if drift_detected else "no_retraining_needed"


def _smtp_config() -> dict[str, str]:
    return {
        "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
        "port": os.getenv("SMTP_PORT", "587"),
        "username": os.getenv("SMTP_USERNAME") or os.getenv("ALERT_EMAIL_USER", ""),
        "password": os.getenv("SMTP_AUTH_PASSWORD") or os.getenv("ALERT_EMAIL_PASS", ""),
        "from_email": os.getenv("SMTP_FROM") or os.getenv("ALERT_EMAIL_FROM", ""),
        "to_email": os.getenv("SMTP_TO") or os.getenv("ALERT_EMAIL_TO") or os.getenv("ALERT_EMAIL_FROM", ""),
    }


def task_notify_drift_detected(**ctx):
    ti = ctx["ti"]
    drift = ti.xcom_pull(key="drift_report", task_ids="compute_drift") or {}
    integrity = ti.xcom_pull(key="integrity", task_ids="check_data_integrity") or {}
    splits = ti.xcom_pull(key="split_validation", task_ids="validate_splits") or {}

    request = {
        "run_id": ctx["run_id"],
        "execution_date": ctx["ds"],
        "timestamp": datetime.utcnow().isoformat(),
        "action_required": "Review drift and run DVC repro on a GPU-capable training environment.",
        "recommended_command": "dvc repro",
        "drift": drift,
        "integrity": integrity,
        "splits": splits,
    }
    with open(ensure_parent(RETRAINING_REQUEST_PATH), "w", encoding="utf-8") as f:
        json.dump(request, f, indent=2)

    smtp = _smtp_config()
    missing = [key for key in ("username", "password", "from_email", "to_email") if not smtp[key]]
    if missing:
        raise RuntimeError(
            "Missing SMTP configuration for drift email: "
            + ", ".join(missing)
            + ". Set them in .env before running the DAG."
        )

    body = f"""
DermAI data drift detected.

Run ID: {ctx['run_id']}
Execution date: {ctx['ds']}
Total PSI: {drift.get('total_psi')}
Jensen-Shannon divergence: {drift.get('jensen_shannon_divergence')}
Total variation distance: {drift.get('total_variation_distance')}
Sample ratio change: {drift.get('sample_ratio_change')}
Missing classes: {drift.get('missing_classes')}
Unseen classes: {drift.get('unseen_classes')}

Airflow saved the request at:
{RETRAINING_REQUEST_PATH}

This DAG intentionally does not run dvc repro on the current CPU-only machine.
Please review the request and trigger retraining on your GPU environment if needed.
""".strip()

    message = MIMEText(body, "plain", "utf-8")
    message["Subject"] = "[DermAI] Data drift detected - retraining review required"
    message["From"] = smtp["from_email"]
    message["To"] = smtp["to_email"]
    message["Date"] = formatdate(localtime=True)

    with smtplib.SMTP(smtp["host"], int(smtp["port"])) as server:
        server.starttls()
        server.login(smtp["username"], smtp["password"])
        server.send_message(message)

    logger.warning("Drift notification email sent to %s", smtp["to_email"])
    return request


with DAG(
    dag_id="dermai_data_ingestion",
    description="DermAI nightly data validation, drift detection, and retraining notification",
    default_args=DEFAULT_ARGS,
    schedule_interval=AIRFLOW_PARAMS["schedule"],
    catchup=False,
    max_active_runs=1,
    tags=["dermai", "mlops", "da5402", "drift"],
) as dag:
    check_integrity = PythonOperator(
        task_id="check_data_integrity",
        python_callable=task_check_data_integrity,
        provide_context=True,
    )

    compute_drift = PythonOperator(
        task_id="compute_drift",
        python_callable=task_compute_drift,
        provide_context=True,
    )

    validate_splits = PythonOperator(
        task_id="validate_splits",
        python_callable=task_validate_splits,
        provide_context=True,
    )

    save_summary = PythonOperator(
        task_id="save_summary",
        python_callable=task_save_summary,
        provide_context=True,
    )

    branch = BranchPythonOperator(
        task_id="branch_on_drift",
        python_callable=task_branch_on_drift,
        provide_context=True,
    )

    notify_drift = PythonOperator(
        task_id="notify_drift_detected",
        python_callable=task_notify_drift_detected,
        provide_context=True,
    )

    no_retraining = EmptyOperator(task_id="no_retraining_needed")

    check_integrity >> [compute_drift, validate_splits]
    [compute_drift, validate_splits] >> save_summary
    save_summary >> branch
    branch >> [notify_drift, no_retraining]
    