"""
prediction.py - Priority scoring and prediction service.
Handles single-task and batch prediction flows.
"""

import os
import io
import logging
import joblib
import pandas as pd
from datetime import date

from preprocessing import (
    prepare_single_input,
    clean_dataframe,
    get_feature_columns,
    calculate_priority_score,
    calculate_deadline_points,
    score_to_label,
    build_explanation,
    check_required_columns,
    TASK_TYPES,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/priority_model.pkl")

_model_cache = None


def load_model():
    """Load and cache the trained Decision Tree model."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. "
                "Run train_model.py first."
            )
        _model_cache = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model_cache


def reload_model():
    """Force reload the model (call after retraining)."""
    global _model_cache
    _model_cache = None
    return load_model()


# ---------------------------------------------------------------------------
# Single task prediction
# ---------------------------------------------------------------------------

def predict_single(deadline_date, estimated_effort: float, business_impact: int,
                   urgency: int, dependency_count: int, task_type: str) -> dict:
    """
    Full prediction pipeline for one task.
    Returns a result dict with score, label, model_prediction, and reasons.
    """
    try:
        model = load_model()
    except FileNotFoundError as exc:
        logger.error("predict_single: %s", exc)
        raise

    # Prepare features and calculate rule-based score
    feature_df, days_to_deadline, deadline_points, score = prepare_single_input(
        deadline_date=deadline_date,
        estimated_effort=float(estimated_effort),
        business_impact=int(business_impact),
        urgency=int(urgency),
        dependency_count=int(dependency_count),
        task_type=task_type,
    )

    # Get model prediction
    model_label = model.predict(feature_df)[0]

    # Resolve final label: rule-based score takes precedence when model disagrees
    rule_label = score_to_label(score)
    final_label = rule_label  # rule-based score is authoritative (SRS 10.1 step 46)

    # Generate explanation reasons
    reasons = build_explanation(
        urgency=int(urgency),
        business_impact=int(business_impact),
        days_to_deadline=days_to_deadline,
        dependency_count=int(dependency_count),
        estimated_effort=float(estimated_effort),
    )

    result = {
        "priority_score":   round(score, 1),
        "priority_label":   final_label,
        "model_prediction": model_label,
        "days_to_deadline": int(days_to_deadline),
        "deadline_points":  deadline_points,
        "reasons":          reasons,
    }

    logger.info(
        "Prediction: score=%.1f label=%s model=%s",
        score, final_label, model_label,
    )
    return result


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

BATCH_REQUIRED_COLUMNS = [
    "deadline_date",
    "estimated_effort",
    "business_impact",
    "urgency",
    "dependency_count",
    "task_type",
]


def predict_batch(file_storage, max_rows: int = 500) -> dict:
    """
    Process a CSV file upload.
    Returns dict with keys: results_df, errors, total, valid_count.
    """
    try:
        df_raw = pd.read_csv(file_storage)
    except Exception as exc:
        return {"error": f"Could not read CSV file: {exc}"}

    # Check required columns
    missing_cols = [c for c in BATCH_REQUIRED_COLUMNS if c not in df_raw.columns]
    if missing_cols:
        return {"error": f"Missing required columns: {', '.join(missing_cols)}"}

    if len(df_raw) > max_rows:
        df_raw = df_raw.iloc[:max_rows]
        logger.warning("Batch truncated to %d rows", max_rows)

    errors = []
    results = []

    try:
        model = load_model()
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    for idx, row in df_raw.iterrows():
        row_errors = _validate_batch_row(row, idx)
        if row_errors:
            errors.append({"row": idx + 2, "issues": row_errors})
            continue

        try:
            feature_df, days, d_points, score = prepare_single_input(
                deadline_date=row["deadline_date"],
                estimated_effort=float(row["estimated_effort"]),
                business_impact=int(row["business_impact"]),
                urgency=int(row["urgency"]),
                dependency_count=int(row["dependency_count"]),
                task_type=str(row.get("task_type", "Other")),
            )
            model_label = model.predict(feature_df)[0]
            final_label = score_to_label(score)

            result_row = row.to_dict()
            result_row["priority_score"] = round(score, 1)
            result_row["priority_label"] = final_label
            result_row["model_prediction"] = model_label
            result_row["days_to_deadline"] = int(days)
            results.append(result_row)

        except Exception as exc:
            errors.append({"row": idx + 2, "issues": [str(exc)]})

    results_df = pd.DataFrame(results) if results else pd.DataFrame()

    logger.info("Batch prediction: %d valid, %d errors", len(results), len(errors))
    return {
        "results_df":  results_df,
        "errors":      errors,
        "total":       len(df_raw),
        "valid_count": len(results),
    }


def _validate_batch_row(row, idx: int) -> list:
    """Return list of validation error strings for a single row."""
    errs = []
    try:
        effort = float(row.get("estimated_effort", 0))
        if effort <= 0:
            errs.append("estimated_effort must be > 0")
    except (ValueError, TypeError):
        errs.append("estimated_effort must be a number")

    for field in ("business_impact", "urgency"):
        try:
            val = int(row.get(field, 0))
            if not (1 <= val <= 10):
                errs.append(f"{field} must be 1-10 (got {val})")
        except (ValueError, TypeError):
            errs.append(f"{field} must be an integer")

    try:
        dep = int(row.get("dependency_count", 0))
        if dep < 0:
            errs.append("dependency_count must be >= 0")
    except (ValueError, TypeError):
        errs.append("dependency_count must be an integer")

    try:
        pd.to_datetime(row.get("deadline_date"))
    except Exception:
        errs.append("deadline_date is not a valid date")

    return errs
