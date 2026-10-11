"""
preprocessing.py - Data cleaning and feature preparation.
Reused during both model training and live prediction.
"""

import pandas as pd
import numpy as np
import logging
from datetime import date

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "deadline_date",
    "estimated_effort",
    "business_impact",
    "urgency",
    "dependency_count",
    "task_type",
]

TASK_TYPES = ["Development", "Testing", "Design", "Documentation", "Other"]


def calculate_deadline_points(days: float) -> int:
    """Return deadline score points based on days remaining."""
    if days < 0:
        return 20        # overdue
    elif days <= 2:
        return 15        # due today or within 2 days
    elif days <= 7:
        return 10
    elif days <= 14:
        return 7
    else:
        return 3


def calculate_priority_score(business_impact: float, urgency: float,
                              dependency_count: float, deadline_points: int,
                              estimated_effort: float) -> float:
    """
    Core scoring formula from SRS section 6.
    score = (business_impact * 4) + (urgency * 3) + (dependency_count * 2) + deadline_points
    Adjusted for effort, clamped to [0, 100].
    """
    score = (business_impact * 4) + (urgency * 3) + (dependency_count * 2) + deadline_points

    if estimated_effort < 3:
        score += 5   # quick win
    elif estimated_effort > 20:
        score -= 5   # large effort slightly reduces urgency

    return float(max(0, min(100, score)))


def score_to_label(score: float) -> str:
    """Map numeric score to priority label (SRS section 6.4)."""
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    else:
        return "Low"


def build_explanation(urgency: int, business_impact: int,
                      days_to_deadline: float, dependency_count: int,
                      estimated_effort: float) -> list:
    """Generate simple rule-based explanation messages (SRS 10.2)."""
    reasons = []
    if urgency >= 7:
        reasons.append("Urgency is high and increases the priority.")
    if business_impact >= 7:
        reasons.append("Business impact is high and increases the priority.")
    if days_to_deadline <= 2:
        reasons.append("The deadline is close and increases the priority.")
    if dependency_count >= 4:
        reasons.append("The task has several dependencies and may affect other work.")
    if estimated_effort > 20:
        reasons.append("High effort slightly reduces the score.")
    if estimated_effort < 3:
        reasons.append("Low effort makes this a useful quick-win task.")
    if not reasons:
        reasons.append("The task has moderate priority based on its attributes.")
    return reasons


# ---------------------------------------------------------------------------
# DataFrame-level helpers used by train_model.py and batch prediction
# ---------------------------------------------------------------------------

def check_required_columns(df: pd.DataFrame) -> list:
    """Return list of missing required columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def clean_dataframe(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Convert date columns
      2. Fill missing numerical values with median
      3. Fill missing task_type with 'Other'
      4. Remove duplicate task_ids (if column present)
      5. Validate numeric ranges
      6. Compute days_to_deadline
      7. One-hot encode task_type
    Returns a clean DataFrame with model-ready feature columns.
    """
    df = df.copy()

    if reference_date is None:
        reference_date = pd.Timestamp(date.today())

    # --- dates ---
    if "created_date" in df.columns:
        df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["deadline_date"] = pd.to_datetime(df["deadline_date"], errors="coerce")

    # --- remove duplicate task_ids ---
    if "task_id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["task_id"])
        if len(df) < before:
            logger.warning("Removed %d duplicate task_id rows", before - len(df))

    # --- fill missing numeric values with median ---
    numeric_cols = ["estimated_effort", "business_impact", "urgency", "dependency_count"]
    for col in numeric_cols:
        if col in df.columns:
            median_val = df[col].median()
            missing_count = df[col].isna().sum()
            if missing_count:
                logger.info("Filling %d missing values in '%s' with median %.2f",
                            missing_count, col, median_val)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)

    # --- fill missing task_type ---
    if "task_type" in df.columns:
        df["task_type"] = df["task_type"].fillna("Other")
        df["task_type"] = df["task_type"].apply(
            lambda x: x if x in TASK_TYPES else "Other"
        )
    else:
        df["task_type"] = "Other"

    # --- validate ranges (clip rather than drop for training data) ---
    df["business_impact"] = df["business_impact"].clip(1, 10)
    df["urgency"] = df["urgency"].clip(1, 10)
    df["dependency_count"] = df["dependency_count"].clip(0, None)
    df["estimated_effort"] = df["estimated_effort"].clip(0.1, None)

    # --- days_to_deadline ---
    df["days_to_deadline"] = (df["deadline_date"] - reference_date).dt.days
    df["days_to_deadline"] = df["days_to_deadline"].fillna(30).astype(float)

    # --- deadline_points ---
    df["deadline_points"] = df["days_to_deadline"].apply(calculate_deadline_points)

    # --- one-hot encode task_type ---
    for t in TASK_TYPES:
        col_name = f"task_type_{t}"
        df[col_name] = (df["task_type"] == t).astype(int)

    return df


def get_feature_columns() -> list:
    """Return the ordered list of feature column names expected by the model."""
    base = [
        "days_to_deadline",
        "estimated_effort",
        "business_impact",
        "urgency",
        "dependency_count",
    ]
    type_cols = [f"task_type_{t}" for t in TASK_TYPES]
    return base + type_cols


def prepare_single_input(deadline_date, estimated_effort: float,
                          business_impact: int, urgency: int,
                          dependency_count: int, task_type: str,
                          reference_date=None) -> tuple:
    """
    Prepare a single task's feature vector and also return the raw
    score components for the rule engine.
    Returns: (feature_df, days_to_deadline, deadline_points, score)
    """
    if reference_date is None:
        reference_date = pd.Timestamp(date.today())

    deadline_ts = pd.Timestamp(deadline_date)
    days = float((deadline_ts - reference_date).days)
    d_points = calculate_deadline_points(days)
    score = calculate_priority_score(
        business_impact, urgency, dependency_count, d_points, estimated_effort
    )

    row = {
        "deadline_date": deadline_ts,
        "estimated_effort": float(estimated_effort),
        "business_impact": int(business_impact),
        "urgency": int(urgency),
        "dependency_count": int(dependency_count),
        "task_type": task_type,
    }
    df = pd.DataFrame([row])
    df = clean_dataframe(df, reference_date=reference_date)

    feature_df = df[get_feature_columns()]
    return feature_df, days, d_points, score
