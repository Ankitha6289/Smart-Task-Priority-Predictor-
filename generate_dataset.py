"""
generate_dataset.py - Generates a balanced synthetic training dataset.
Run this once to create data/tasks.csv before training the model.
"""

import pandas as pd
import numpy as np
import random
from datetime import date, timedelta
import os

from preprocessing import calculate_deadline_points, calculate_priority_score, score_to_label

TASK_TYPES = ["Development", "Testing", "Design", "Documentation", "Other"]
STATUSES = ["Not Started", "In Progress", "Blocked"]

random.seed(42)
np.random.seed(42)

NUM_RECORDS = 2000   # more records; balanced across 4 classes


def random_date_offset(min_days: int, max_days: int) -> date:
    offset = random.randint(min_days, max_days)
    return date.today() + timedelta(days=offset)


def generate_row(task_id: int):
    """Generate one synthetic task row with a deterministic priority label."""
    task_type = random.choice(TASK_TYPES)
    created = date.today() - timedelta(days=random.randint(0, 30))

    # Force balanced class distribution using segments
    segment = task_id % 4
    if segment == 0:
        # Critical: high impact + urgency + near/overdue deadline
        deadline = date.today() + timedelta(days=random.randint(-5, 3))
        business_impact = random.randint(8, 10)
        urgency = random.randint(8, 10)
        dependency_count = random.randint(3, 8)
        estimated_effort = round(random.uniform(0.5, 15.0), 1)
    elif segment == 1:
        # High: good values but not maxed out
        deadline = date.today() + timedelta(days=random.randint(1, 10))
        business_impact = random.randint(6, 9)
        urgency = random.randint(6, 9)
        dependency_count = random.randint(1, 5)
        estimated_effort = round(random.uniform(2.0, 25.0), 1)
    elif segment == 2:
        # Medium: moderate values
        deadline = date.today() + timedelta(days=random.randint(5, 20))
        business_impact = random.randint(4, 7)
        urgency = random.randint(4, 7)
        dependency_count = random.randint(0, 4)
        estimated_effort = round(random.uniform(3.0, 30.0), 1)
    else:
        # Low: low impact + far deadline
        deadline = date.today() + timedelta(days=random.randint(15, 90))
        business_impact = random.randint(1, 5)
        urgency = random.randint(1, 5)
        dependency_count = random.randint(0, 3)
        estimated_effort = round(random.uniform(0.5, 40.0), 1)

    days = (deadline - date.today()).days
    d_points = calculate_deadline_points(days)
    score = calculate_priority_score(business_impact, urgency,
                                     dependency_count, d_points, estimated_effort)
    label = score_to_label(score)

    return {
        "task_id": task_id,
        "created_date": created.isoformat(),
        "deadline_date": deadline.isoformat(),
        "estimated_effort": estimated_effort,
        "business_impact": business_impact,
        "urgency": urgency,
        "dependency_count": dependency_count,
        "task_type": task_type,
        "priority_label": label,
    }


def generate_dataset(n: int = NUM_RECORDS) -> pd.DataFrame:
    rows = [generate_row(i + 1) for i in range(n)]
    df = pd.DataFrame(rows)

    print("Label distribution:")
    print(df["priority_label"].value_counts())
    print(f"\nTotal records: {len(df)}")
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv("data/tasks.csv", index=False)
    print("\nDataset saved to data/tasks.csv")
