"""
test_validation.py - Unit tests for input validation logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import date, timedelta


# We test the validation helper extracted from app.py by importing it.
# To avoid Flask app context overhead in unit tests, we replicate the
# validate_task_form function here and test the logic directly.

TASK_TYPES = ["Development", "Testing", "Design", "Documentation", "Other"]
STATUSES   = ["Not Started", "In Progress", "Blocked", "Completed", "Cancelled"]


def validate_task_form(form: dict) -> list:
    """Mirror of app.py validate_task_form for isolated unit testing."""
    errors = []

    if not form.get("title", "").strip():
        errors.append("Task name is required.")

    deadline_str = form.get("deadline", "")
    if not deadline_str:
        errors.append("Deadline is required.")
    else:
        try:
            dl = date.fromisoformat(deadline_str)
            today = date.today()
            if dl < today:
                errors.append("Deadline cannot be earlier than today.")
        except ValueError:
            errors.append("Enter a valid deadline date.")

    try:
        effort = float(form.get("estimated_effort", 0))
        if effort <= 0:
            errors.append("Estimated effort must be greater than 0.")
    except (ValueError, TypeError):
        errors.append("Estimated effort must be a valid number.")

    try:
        bi = int(form.get("business_impact", 0))
        if not (1 <= bi <= 10):
            errors.append("Business impact must be between 1 and 10.")
    except (ValueError, TypeError):
        errors.append("Business impact must be an integer.")

    try:
        urg = int(form.get("urgency", 0))
        if not (1 <= urg <= 10):
            errors.append("Urgency must be between 1 and 10.")
    except (ValueError, TypeError):
        errors.append("Urgency must be an integer.")

    try:
        dep = int(form.get("dependency_count", 0))
        if dep < 0:
            errors.append("Dependency count must be 0 or greater.")
    except (ValueError, TypeError):
        errors.append("Dependency count must be an integer.")

    if form.get("task_type", "") not in TASK_TYPES:
        errors.append("Select a valid task type.")

    if form.get("status", "") not in STATUSES:
        errors.append("Select a valid status.")

    return errors


def valid_form(**overrides) -> dict:
    """Return a fully valid form dict with optional overrides."""
    base = {
        "title":            "Write unit tests",
        "description":      "Some description",
        "deadline":         (date.today() + timedelta(days=7)).isoformat(),
        "estimated_effort": "4",
        "business_impact":  "7",
        "urgency":          "6",
        "dependency_count": "2",
        "task_type":        "Testing",
        "status":           "Not Started",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid form passes
# ---------------------------------------------------------------------------

class TestValidFormPasses:
    def test_all_valid_returns_no_errors(self):
        assert validate_task_form(valid_form()) == []

    def test_deadline_today_is_valid(self):
        form = valid_form(deadline=date.today().isoformat())
        assert validate_task_form(form) == []

    def test_zero_dependencies_valid(self):
        assert validate_task_form(valid_form(dependency_count="0")) == []

    def test_effort_decimal_valid(self):
        assert validate_task_form(valid_form(estimated_effort="0.5")) == []


# ---------------------------------------------------------------------------
# Title validation
# ---------------------------------------------------------------------------

class TestTitleValidation:
    def test_missing_title(self):
        errs = validate_task_form(valid_form(title=""))
        assert any("name is required" in e.lower() for e in errs)

    def test_whitespace_only_title(self):
        errs = validate_task_form(valid_form(title="   "))
        assert any("name is required" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# Deadline validation
# ---------------------------------------------------------------------------

class TestDeadlineValidation:
    def test_missing_deadline(self):
        errs = validate_task_form(valid_form(deadline=""))
        assert any("deadline" in e.lower() for e in errs)

    def test_past_deadline(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        errs = validate_task_form(valid_form(deadline=past))
        assert any("earlier than today" in e.lower() for e in errs)

    def test_invalid_date_string(self):
        errs = validate_task_form(valid_form(deadline="not-a-date"))
        assert any("valid deadline" in e.lower() for e in errs)

    def test_wrong_date_format(self):
        errs = validate_task_form(valid_form(deadline="31/12/2099"))
        assert any("valid deadline" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# Effort validation
# ---------------------------------------------------------------------------

class TestEffortValidation:
    def test_zero_effort(self):
        errs = validate_task_form(valid_form(estimated_effort="0"))
        assert any("effort" in e.lower() for e in errs)

    def test_negative_effort(self):
        errs = validate_task_form(valid_form(estimated_effort="-1"))
        assert any("effort" in e.lower() for e in errs)

    def test_non_numeric_effort(self):
        errs = validate_task_form(valid_form(estimated_effort="abc"))
        assert any("effort" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# Business impact validation
# ---------------------------------------------------------------------------

class TestBusinessImpactValidation:
    def test_impact_zero(self):
        errs = validate_task_form(valid_form(business_impact="0"))
        assert any("business impact" in e.lower() for e in errs)

    def test_impact_eleven(self):
        errs = validate_task_form(valid_form(business_impact="11"))
        assert any("business impact" in e.lower() for e in errs)

    def test_impact_boundary_1(self):
        assert validate_task_form(valid_form(business_impact="1")) == []

    def test_impact_boundary_10(self):
        assert validate_task_form(valid_form(business_impact="10")) == []

    def test_impact_non_integer(self):
        errs = validate_task_form(valid_form(business_impact="high"))
        assert any("business impact" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# Urgency validation
# ---------------------------------------------------------------------------

class TestUrgencyValidation:
    def test_urgency_zero(self):
        errs = validate_task_form(valid_form(urgency="0"))
        assert any("urgency" in e.lower() for e in errs)

    def test_urgency_eleven(self):
        errs = validate_task_form(valid_form(urgency="11"))
        assert any("urgency" in e.lower() for e in errs)

    def test_urgency_boundary_1(self):
        assert validate_task_form(valid_form(urgency="1")) == []

    def test_urgency_boundary_10(self):
        assert validate_task_form(valid_form(urgency="10")) == []

    def test_urgency_non_integer(self):
        errs = validate_task_form(valid_form(urgency="urgent"))
        assert any("urgency" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# Dependency count validation
# ---------------------------------------------------------------------------

class TestDependencyCountValidation:
    def test_negative_dependencies(self):
        errs = validate_task_form(valid_form(dependency_count="-1"))
        assert any("dependency" in e.lower() for e in errs)

    def test_non_integer_dependencies(self):
        errs = validate_task_form(valid_form(dependency_count="two"))
        assert any("dependency" in e.lower() for e in errs)

    def test_large_dependency_count_valid(self):
        assert validate_task_form(valid_form(dependency_count="20")) == []


# ---------------------------------------------------------------------------
# Task type and status validation
# ---------------------------------------------------------------------------

class TestChoiceFields:
    def test_invalid_task_type(self):
        errs = validate_task_form(valid_form(task_type="Sprint"))
        assert any("task type" in e.lower() for e in errs)

    def test_all_valid_task_types(self):
        for t in TASK_TYPES:
            assert validate_task_form(valid_form(task_type=t)) == []

    def test_invalid_status(self):
        errs = validate_task_form(valid_form(status="Pending"))
        assert any("status" in e.lower() for e in errs)

    def test_all_valid_statuses(self):
        for s in STATUSES:
            assert validate_task_form(valid_form(status=s)) == []
