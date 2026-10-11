"""
test_priority.py - Unit tests for priority scoring and calculation logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from preprocessing import (
    calculate_deadline_points,
    calculate_priority_score,
    score_to_label,
    build_explanation,
    prepare_single_input,
    get_feature_columns,
)
from datetime import date, timedelta


# ---------------------------------------------------------------------------
# calculate_deadline_points
# ---------------------------------------------------------------------------

class TestDeadlinePoints:
    def test_overdue(self):
        assert calculate_deadline_points(-1) == 20

    def test_overdue_many_days(self):
        assert calculate_deadline_points(-30) == 20

    def test_due_today(self):
        assert calculate_deadline_points(0) == 15

    def test_due_within_2_days(self):
        assert calculate_deadline_points(2) == 15

    def test_due_within_7_days(self):
        assert calculate_deadline_points(5) == 10

    def test_due_within_14_days(self):
        assert calculate_deadline_points(10) == 7

    def test_due_beyond_14_days(self):
        assert calculate_deadline_points(30) == 3

    def test_boundary_7_days(self):
        assert calculate_deadline_points(7) == 10

    def test_boundary_14_days(self):
        assert calculate_deadline_points(14) == 7


# ---------------------------------------------------------------------------
# calculate_priority_score
# ---------------------------------------------------------------------------

class TestPriorityScore:
    def test_maximum_score_capped_at_100(self):
        score = calculate_priority_score(10, 10, 10, 20, 1)
        assert score <= 100.0

    def test_minimum_score_at_least_0(self):
        score = calculate_priority_score(1, 1, 0, 3, 25)
        assert score >= 0.0

    def test_quick_win_adds_points(self):
        base  = calculate_priority_score(5, 5, 0, 10, 5)
        quick = calculate_priority_score(5, 5, 0, 10, 2)
        assert quick > base

    def test_high_effort_removes_points(self):
        base = calculate_priority_score(5, 5, 0, 10, 5)
        hard = calculate_priority_score(5, 5, 0, 10, 25)
        assert hard < base

    def test_score_formula_components(self):
        # impact=8, urgency=7, deps=2, deadline_pts=10, effort=5 (no adj)
        # expected = 8*4 + 7*3 + 2*2 + 10 = 32+21+4+10 = 67
        score = calculate_priority_score(8, 7, 2, 10, 5)
        assert score == pytest.approx(67.0)

    def test_overdue_high_impact_is_critical_range(self):
        score = calculate_priority_score(9, 9, 5, 20, 5)
        assert score >= 80

    def test_low_values_is_low_range(self):
        score = calculate_priority_score(1, 1, 0, 3, 5)
        assert score < 40


# ---------------------------------------------------------------------------
# score_to_label
# ---------------------------------------------------------------------------

class TestScoreToLabel:
    def test_critical(self):
        assert score_to_label(80) == "Critical"
        assert score_to_label(100) == "Critical"

    def test_high(self):
        assert score_to_label(60) == "High"
        assert score_to_label(79) == "High"

    def test_medium(self):
        assert score_to_label(40) == "Medium"
        assert score_to_label(59) == "Medium"

    def test_low(self):
        assert score_to_label(0) == "Low"
        assert score_to_label(39) == "Low"

    def test_boundary_80(self):
        assert score_to_label(80) == "Critical"

    def test_boundary_60(self):
        assert score_to_label(60) == "High"

    def test_boundary_40(self):
        assert score_to_label(40) == "Medium"


# ---------------------------------------------------------------------------
# build_explanation
# ---------------------------------------------------------------------------

class TestBuildExplanation:
    def test_high_urgency_message(self):
        reasons = build_explanation(urgency=9, business_impact=5,
                                    days_to_deadline=30, dependency_count=0,
                                    estimated_effort=5)
        assert any("urgency" in r.lower() for r in reasons)

    def test_high_impact_message(self):
        reasons = build_explanation(urgency=5, business_impact=9,
                                    days_to_deadline=30, dependency_count=0,
                                    estimated_effort=5)
        assert any("business impact" in r.lower() for r in reasons)

    def test_near_deadline_message(self):
        reasons = build_explanation(urgency=5, business_impact=5,
                                    days_to_deadline=1, dependency_count=0,
                                    estimated_effort=5)
        assert any("deadline" in r.lower() for r in reasons)

    def test_many_dependencies_message(self):
        reasons = build_explanation(urgency=5, business_impact=5,
                                    days_to_deadline=30, dependency_count=5,
                                    estimated_effort=5)
        assert any("dependenc" in r.lower() for r in reasons)

    def test_quick_win_message(self):
        reasons = build_explanation(urgency=5, business_impact=5,
                                    days_to_deadline=30, dependency_count=0,
                                    estimated_effort=1)
        assert any("quick" in r.lower() for r in reasons)

    def test_high_effort_message(self):
        reasons = build_explanation(urgency=5, business_impact=5,
                                    days_to_deadline=30, dependency_count=0,
                                    estimated_effort=25)
        assert any("effort" in r.lower() for r in reasons)

    def test_always_returns_at_least_one_reason(self):
        reasons = build_explanation(urgency=5, business_impact=5,
                                    days_to_deadline=30, dependency_count=0,
                                    estimated_effort=5)
        assert len(reasons) >= 1


# ---------------------------------------------------------------------------
# prepare_single_input
# ---------------------------------------------------------------------------

class TestPrepareSingleInput:
    def test_returns_correct_feature_count(self):
        deadline = date.today() + timedelta(days=7)
        feature_df, days, d_points, score = prepare_single_input(
            deadline_date=deadline.isoformat(),
            estimated_effort=5.0,
            business_impact=7,
            urgency=6,
            dependency_count=2,
            task_type="Development",
        )
        expected_cols = get_feature_columns()
        assert list(feature_df.columns) == expected_cols

    def test_overdue_days_is_negative(self):
        deadline = date.today() - timedelta(days=3)
        _, days, d_points, _ = prepare_single_input(
            deadline_date=deadline.isoformat(),
            estimated_effort=5.0,
            business_impact=5,
            urgency=5,
            dependency_count=0,
            task_type="Other",
        )
        assert days < 0
        assert d_points == 20

    def test_score_within_bounds(self):
        deadline = date.today() + timedelta(days=5)
        _, _, _, score = prepare_single_input(
            deadline_date=deadline.isoformat(),
            estimated_effort=8.0,
            business_impact=8,
            urgency=8,
            dependency_count=3,
            task_type="Testing",
        )
        assert 0 <= score <= 100

    def test_one_hot_encoding_present(self):
        deadline = date.today() + timedelta(days=10)
        feature_df, *_ = prepare_single_input(
            deadline_date=deadline.isoformat(),
            estimated_effort=4.0,
            business_impact=5,
            urgency=5,
            dependency_count=1,
            task_type="Design",
        )
        assert feature_df["task_type_Design"].iloc[0] == 1
        assert feature_df["task_type_Development"].iloc[0] == 0
