"""
seed_tasks.py - Populate sample task examples with ML priority predictions.
Usage:
    python seed_tasks.py                  # Seeds tasks for all registered users
    python seed_tasks.py --user-id 1      # Seeds tasks for user with ID 1
    python seed_tasks.py --email demo@example.com
"""

import argparse
import logging
from datetime import date, timedelta
from database import get_db_connection
from models import create_task, save_prediction, get_user_by_email, get_user_by_id
from prediction import predict_single

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_sample_tasks():
    today = date.today()
    return [
        {
            "title": "Patch Zero-Day SQL Injection Vulnerability in Auth API",
            "description": "Critical security vulnerability discovered in user authentication endpoint. Requires immediate hotfix and regression testing before deployment.",
            "deadline": (today + timedelta(days=1)).isoformat(),
            "estimated_effort": 2.5,
            "business_impact": 10,
            "urgency": 10,
            "dependency_count": 4,
            "task_type": "Development",
            "status": "In Progress",
        },
        {
            "title": "Implement Stripe Checkout & Subscription Billing",
            "description": "Integrate recurring subscription tiers and webhook listener to unlock paid enterprise customer conversions for Q4.",
            "deadline": (today + timedelta(days=5)).isoformat(),
            "estimated_effort": 16.0,
            "business_impact": 8,
            "urgency": 7,
            "dependency_count": 3,
            "task_type": "Development",
            "status": "Not Started",
        },
        {
            "title": "Redesign User Settings & Notification Preferences",
            "description": "Revamp settings dashboard with dark mode toggles, session management, and granular email notification preferences.",
            "deadline": (today + timedelta(days=12)).isoformat(),
            "estimated_effort": 8.0,
            "business_impact": 5,
            "urgency": 5,
            "dependency_count": 1,
            "task_type": "Design",
            "status": "Not Started",
        },
        {
            "title": "Update Developer Onboarding & API Architecture Docs",
            "description": "Refresh OpenAPI / Swagger specifications and local environment setup guide for incoming backend engineering hires.",
            "deadline": (today + timedelta(days=30)).isoformat(),
            "estimated_effort": 5.0,
            "business_impact": 3,
            "urgency": 2,
            "dependency_count": 0,
            "task_type": "Documentation",
            "status": "Not Started",
        },
        {
            "title": "Automate End-to-End Smoke Tests for Payment Gateway",
            "description": "Build automated Playwright regression test suite covering checkout and refund flows to prevent deploy-time breakages.",
            "deadline": (today + timedelta(days=3)).isoformat(),
            "estimated_effort": 2.0,
            "business_impact": 9,
            "urgency": 7,
            "dependency_count": 2,
            "task_type": "Testing",
            "status": "Not Started",
        },
    ]


def seed_for_user(user_id: int):
    tasks = get_sample_tasks()
    created_count = 0

    for item in tasks:
        task_id = create_task(
            user_id=user_id,
            title=item["title"],
            description=item["description"],
            deadline=item["deadline"],
            estimated_effort=item["estimated_effort"],
            business_impact=item["business_impact"],
            urgency=item["urgency"],
            dependency_count=item["dependency_count"],
            task_type=item["task_type"],
            status=item["status"],
        )

        if not task_id:
            logger.warning("Could not create task: %s", item["title"])
            continue

        try:
            pred = predict_single(
                deadline_date=item["deadline"],
                estimated_effort=item["estimated_effort"],
                business_impact=item["business_impact"],
                urgency=item["urgency"],
                dependency_count=item["dependency_count"],
                task_type=item["task_type"],
            )
            save_prediction(
                task_id=task_id,
                priority_score=pred["priority_score"],
                priority_label=pred["priority_label"],
                model_prediction=pred["model_prediction"],
                reasons=pred.get("reasons", []),
            )
            logger.info("Created Task #%d: '%s' -> %s (Score: %.1f)",
                        task_id, item["title"], pred["priority_label"], pred["priority_score"])
            created_count += 1
        except Exception as exc:
            logger.error("Failed prediction for task #%d: %s", task_id, exc)

    print(f"Successfully seeded {created_count} tasks for user #{user_id}.")


def main():
    parser = argparse.ArgumentParser(description="Seed sample tasks into Smart Task Priority Predictor database.")
    parser.add_argument("--user-id", type=int, help="Target user ID to seed tasks for.")
    parser.add_argument("--email", type=str, help="Target user email to seed tasks for.")
    args = parser.parse_args()

    conn = get_db_connection()
    try:
        if args.user_id:
            user = get_user_by_id(args.user_id)
            if not user:
                print(f"User with ID {args.user_id} not found.")
                return
            seed_for_user(user["user_id"])
        elif args.email:
            user = get_user_by_email(args.email)
            if not user:
                print(f"User with email '{args.email}' not found.")
                return
            seed_for_user(user["user_id"])
        else:
            # Seed for all users found in the database
            users = conn.execute("SELECT user_id, name, email FROM users").fetchall()
            if not users:
                print("No users found in database. Please register a user first via the web app.")
                return
            for u in users:
                print(f"\n--- Seeding tasks for {u['name']} ({u['email']}, ID: {u['user_id']}) ---")
                seed_for_user(u["user_id"])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
