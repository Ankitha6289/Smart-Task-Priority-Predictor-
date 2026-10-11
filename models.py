"""
models.py - Database helper functions for all CRUD operations.
All queries use parameterised statements to prevent SQL injection.
"""

import logging
from database import get_db_connection
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------

def create_user(name: str, email: str, password: str):
    """Insert a new user. Returns the new user_id or None on duplicate email."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
            (name.strip(), email.strip().lower(), generate_password_hash(password)),
        )
        conn.commit()
        user_id = cursor.lastrowid
        logger.info("User created: id=%s email=%s", user_id, email)
        return user_id
    except Exception as exc:
        logger.error("create_user error: %s", exc)
        return None
    finally:
        conn.close()


def get_user_by_email(email: str):
    """Return the user row for the given email, or None."""
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return row
    finally:
        conn.close()


def verify_user(email: str, password: str):
    """Return the user row if credentials are valid, else None."""
    user = get_user_by_email(email)
    if user and check_password_hash(user["password"], password):
        return user
    return None


def get_user_by_id(user_id: int):
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def create_task(user_id, title, description, deadline, estimated_effort,
                business_impact, urgency, dependency_count, task_type, status):
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO tasks
               (user_id, title, description, deadline, estimated_effort,
                business_impact, urgency, dependency_count, task_type, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, title.strip(), description, deadline, float(estimated_effort),
             int(business_impact), int(urgency), int(dependency_count),
             task_type, status),
        )
        conn.commit()
        task_id = cursor.lastrowid
        logger.info("Task created: id=%s user=%s", task_id, user_id)
        return task_id
    except Exception as exc:
        logger.error("create_task error: %s", exc)
        return None
    finally:
        conn.close()


def get_tasks_for_user(user_id, status_filter=None, priority_filter=None,
                       search=None, sort_by_priority=False):
    """
    Return tasks for the given user, optionally filtered and sorted.
    Joins the latest prediction for priority_label and priority_score.
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT t.*,
                   p.priority_score,
                   p.priority_label,
                   p.model_prediction
            FROM tasks t
            LEFT JOIN predictions p ON p.prediction_id = (
                SELECT MAX(prediction_id) FROM predictions WHERE task_id = t.task_id
            )
            WHERE t.user_id = ?
              AND t.status != 'Cancelled'
        """
        params = [user_id]

        if status_filter:
            query += " AND t.status = ?"
            params.append(status_filter)

        if priority_filter:
            query += " AND p.priority_label = ?"
            params.append(priority_filter)

        if search:
            query += " AND t.title LIKE ?"
            params.append(f"%{search}%")

        if sort_by_priority:
            query += """
                ORDER BY CASE p.priority_label
                    WHEN 'Critical' THEN 1
                    WHEN 'High'     THEN 2
                    WHEN 'Medium'   THEN 3
                    WHEN 'Low'      THEN 4
                    ELSE 5 END,
                t.deadline ASC
            """
        else:
            query += " ORDER BY t.created_at DESC"

        return conn.execute(query, params).fetchall()
    finally:
        conn.close()


def get_task_by_id(task_id: int, user_id: int = None):
    """Return a single task. If user_id supplied, enforces ownership."""
    conn = get_db_connection()
    try:
        query = """
            SELECT t.*,
                   p.priority_score,
                   p.priority_label,
                   p.model_prediction,
                   p.reasons,
                   p.created_at AS prediction_date
            FROM tasks t
            LEFT JOIN predictions p ON p.prediction_id = (
                SELECT MAX(prediction_id) FROM predictions WHERE task_id = t.task_id
            )
            WHERE t.task_id = ?
        """
        params = [task_id]
        if user_id is not None:
            query += " AND t.user_id = ?"
            params.append(user_id)
        return conn.execute(query, params).fetchone()
    finally:
        conn.close()


def update_task(task_id, user_id, title, description, deadline, estimated_effort,
                business_impact, urgency, dependency_count, task_type, status):
    conn = get_db_connection()
    try:
        conn.execute(
            """UPDATE tasks SET
               title=?, description=?, deadline=?, estimated_effort=?,
               business_impact=?, urgency=?, dependency_count=?,
               task_type=?, status=?
               WHERE task_id=? AND user_id=?""",
            (title.strip(), description, deadline, float(estimated_effort),
             int(business_impact), int(urgency), int(dependency_count),
             task_type, status, task_id, user_id),
        )
        conn.commit()
        logger.info("Task updated: id=%s", task_id)
    except Exception as exc:
        logger.error("update_task error: %s", exc)
    finally:
        conn.close()


def delete_task(task_id: int, user_id: int):
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM tasks WHERE task_id=? AND user_id=?", (task_id, user_id)
        )
        conn.commit()
        logger.info("Task deleted: id=%s", task_id)
    except Exception as exc:
        logger.error("delete_task error: %s", exc)
    finally:
        conn.close()


def update_task_status(task_id: int, user_id: int, status: str):
    conn = get_db_connection()
    try:
        conn.execute(
            "UPDATE tasks SET status=? WHERE task_id=? AND user_id=?",
            (status, task_id, user_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def save_prediction(task_id, priority_score, priority_label, model_prediction, reasons=None):
    """Save prediction result including explanation reasons (stored as pipe-separated string)."""
    if reasons is None:
        reasons = []
    reasons_str = " | ".join(reasons) if reasons else ""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """INSERT INTO predictions
               (task_id, priority_score, priority_label, model_prediction, reasons)
               VALUES (?, ?, ?, ?, ?)""",
            (task_id, float(priority_score), priority_label, model_prediction, reasons_str),
        )
        conn.commit()
        logger.info("Prediction saved for task_id=%s label=%s score=%.1f",
                    task_id, priority_label, priority_score)
        return cursor.lastrowid
    except Exception as exc:
        logger.error("save_prediction error: %s", exc)
        return None
    finally:
        conn.close()


def get_prediction_for_task(task_id: int):
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM predictions WHERE task_id=? ORDER BY prediction_id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------

def save_override(task_id, original_priority, new_priority, reason):
    conn = get_db_connection()
    try:
        conn.execute(
            """INSERT INTO override_history
               (task_id, original_priority, new_priority, reason)
               VALUES (?, ?, ?, ?)""",
            (task_id, original_priority, new_priority, reason.strip()),
        )
        # Also update the priority_label on the latest prediction row so the
        # task list reflects the overridden value immediately.
        conn.execute(
            """UPDATE predictions SET priority_label=?
               WHERE prediction_id = (
                   SELECT MAX(prediction_id) FROM predictions WHERE task_id=?
               )""",
            (new_priority, task_id),
        )
        conn.commit()
        logger.info("Override saved: task_id=%s %s->%s", task_id,
                    original_priority, new_priority)
    except Exception as exc:
        logger.error("save_override error: %s", exc)
    finally:
        conn.close()


def get_overrides_for_task(task_id: int):
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM override_history WHERE task_id=? ORDER BY created_at DESC",
            (task_id,),
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

def get_dashboard_stats(user_id: int):
    """Return counts needed for the dashboard."""
    conn = get_db_connection()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE user_id=? AND status!='Cancelled'",
            (user_id,),
        ).fetchone()[0]

        priority_counts = {}
        for label in ("Critical", "High", "Medium", "Low"):
            count = conn.execute(
                """SELECT COUNT(*) FROM tasks t
                   JOIN predictions p ON p.prediction_id = (
                       SELECT MAX(prediction_id) FROM predictions WHERE task_id = t.task_id
                   )
                   WHERE t.user_id=? AND t.status!='Cancelled' AND p.priority_label=?""",
                (user_id, label),
            ).fetchone()[0]
            priority_counts[label] = count

        overdue = conn.execute(
            """SELECT COUNT(*) FROM tasks
               WHERE user_id=? AND status NOT IN ('Completed','Cancelled')
               AND deadline < DATE('now')""",
            (user_id,),
        ).fetchone()[0]

        recent = conn.execute(
            """SELECT t.*, p.priority_label, p.priority_score
               FROM tasks t
               LEFT JOIN predictions p ON p.prediction_id = (
                   SELECT MAX(prediction_id) FROM predictions WHERE task_id = t.task_id
               )
               WHERE t.user_id=? AND t.status!='Cancelled'
               ORDER BY t.created_at DESC LIMIT 5""",
            (user_id,),
        ).fetchall()

        return {
            "total": total,
            "priority_counts": priority_counts,
            "overdue": overdue,
            "recent_tasks": recent,
        }
    finally:
        conn.close()
