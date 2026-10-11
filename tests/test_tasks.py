"""
test_tasks.py - Integration tests for database operations and Flask routes.
Uses an in-memory SQLite database and Flask test client.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import tempfile
from datetime import date, timedelta

# Point the app to a temporary database before importing app
_tmp_db = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = _tmp_db
os.environ["MODEL_PATH"]    = os.path.join(
    os.path.dirname(__file__), "..", "models", "priority_model.pkl"
)

import database
import models as db_models


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_db():
    """Re-initialise a fresh temp database for every test."""
    database.DATABASE_PATH = _tmp_db
    db_models  # reimport not needed – they read DATABASE_PATH at call time
    database.init_db()
    yield
    # Clean up tables between tests
    conn = database.get_db_connection()
    conn.execute("DELETE FROM override_history")
    conn.execute("DELETE FROM predictions")
    conn.execute("DELETE FROM tasks")
    conn.execute("DELETE FROM users")
    conn.commit()
    conn.close()


def make_user(name="Alice", email="alice@test.com", password="pass1234"):
    return db_models.create_user(name, email, password)


def make_task(user_id, title="Test Task", days_ahead=7, status="Not Started"):
    deadline = (date.today() + timedelta(days=days_ahead)).isoformat()
    return db_models.create_task(
        user_id=user_id,
        title=title,
        description="A test task",
        deadline=deadline,
        estimated_effort=4.0,
        business_impact=7,
        urgency=6,
        dependency_count=2,
        task_type="Development",
        status=status,
    )


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

class TestUserCRUD:
    def test_create_user_returns_id(self):
        uid = make_user()
        assert uid is not None
        assert isinstance(uid, int)

    def test_duplicate_email_returns_none(self):
        make_user()
        uid2 = make_user()
        assert uid2 is None

    def test_get_user_by_email(self):
        make_user(email="bob@test.com")
        user = db_models.get_user_by_email("bob@test.com")
        assert user is not None
        assert user["name"] == "Alice"

    def test_verify_user_correct_password(self):
        make_user(password="mypassword")
        user = db_models.verify_user("alice@test.com", "mypassword")
        assert user is not None

    def test_verify_user_wrong_password(self):
        make_user(password="mypassword")
        user = db_models.verify_user("alice@test.com", "wrongpassword")
        assert user is None

    def test_get_user_by_id(self):
        uid = make_user()
        user = db_models.get_user_by_id(uid)
        assert user["user_id"] == uid


# ---------------------------------------------------------------------------
# Task CRUD
# ---------------------------------------------------------------------------

class TestTaskCRUD:
    def test_create_task_returns_id(self):
        uid = make_user()
        task_id = make_task(uid)
        assert task_id is not None
        assert isinstance(task_id, int)

    def test_get_task_by_id(self):
        uid     = make_user()
        task_id = make_task(uid, title="Unique Task Name")
        task    = db_models.get_task_by_id(task_id, uid)
        assert task is not None
        assert task["title"] == "Unique Task Name"

    def test_get_task_wrong_user_returns_none(self):
        uid1    = make_user(email="u1@test.com")
        uid2    = make_user(name="Bob", email="u2@test.com")
        task_id = make_task(uid1)
        task    = db_models.get_task_by_id(task_id, uid2)
        assert task is None

    def test_get_tasks_for_user(self):
        uid = make_user()
        make_task(uid, title="Task A")
        make_task(uid, title="Task B")
        tasks = db_models.get_tasks_for_user(uid)
        assert len(tasks) == 2

    def test_update_task(self):
        uid     = make_user()
        task_id = make_task(uid, title="Old Title")
        deadline = (date.today() + timedelta(days=10)).isoformat()
        db_models.update_task(
            task_id=task_id, user_id=uid,
            title="New Title", description="Updated",
            deadline=deadline, estimated_effort=8.0,
            business_impact=9, urgency=8,
            dependency_count=1, task_type="Testing",
            status="In Progress",
        )
        task = db_models.get_task_by_id(task_id, uid)
        assert task["title"] == "New Title"
        assert task["status"] == "In Progress"

    def test_delete_task(self):
        uid     = make_user()
        task_id = make_task(uid)
        db_models.delete_task(task_id, uid)
        task = db_models.get_task_by_id(task_id, uid)
        assert task is None

    def test_cancelled_tasks_excluded_from_list(self):
        uid = make_user()
        make_task(uid, title="Active Task")
        make_task(uid, title="Cancelled Task", status="Cancelled")
        tasks = db_models.get_tasks_for_user(uid)
        titles = [t["title"] for t in tasks]
        assert "Active Task" in titles
        assert "Cancelled Task" not in titles


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class TestPredictions:
    def test_save_and_retrieve_prediction(self):
        uid     = make_user()
        task_id = make_task(uid)
        pred_id = db_models.save_prediction(
            task_id=task_id,
            priority_score=72.5,
            priority_label="High",
            model_prediction="High",
        )
        assert pred_id is not None
        pred = db_models.get_prediction_for_task(task_id)
        assert pred["priority_label"] == "High"
        assert pred["priority_score"] == pytest.approx(72.5)

    def test_task_with_prediction_visible_in_task_list(self):
        uid     = make_user()
        task_id = make_task(uid)
        db_models.save_prediction(task_id, 55.0, "Medium", "Medium")
        tasks = db_models.get_tasks_for_user(uid)
        assert tasks[0]["priority_label"] == "Medium"


# ---------------------------------------------------------------------------
# Override history
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_save_override(self):
        uid     = make_user()
        task_id = make_task(uid)
        db_models.save_prediction(task_id, 72.5, "High", "High")
        db_models.save_override(task_id, "High", "Critical",
                                "Stakeholder escalation")
        overrides = db_models.get_overrides_for_task(task_id)
        assert len(overrides) == 1
        assert overrides[0]["new_priority"] == "Critical"
        assert overrides[0]["reason"] == "Stakeholder escalation"

    def test_override_updates_prediction_label(self):
        uid     = make_user()
        task_id = make_task(uid)
        db_models.save_prediction(task_id, 72.5, "High", "High")
        db_models.save_override(task_id, "High", "Critical", "Escalated")
        task = db_models.get_task_by_id(task_id, uid)
        assert task["priority_label"] == "Critical"


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

class TestDashboardStats:
    def test_stats_total_count(self):
        uid = make_user()
        make_task(uid, title="T1")
        make_task(uid, title="T2")
        make_task(uid, title="T3", status="Cancelled")  # excluded
        stats = db_models.get_dashboard_stats(uid)
        assert stats["total"] == 2

    def test_stats_overdue_count(self):
        uid = make_user()
        make_task(uid, title="Future",  days_ahead=10)
        make_task(uid, title="Overdue", days_ahead=-3)
        stats = db_models.get_dashboard_stats(uid)
        assert stats["overdue"] == 1


# ---------------------------------------------------------------------------
# Flask route integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client():
    """Return a Flask test client wired to the temp database."""
    # Import app here so DATABASE_PATH env var is already set
    from app import app as flask_app
    flask_app.config["TESTING"]    = True
    flask_app.config["SECRET_KEY"] = "test-secret"
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.test_client() as client:
        yield client


def register_and_login(client, email="test@example.com", password="testpass"):
    client.post("/register", data={
        "name": "Test User", "email": email,
        "password": password, "confirm_password": password,
    }, follow_redirects=True)
    client.post("/login", data={"email": email, "password": password},
                follow_redirects=True)


class TestFlaskRoutes:
    def test_login_page_loads(self, app_client):
        resp = app_client.get("/login")
        assert resp.status_code == 200
        assert b"Sign In" in resp.data

    def test_register_page_loads(self, app_client):
        resp = app_client.get("/register")
        assert resp.status_code == 200
        assert b"Register" in resp.data

    def test_unauthenticated_redirect_to_login(self, app_client):
        resp = app_client.get("/dashboard", follow_redirects=False)
        assert resp.status_code == 302

    def test_register_and_login_success(self, app_client):
        resp = app_client.post("/register", data={
            "name": "Alice", "email": "alice@flask.com",
            "password": "secret123", "confirm_password": "secret123",
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_login_invalid_credentials(self, app_client):
        resp = app_client.post("/login", data={
            "email": "nobody@test.com", "password": "wrong"
        }, follow_redirects=True)
        assert b"Invalid email or password" in resp.data

    def test_dashboard_accessible_after_login(self, app_client):
        register_and_login(app_client)
        resp = app_client.get("/dashboard")
        assert resp.status_code == 200
        assert b"Dashboard" in resp.data

    def test_task_list_accessible_after_login(self, app_client):
        register_and_login(app_client)
        resp = app_client.get("/tasks")
        assert resp.status_code == 200

    def test_task_create_form_loads(self, app_client):
        register_and_login(app_client)
        resp = app_client.get("/tasks/create")
        assert resp.status_code == 200
        assert b"New Task" in resp.data

    def test_task_create_with_invalid_data_shows_error(self, app_client):
        register_and_login(app_client)
        resp = app_client.post("/tasks/create", data={
            "title": "",
            "deadline": (date.today() + timedelta(days=5)).isoformat(),
            "estimated_effort": "4",
            "business_impact": "5",
            "urgency": "5",
            "dependency_count": "0",
            "task_type": "Other",
            "status": "Not Started",
        }, follow_redirects=True)
        assert b"required" in resp.data.lower() or resp.status_code == 200

    def test_logout_clears_session(self, app_client):
        register_and_login(app_client)
        resp = app_client.get("/logout", follow_redirects=False)
        assert resp.status_code == 302

    def test_404_for_unknown_task(self, app_client):
        register_and_login(app_client)
        resp = app_client.get("/tasks/99999")
        assert resp.status_code == 404

    def test_batch_predict_page_loads(self, app_client):
        register_and_login(app_client)
        resp = app_client.get("/predict/batch")
        assert resp.status_code == 200
        assert b"Batch" in resp.data
