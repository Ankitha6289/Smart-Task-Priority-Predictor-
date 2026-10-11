"""
database.py - SQLite database setup and connection handling
"""

import sqlite3
import logging
import os

DATABASE_PATH = os.environ.get("DATABASE_PATH", "task_priority.db")

logger = logging.getLogger(__name__)


def get_db_connection():
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create all tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name      TEXT    NOT NULL,
            email     TEXT    NOT NULL UNIQUE,
            password  TEXT    NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Tasks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            title            TEXT    NOT NULL,
            description      TEXT,
            deadline         DATE    NOT NULL,
            estimated_effort REAL    NOT NULL,
            business_impact  INTEGER NOT NULL,
            urgency          INTEGER NOT NULL,
            dependency_count INTEGER NOT NULL DEFAULT 0,
            task_type        TEXT    NOT NULL DEFAULT 'Other',
            status           TEXT    NOT NULL DEFAULT 'Not Started',
            created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # Predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id          INTEGER NOT NULL,
            priority_score   REAL    NOT NULL,
            priority_label   TEXT    NOT NULL,
            model_prediction TEXT    NOT NULL,
            reasons          TEXT    DEFAULT '',
            created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
    """)

    # Add reasons column to existing DB if it was created before this migration
    try:
        cursor.execute("ALTER TABLE predictions ADD COLUMN reasons TEXT DEFAULT ''")
        conn.commit()
        logger.info("Migrated: added 'reasons' column to predictions table")
    except Exception:
        pass  # Column already exists – ignore

    # Override history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS override_history (
            override_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id           INTEGER NOT NULL,
            original_priority TEXT    NOT NULL,
            new_priority      TEXT    NOT NULL,
            reason            TEXT    NOT NULL,
            created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised at %s", DATABASE_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    print("Database created successfully.")
