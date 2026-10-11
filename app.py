"""
app.py - Main Flask application with all routes.
"""

import os
import io
import logging
import tempfile
from datetime import date
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, g,
)

from database import init_db
from models import (
    create_user, verify_user, get_user_by_id,
    create_task, get_tasks_for_user, get_task_by_id,
    update_task, delete_task, update_task_status,
    save_prediction, save_override, get_overrides_for_task,
    get_dashboard_stats,
)
from prediction import predict_single, predict_batch

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-prod-2024")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TASK_TYPES  = ["Development", "Testing", "Design", "Documentation", "Other"]
STATUSES    = ["Not Started", "In Progress", "Blocked", "Completed", "Cancelled"]
PRIORITIES  = ["Low", "Medium", "High", "Critical"]

# ---------------------------------------------------------------------------
# DB init on first request
# ---------------------------------------------------------------------------

with app.app_context():
    init_db()
    logger.info("Application started")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def current_user():
    if "user_id" in session:
        return get_user_by_id(session["user_id"])
    return None


@app.context_processor
def inject_user():
    return {
        "current_user": current_user(),
        "now_date": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_task_form(form) -> list:
    """Return list of error strings. Empty list means valid."""
    errors = []
    if not form.get("title", "").strip():
        errors.append("Task name is required.")

    # deadline
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

    # estimated effort
    try:
        effort = float(form.get("estimated_effort", 0))
        if effort <= 0:
            errors.append("Estimated effort must be greater than 0.")
    except (ValueError, TypeError):
        errors.append("Estimated effort must be a valid number.")

    # business impact
    try:
        bi = int(form.get("business_impact", 0))
        if not (1 <= bi <= 10):
            errors.append("Business impact must be between 1 and 10.")
    except (ValueError, TypeError):
        errors.append("Business impact must be an integer.")

    # urgency
    try:
        urg = int(form.get("urgency", 0))
        if not (1 <= urg <= 10):
            errors.append("Urgency must be between 1 and 10.")
    except (ValueError, TypeError):
        errors.append("Urgency must be an integer.")

    # dependency count
    try:
        dep = int(form.get("dependency_count", 0))
        if dep < 0:
            errors.append("Dependency count must be 0 or greater.")
    except (ValueError, TypeError):
        errors.append("Dependency count must be an integer.")

    task_type = form.get("task_type", "")
    if task_type not in TASK_TYPES:
        errors.append("Select a valid task type.")

    status = form.get("status", "")
    if status not in STATUSES:
        errors.append("Select a valid status.")

    return errors


# ---------------------------------------------------------------------------
# Routes – Auth
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm_password", "")

        errors = []
        if not name:
            errors.append("Name is required.")
        if not email or "@" not in email:
            errors.append("Enter a valid email address.")
        if len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        if not errors:
            user_id = create_user(name, email, password)
            if user_id:
                session["user_id"]   = user_id
                session["user_name"] = name
                flash("Account created! Welcome, " + name + ".", "success")
                logger.info("New user registered: %s", email)
                return redirect(url_for("dashboard"))
            else:
                errors.append("Email is already registered.")

        for e in errors:
            flash(e, "danger")
        return render_template("register.html", name=name, email=email)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = verify_user(email, password)
        if user:
            session["user_id"]   = user["user_id"]
            session["user_name"] = user["name"]
            flash("Welcome back, " + user["name"] + "!", "success")
            logger.info("User login success: %s", email)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "danger")
            logger.warning("User login failed: %s", email)
            return render_template("login.html", email=email)

    return render_template("login.html")


@app.route("/logout")
def logout():
    name = session.get("user_name", "")
    session.clear()
    flash("You have been logged out.", "info")
    logger.info("User logged out: %s", name)
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes – Dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    stats   = get_dashboard_stats(user_id)

    # Build chart data for Matplotlib base64 image
    chart_b64 = _build_priority_chart(stats["priority_counts"])

    return render_template(
        "dashboard.html",
        stats=stats,
        chart_b64=chart_b64,
    )


def _build_priority_chart(priority_counts: dict) -> str:
    """Return a base64-encoded PNG bar chart of priority distribution."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import base64

    labels = ["Critical", "High", "Medium", "Low"]
    values = [priority_counts.get(l, 0) for l in labels]
    colours = ["#dc3545", "#fd7e14", "#ffc107", "#198754"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_title("Task Priority Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Number of Tasks")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes – Tasks
# ---------------------------------------------------------------------------

@app.route("/tasks")
@login_required
def task_list():
    user_id        = session["user_id"]
    status_filter  = request.args.get("status", "")
    priority_filter= request.args.get("priority", "")
    search         = request.args.get("search", "")
    sort_priority  = request.args.get("sort", "") == "priority"

    tasks = get_tasks_for_user(
        user_id,
        status_filter   = status_filter  or None,
        priority_filter = priority_filter or None,
        search          = search          or None,
        sort_by_priority= sort_priority,
    )

    return render_template(
        "tasks.html",
        tasks          = tasks,
        statuses       = STATUSES,
        priorities     = PRIORITIES,
        status_filter  = status_filter,
        priority_filter= priority_filter,
        search         = search,
        sort_priority  = sort_priority,
    )


@app.route("/tasks/create", methods=["GET", "POST"])
@login_required
def task_create():
    if request.method == "POST":
        errors = validate_task_form(request.form)

        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template(
                "task_form.html",
                task=request.form, task_types=TASK_TYPES,
                statuses=STATUSES, action="create",
            )

        # Business rule BR-01: don't predict for Completed tasks
        status = request.form.get("status")

        task_id = create_task(
            user_id          = session["user_id"],
            title            = request.form["title"],
            description      = request.form.get("description", ""),
            deadline         = request.form["deadline"],
            estimated_effort = float(request.form["estimated_effort"]),
            business_impact  = int(request.form["business_impact"]),
            urgency          = int(request.form["urgency"]),
            dependency_count = int(request.form["dependency_count"]),
            task_type        = request.form["task_type"],
            status           = status,
        )

        if not task_id:
            flash("Failed to create task. Please try again.", "danger")
            return render_template(
                "task_form.html",
                task=request.form, task_types=TASK_TYPES,
                statuses=STATUSES, action="create",
            )

        # Run prediction unless task is Completed/Cancelled
        prediction_result = None
        if status not in ("Completed", "Cancelled"):
            try:
                prediction_result = predict_single(
                    deadline_date    = request.form["deadline"],
                    estimated_effort = float(request.form["estimated_effort"]),
                    business_impact  = int(request.form["business_impact"]),
                    urgency          = int(request.form["urgency"]),
                    dependency_count = int(request.form["dependency_count"]),
                    task_type        = request.form["task_type"],
                )
                save_prediction(
                    task_id          = task_id,
                    priority_score   = prediction_result["priority_score"],
                    priority_label   = prediction_result["priority_label"],
                    model_prediction = prediction_result["model_prediction"],
                    reasons          = prediction_result.get("reasons", []),
                )
                logger.info("Task created with prediction: task_id=%s", task_id)
            except FileNotFoundError:
                flash("Prediction model not found. Run train_model.py first.", "warning")
            except Exception as exc:
                logger.error("Prediction error: %s", exc)
                flash("Prediction failed. Task saved without priority.", "warning")

        flash("Task created successfully!", "success")
        return redirect(url_for("task_detail", task_id=task_id))

    return render_template(
        "task_form.html",
        task=None, task_types=TASK_TYPES,
        statuses=STATUSES, action="create",
        today=date.today().isoformat(),
    )


@app.route("/tasks/<int:task_id>")
@login_required
def task_detail(task_id):
    task = get_task_by_id(task_id, user_id=session["user_id"])
    if not task:
        flash("Task not found.", "danger")
        return render_template("404.html"), 404

    overrides = get_overrides_for_task(task_id)
    return render_template(
        "task_detail.html",
        task=task,
        overrides=overrides,
        priorities=PRIORITIES,
    )


@app.route("/tasks/<int:task_id>/edit", methods=["GET", "POST"])
@login_required
def task_edit(task_id):
    task = get_task_by_id(task_id, user_id=session["user_id"])
    if not task:
        flash("Task not found.", "danger")
        return render_template("404.html"), 404

    if request.method == "POST":
        errors = validate_task_form(request.form)
        if errors:
            for e in errors:
                flash(e, "danger")
            return render_template(
                "task_form.html",
                task=request.form, task_types=TASK_TYPES,
                statuses=STATUSES, action="edit", task_id=task_id,
            )

        update_task(
            task_id          = task_id,
            user_id          = session["user_id"],
            title            = request.form["title"],
            description      = request.form.get("description", ""),
            deadline         = request.form["deadline"],
            estimated_effort = float(request.form["estimated_effort"]),
            business_impact  = int(request.form["business_impact"]),
            urgency          = int(request.form["urgency"]),
            dependency_count = int(request.form["dependency_count"]),
            task_type        = request.form["task_type"],
            status           = request.form["status"],
        )

        # Re-run prediction on edit unless Completed/Cancelled
        status = request.form.get("status")
        if status not in ("Completed", "Cancelled"):
            try:
                prediction_result = predict_single(
                    deadline_date    = request.form["deadline"],
                    estimated_effort = float(request.form["estimated_effort"]),
                    business_impact  = int(request.form["business_impact"]),
                    urgency          = int(request.form["urgency"]),
                    dependency_count = int(request.form["dependency_count"]),
                    task_type        = request.form["task_type"],
                )
                save_prediction(
                    task_id          = task_id,
                    priority_score   = prediction_result["priority_score"],
                    priority_label   = prediction_result["priority_label"],
                    model_prediction = prediction_result["model_prediction"],
                    reasons          = prediction_result.get("reasons", []),
                )
                logger.info("Task updated with new prediction: task_id=%s", task_id)
            except Exception as exc:
                logger.error("Re-prediction error on edit: %s", exc)
                flash("Task updated, but prediction failed.", "warning")

        flash("Task updated successfully!", "success")
        return redirect(url_for("task_detail", task_id=task_id))

    return render_template(
        "task_form.html",
        task=task, task_types=TASK_TYPES,
        statuses=STATUSES, action="edit", task_id=task_id,
        today=date.today().isoformat(),
    )


@app.route("/tasks/<int:task_id>/delete", methods=["POST"])
@login_required
def task_delete(task_id):
    task = get_task_by_id(task_id, user_id=session["user_id"])
    if not task:
        flash("Task not found.", "danger")
        return redirect(url_for("task_list"))

    delete_task(task_id, user_id=session["user_id"])
    flash("Task deleted.", "success")
    logger.info("Task deleted: id=%s by user=%s", task_id, session["user_id"])
    return redirect(url_for("task_list"))


@app.route("/tasks/<int:task_id>/override", methods=["POST"])
@login_required
def task_override(task_id):
    task = get_task_by_id(task_id, user_id=session["user_id"])
    if not task:
        flash("Task not found.", "danger")
        return redirect(url_for("task_list"))

    new_priority = request.form.get("new_priority", "")
    reason       = request.form.get("reason", "").strip()

    if new_priority not in PRIORITIES:
        flash("Select a valid priority.", "danger")
        return redirect(url_for("task_detail", task_id=task_id))

    if not reason:
        flash("A reason is required for manual override (BR-10).", "danger")
        return redirect(url_for("task_detail", task_id=task_id))

    original = task["priority_label"] or "Unknown"
    save_override(task_id, original, new_priority, reason)
    flash(f"Priority overridden to {new_priority}.", "success")
    return redirect(url_for("task_detail", task_id=task_id))


# ---------------------------------------------------------------------------
# Routes – Standalone predict (quick predict without saving a task)
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    errors = validate_task_form(request.form)
    if errors:
        return {"errors": errors}, 400

    try:
        result = predict_single(
            deadline_date    = request.form["deadline"],
            estimated_effort = float(request.form["estimated_effort"]),
            business_impact  = int(request.form["business_impact"]),
            urgency          = int(request.form["urgency"]),
            dependency_count = int(request.form["dependency_count"]),
            task_type        = request.form.get("task_type", "Other"),
        )
        return result
    except FileNotFoundError as exc:
        return {"error": str(exc)}, 503
    except Exception as exc:
        logger.error("/predict error: %s", exc)
        return {"error": "Prediction service error."}, 500


# ---------------------------------------------------------------------------
# Routes – Batch prediction
# ---------------------------------------------------------------------------

@app.route("/predict/batch", methods=["GET", "POST"])
@login_required
def batch_predict():
    if request.method == "POST":
        if "csv_file" not in request.files or request.files["csv_file"].filename == "":
            flash("Please select a CSV file to upload.", "danger")
            return render_template("batch.html")

        file = request.files["csv_file"]
        if not file.filename.lower().endswith(".csv"):
            flash("Only .csv files are accepted.", "danger")
            return render_template("batch.html")

        result = predict_batch(file)

        if "error" in result:
            flash(result["error"], "danger")
            return render_template("batch.html")

        # Save results CSV to a temp file for download
        results_df   = result["results_df"]
        errors_list  = result["errors"]
        valid_count  = result["valid_count"]
        total        = result["total"]

        csv_bytes = results_df.to_csv(index=False).encode("utf-8") if not results_df.empty else b""

        # Store CSV in session temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv",
                                          dir=tempfile.gettempdir(), mode="wb")
        tmp.write(csv_bytes)
        tmp.close()
        session["batch_result_file"] = tmp.name

        logger.info("Batch prediction complete: %d/%d valid", valid_count, total)
        flash(f"Processed {total} rows — {valid_count} valid, {len(errors_list)} with errors.", "info")

        return render_template(
            "batch.html",
            results_df  = results_df,
            errors_list = errors_list,
            valid_count = valid_count,
            total       = total,
            has_results = not results_df.empty,
        )

    return render_template("batch.html")


@app.route("/download/<path:filename>")
@login_required
def download_file(filename):
    """Download the batch result CSV."""
    filepath = session.get("batch_result_file", "")
    if not filepath or not os.path.exists(filepath):
        flash("No result file available. Run a batch prediction first.", "warning")
        return redirect(url_for("batch_predict"))

    return send_file(
        filepath,
        mimetype="text/csv",
        as_attachment=True,
        download_name="batch_prediction_results.csv",
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(e):
    logger.error("500 error: %s", e)
    return render_template("500.html"), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
