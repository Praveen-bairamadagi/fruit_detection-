import os
import sqlite3
import time
from functools import wraps

import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "fruit_detection_ai_secret_key_2026"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fruit_detection_model.keras"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "fruit_detection.db"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# ALLOWED IMAGE EXTENSIONS
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


# ============================================================
# FRUIT CLASSES
# ============================================================

class_names = [
    "apple",
    "banana",
    "grape",
    "mango",
    "orange"
]


# ============================================================
# LOAD TENSORFLOW MODEL
# ============================================================

model = None

if os.path.exists(MODEL_PATH):

    try:

        model = load_model(MODEL_PATH)

        print("==============================================")
        print("TensorFlow model loaded successfully.")
        print("Model:", MODEL_PATH)
        print("==============================================")

    except Exception as error:

        print("==============================================")
        print("MODEL LOADING ERROR")
        print(error)
        print("==============================================")

else:

    print("==============================================")
    print("WARNING: Model file not found.")
    print("Please train the model first.")
    print("Expected:", MODEL_PATH)
    print("==============================================")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_database():

    connection = get_db_connection()

    cursor = connection.cursor()

    # ========================================================
    # CUSTOMERS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ========================================================
    # PREDICTIONS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER,
            image_name TEXT,
            predicted_fruit TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(customer_id)
            REFERENCES customers(id)
        )
    """)

    # ========================================================
    # ADMINS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # ========================================================
    # REPAIR OLD PREDICTIONS TABLE
    # ========================================================

    columns = cursor.execute(
        "PRAGMA table_info(predictions)"
    ).fetchall()

    column_names = [
        column["name"]
        for column in columns
    ]

    if "customer_id" not in column_names:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN customer_id INTEGER
        """)

    if "image_name" not in column_names:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN image_name TEXT DEFAULT ''
        """)

    if "predicted_fruit" not in column_names:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN predicted_fruit TEXT DEFAULT ''
        """)

    if "confidence" not in column_names:

        cursor.execute("""
            ALTER TABLE predictions
            ADD COLUMN confidence REAL DEFAULT 0
        """)

    # ========================================================
    # CREATE DEFAULT ADMIN
    # ========================================================

    admin = cursor.execute("""
        SELECT id
        FROM admins
        WHERE username = ?
    """, (
        "admin",
    )).fetchone()

    if admin is None:

        cursor.execute("""
            INSERT INTO admins
            (
                username,
                password
            )
            VALUES (?, ?)
        """, (
            "admin",
            "admin123"
        ))

        print("----------------------------------------------")
        print("Default admin created")
        print("Username: admin")
        print("Password: admin123")
        print("----------------------------------------------")

    connection.commit()

    connection.close()


# ============================================================
# CUSTOMER LOGIN REQUIRED
# ============================================================

def customer_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "customer_id" not in session:

            flash(
                "Please login as a customer first.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        connection = get_db_connection()

        customer = connection.execute("""
            SELECT *
            FROM customers
            WHERE id = ?
        """, (
            session["customer_id"],
        )).fetchone()

        connection.close()

        if customer is None:

            session.pop(
                "customer_id",
                None
            )

            session.pop(
                "customer_name",
                None
            )

            session.pop(
                "customer_email",
                None
            )

            flash(
                "Customer account not found. Please login again.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper


# ============================================================
# ADMIN LOGIN REQUIRED
# ============================================================

def admin_required(function):

    @wraps(function)
    def wrapper(*args, **kwargs):

        if "admin_id" not in session:

            flash(
                "Please login as admin first.",
                "warning"
            )

            return redirect(
                url_for("admin_login")
            )

        connection = get_db_connection()

        admin = connection.execute("""
            SELECT *
            FROM admins
            WHERE id = ?
        """, (
            session["admin_id"],
        )).fetchone()

        connection.close()

        if admin is None:

            session.pop(
                "admin_id",
                None
            )

            session.pop(
                "admin_username",
                None
            )

            flash(
                "Admin session expired. Please login again.",
                "warning"
            )

            return redirect(
                url_for("admin_login")
            )

        return function(
            *args,
            **kwargs
        )

    return wrapper


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ============================================================
# CUSTOMER REGISTER
#
# IMPORTANT:
# endpoint="register" means:
#
# url_for("register")
#
# will work.
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"],
    endpoint="register"
)
def customer_register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        ).strip()

        confirm_password = request.form.get(
            "confirm_password",
            ""
        ).strip()

        # ====================================================
        # VALIDATION
        # ====================================================

        if not name or not email or not password:

            flash(
                "Please fill all fields.",
                "danger"
            )

            return render_template(
                "customer/register.html"
            )

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return render_template(
                "customer/register.html"
            )

        if len(password) < 4:

            flash(
                "Password must contain at least 4 characters.",
                "danger"
            )

            return render_template(
                "customer/register.html"
            )

        # ====================================================
        # CHECK EXISTING CUSTOMER
        # ====================================================

        connection = get_db_connection()

        existing_customer = connection.execute("""
            SELECT id
            FROM customers
            WHERE email = ?
        """, (
            email,
        )).fetchone()

        if existing_customer:

            connection.close()

            flash(
                "Email already registered. Please login.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        # ====================================================
        # CREATE CUSTOMER
        # ====================================================

        try:

            connection.execute("""
                INSERT INTO customers
                (
                    name,
                    email,
                    password
                )
                VALUES (?, ?, ?)
            """, (
                name,
                email,
                password
            ))

            connection.commit()

            connection.close()

        except sqlite3.IntegrityError:

            connection.close()

            flash(
                "This email is already registered.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        # ====================================================
        # SUCCESS
        # ====================================================

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "customer/register.html"
    )


# ============================================================
# CUSTOMER REGISTER ALIAS
#
# Supports:
#
# url_for("customer_register")
#
# ============================================================

app.add_url_rule(
    "/register",
    endpoint="customer_register",
    view_func=customer_register,
    methods=["GET", "POST"]
)


# ============================================================
# CUSTOMER LOGIN
#
# endpoint="login" makes:
#
# url_for("login")
#
# work.
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"],
    endpoint="login"
)
def customer_login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        ).strip()

        if not email or not password:

            flash(
                "Please enter email and password.",
                "warning"
            )

            return render_template(
                "customer/login.html"
            )

        connection = get_db_connection()

        customer = connection.execute("""
            SELECT *
            FROM customers
            WHERE email = ?
            AND password = ?
        """, (
            email,
            password
        )).fetchone()

        connection.close()

        if customer:

            session["customer_id"] = customer["id"]

            session["customer_name"] = customer["name"]

            session["customer_email"] = customer["email"]

            flash(
                "Welcome, " +
                customer["name"] +
                "!",
                "success"
            )

            return redirect(
                url_for("customer_dashboard")
            )

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template(
        "customer/login.html"
    )


# ============================================================
# CUSTOMER LOGIN ALIAS
#
# Supports:
#
# url_for("customer_login")
#
# ============================================================

app.add_url_rule(
    "/login",
    endpoint="customer_login",
    view_func=customer_login,
    methods=["GET", "POST"]
)


# ============================================================
# CUSTOMER DASHBOARD
# ============================================================

@app.route("/dashboard")
@customer_required
def customer_dashboard():

    customer_id = session["customer_id"]

    connection = get_db_connection()

    customer = connection.execute("""
        SELECT *
        FROM customers
        WHERE id = ?
    """, (
        customer_id,
    )).fetchone()

    prediction_count = connection.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
        WHERE customer_id = ?
    """, (
        customer_id,
    )).fetchone()["total"]

    recent_predictions = connection.execute("""
        SELECT *
        FROM predictions
        WHERE customer_id = ?
        ORDER BY id DESC
        LIMIT 5
    """, (
        customer_id,
    )).fetchall()

    connection.close()

    return render_template(
        "customer/dashboard.html",
        customer=customer,
        prediction_count=prediction_count,
        recent_predictions=recent_predictions
    )


# ============================================================
# CUSTOMER PROFILE
# ============================================================

@app.route("/profile")
@customer_required
def customer_profile():

    customer_id = session["customer_id"]

    connection = get_db_connection()

    customer = connection.execute("""
        SELECT *
        FROM customers
        WHERE id = ?
    """, (
        customer_id,
    )).fetchone()

    connection.close()

    return render_template(
        "customer/profile.html",
        customer=customer
    )


# ============================================================
# CUSTOMER HISTORY
# ============================================================

@app.route("/history")
@customer_required
def customer_history():

    customer_id = session["customer_id"]

    connection = get_db_connection()

    predictions = connection.execute("""
        SELECT *
        FROM predictions
        WHERE customer_id = ?
        ORDER BY id DESC
    """, (
        customer_id,
    )).fetchall()

    connection.close()

    return render_template(
        "customer/history.html",
        predictions=predictions
    )


# ============================================================
# FRUIT PREDICTION
#
# endpoint="predict" makes:
#
# url_for("predict")
#
# work.
# ============================================================

@app.route(
    "/predict",
    methods=["GET", "POST"],
    endpoint="predict"
)
@customer_required
def predict():

    prediction = None

    confidence = None

    image_url = None

    # ========================================================
    # GET REQUEST
    # ========================================================

    if request.method == "GET":

        return render_template(
            "customer/prediction.html",
            prediction=None,
            confidence=None,
            image_url=None
        )

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if model is None:

        flash(
            "Machine learning model is not available. Please train the model first.",
            "danger"
        )

        return redirect(
            url_for("customer_dashboard")
        )

    # ========================================================
    # GET IMAGE
    # ========================================================

    uploaded_file = request.files.get(
        "image"
    )

    if uploaded_file is None:

        flash(
            "Please select an image.",
            "warning"
        )

        return redirect(
            url_for("predict")
        )

    if uploaded_file.filename == "":

        flash(
            "Please select an image.",
            "warning"
        )

        return redirect(
            url_for("predict")
        )

    # ========================================================
    # SECURE FILE NAME
    # ========================================================

    filename = secure_filename(
        uploaded_file.filename
    )

    if not filename:

        flash(
            "Invalid image file.",
            "danger"
        )

        return redirect(
            url_for("predict")
        )

    # ========================================================
    # CHECK EXTENSION
    # ========================================================

    if "." not in filename:

        flash(
            "Invalid image file.",
            "danger"
        )

        return redirect(
            url_for("predict")
        )

    extension = filename.rsplit(
        ".",
        1
    )[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:

        flash(
            "Only JPG, JPEG and PNG images are allowed.",
            "danger"
        )

        return redirect(
            url_for("predict")
        )

    # ========================================================
    # UNIQUE FILE NAME
    # ========================================================

    unique_filename = (
        str(int(time.time() * 1000))
        + "_"
        + filename
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    uploaded_file.save(
        file_path
    )

    # ========================================================
    # MACHINE LEARNING PREDICTION
    # ========================================================

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        img = image.load_img(
            file_path,
            target_size=(128, 128)
        )

        # ----------------------------------------------------
        # CONVERT IMAGE TO ARRAY
        # ----------------------------------------------------

        img_array = image.img_to_array(
            img
        )

        # ----------------------------------------------------
        # ADD BATCH DIMENSION
        # ----------------------------------------------------

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # ----------------------------------------------------
        # NORMALIZE PIXELS
        # ----------------------------------------------------

        img_array = img_array / 255.0

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            img_array,
            verbose=0
        )

        # ----------------------------------------------------
        # FIND HIGHEST PROBABILITY
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )

        # ----------------------------------------------------
        # GET FRUIT NAME
        # ----------------------------------------------------

        prediction = class_names[
            predicted_index
        ]

        # ----------------------------------------------------
        # GET CONFIDENCE
        # ----------------------------------------------------

        confidence = float(
            predictions[0][predicted_index] * 100
        )

        # ====================================================
        # SAVE PREDICTION TO DATABASE
        # ====================================================

        connection = get_db_connection()

        connection.execute("""
            INSERT INTO predictions
            (
                customer_id,
                image_name,
                predicted_fruit,
                confidence
            )
            VALUES (?, ?, ?, ?)
        """, (
            session["customer_id"],
            unique_filename,
            prediction,
            confidence
        ))

        connection.commit()

        connection.close()

        # ====================================================
        # IMAGE URL
        # ====================================================

        image_url = url_for(
            "static",
            filename="uploads/" + unique_filename
        )

        # ====================================================
        # TERMINAL OUTPUT
        # ====================================================

        print("----------------------------------------------")

        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )

        print(
            "Image:",
            unique_filename
        )

        print("----------------------------------------------")

    except Exception as error:

        print("----------------------------------------------")

        print(
            "PREDICTION ERROR"
        )

        print(
            error
        )

        print("----------------------------------------------")

        if os.path.exists(file_path):

            os.remove(file_path)

        flash(
            "Error while predicting the image.",
            "danger"
        )

        return redirect(
            url_for("predict")
        )

    # ========================================================
    # SHOW RESULT
    # ========================================================

    return render_template(
        "customer/prediction.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url
    )


# ============================================================
# PREDICTION ALIAS
#
# Supports old templates using:
#
# url_for("prediction")
#
# ============================================================

app.add_url_rule(
    "/predict",
    endpoint="prediction",
    view_func=predict,
    methods=["GET", "POST"]
)


# ============================================================
# CUSTOMER LOGOUT
# ============================================================

@app.route("/logout")
def customer_logout():

    session.pop(
        "customer_id",
        None
    )

    session.pop(
        "customer_name",
        None
    )

    session.pop(
        "customer_email",
        None
    )

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# ADMIN LOGIN
# ============================================================

@app.route(
    "/admin/login",
    methods=["GET", "POST"]
)
def admin_login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        connection = get_db_connection()

        admin = connection.execute("""
            SELECT *
            FROM admins
            WHERE username = ?
            AND password = ?
        """, (
            username,
            password
        )).fetchone()

        connection.close()

        if admin:

            session["admin_id"] = admin["id"]

            session["admin_username"] = admin["username"]

            flash(
                "Admin login successful.",
                "success"
            )

            return redirect(
                url_for("admin_dashboard")
            )

        flash(
            "Invalid admin username or password.",
            "danger"
        )

    return render_template(
        "admin/login.html"
    )


# ============================================================
# ADMIN DASHBOARD
# ============================================================

@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():

    connection = get_db_connection()

    total_customers = connection.execute("""
        SELECT COUNT(*) AS total
        FROM customers
    """).fetchone()["total"]

    total_predictions = connection.execute("""
        SELECT COUNT(*) AS total
        FROM predictions
    """).fetchone()["total"]

    total_admins = connection.execute("""
        SELECT COUNT(*) AS total
        FROM admins
    """).fetchone()["total"]

    recent_predictions = connection.execute("""
        SELECT
            predictions.*,
            customers.name AS customer_name,
            customers.email AS customer_email
        FROM predictions
        LEFT JOIN customers
        ON predictions.customer_id = customers.id
        ORDER BY predictions.id DESC
        LIMIT 10
    """).fetchall()

    connection.close()

    return render_template(
        "admin/dashboard.html",
        total_customers=total_customers,
        total_predictions=total_predictions,
        total_admins=total_admins,
        recent_predictions=recent_predictions
    )


# ============================================================
# ADMIN CUSTOMERS
# ============================================================

@app.route("/admin/customers")
@admin_required
def admin_customers():

    connection = get_db_connection()

    customers = connection.execute("""
        SELECT
            customers.*,
            (
                SELECT COUNT(*)
                FROM predictions
                WHERE predictions.customer_id = customers.id
            ) AS prediction_count
        FROM customers
        ORDER BY customers.id DESC
    """).fetchall()

    connection.close()

    return render_template(
        "admin/customers.html",
        customers=customers
    )


# ============================================================
# ADMIN ALL PREDICTIONS
# ============================================================

@app.route("/admin/predictions")
@admin_required
def admin_predictions():

    connection = get_db_connection()

    predictions = connection.execute("""
        SELECT
            predictions.*,
            customers.name AS customer_name,
            customers.email AS customer_email
        FROM predictions
        LEFT JOIN customers
        ON predictions.customer_id = customers.id
        ORDER BY predictions.id DESC
    """).fetchall()

    connection.close()

    return render_template(
        "admin/predictions.html",
        predictions=predictions
    )


# ============================================================
# ADMIN LOGOUT
# ============================================================

@app.route("/admin/logout")
def admin_logout():

    session.pop(
        "admin_id",
        None
    )

    session.pop(
        "admin_username",
        None
    )

    flash(
        "Admin logged out successfully.",
        "success"
    )

    return redirect(
        url_for("home")
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_database()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print("")
    print("==============================================")
    print("          FRUIT DETECTION AI")
    print("==============================================")

    print("")
    print("Customer Home:")
    print("http://127.0.0.1:5000/")

    print("")
    print("Customer Login:")
    print("http://127.0.0.1:5000/login")

    print("")
    print("Customer Register:")
    print("http://127.0.0.1:5000/register")

    print("")
    print("Customer Dashboard:")
    print("http://127.0.0.1:5000/dashboard")

    print("")
    print("Fruit Prediction:")
    print("http://127.0.0.1:5000/predict")

    print("")
    print("Customer Profile:")
    print("http://127.0.0.1:5000/profile")

    print("")
    print("Customer History:")
    print("http://127.0.0.1:5000/history")

    print("")
    print("Admin Login:")
    print("http://127.0.0.1:5000/admin/login")

    print("")
    print("Admin Username: admin")
    print("Admin Password: admin123")

    print("==============================================")
    print("")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )