"""
MySQL Database Integration for Smart Recycling Assistant.
Handles database initialization, dataset catalog logging, and prediction history tracking.
"""

import os
from datetime import datetime
import pandas as pd
import mysql.connector
from mysql.connector import Error

# Default MySQL configuration (can be overridden via environment variables or Streamlit UI)
DB_CONFIG = {
    "host": os.environ.get("MYSQL_HOST", "localhost"),
    "port": int(os.environ.get("MYSQL_PORT", 3306)),
    "user": os.environ.get("MYSQL_USER", "recycling_user"),
    "password": os.environ.get("MYSQL_PASSWORD", "recycling_pass123"),
    "database": os.environ.get("MYSQL_DB", "smart_recycling_db"),
}


def get_server_connection(config=None):
    """Connects to the MySQL server without specifying a database first."""
    cfg = config or DB_CONFIG
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
    )


def get_db_connection(config=None):
    """Connects directly to the smart_recycling_db database."""
    cfg = config or DB_CONFIG
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        database=cfg["database"],
    )


def init_database(config=None):
    """
    Creates the MySQL database and required tables if they do not exist:
    1. waste_images_catalog: Stores training/validation image metadata and class labels.
    2. prediction_history: Logs every image classified in the Streamlit app.
    """
    cfg = config or DB_CONFIG
    conn = None
    try:
        conn = get_server_connection(cfg)
        cursor = conn.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {cfg['database']}")
        cursor.execute(f"USE {cfg['database']}")

        # Table 1: Training & Reference Image Catalog
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS waste_images_catalog (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_filename VARCHAR(255) NOT NULL,
                image_filepath VARCHAR(500) NOT NULL,
                material_label ENUM('glass', 'metal', 'paper', 'plastic') NOT NULL,
                dataset_split ENUM('train', 'val') DEFAULT 'train',
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Table 2: Live Prediction History Log
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_name VARCHAR(255) NOT NULL,
                predicted_class ENUM('glass', 'metal', 'paper', 'plastic') NOT NULL,
                confidence_pct DECIMAL(5, 2) NOT NULL,
                bin_recommendation VARCHAR(255) NOT NULL,
                predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.commit()
        cursor.close()
        return True, f"Database '{cfg['database']}' and tables initialized."
    except Error as err:
        return False, f"MySQL Error: {err}"
    finally:
        if conn and conn.is_connected():
            conn.close()


def sync_dataset_to_mysql(dataset_dir="dataset", config=None):
    """
    Scans dataset/<class_name>/ folders and logs image metadata into waste_images_catalog.
    """
    success, msg = init_database(config)
    if not success:
        return False, msg

    valid_classes = ("glass", "metal", "paper", "plastic")
    rows_to_insert = []

    for label in valid_classes:
        class_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        split_idx = int(len(files) * 0.8)
        for idx, fname in enumerate(files):
            split_name = "train" if idx < split_idx else "val"
            fpath = os.path.abspath(os.path.join(class_dir, fname))
            rows_to_insert.append((fname, fpath, label, split_name))

    if not rows_to_insert:
        return False, "No images found in dataset/ directory to register."

    conn = None
    try:
        conn = get_db_connection(config)
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE waste_images_catalog")
        cursor.executemany(
            """
            INSERT INTO waste_images_catalog
                (image_filename, image_filepath, material_label, dataset_split)
            VALUES (%s, %s, %s, %s)
            """,
            rows_to_insert,
        )
        conn.commit()
        cursor.close()
        return True, f"Registered {len(rows_to_insert)} waste images into MySQL."
    except Error as err:
        return False, f"MySQL Insert Error: {err}"
    finally:
        if conn and conn.is_connected():
            conn.close()


def log_prediction(image_name, predicted_class, confidence_pct, bin_recommendation, config=None):
    """Inserts a new waste classification record into prediction_history."""
    conn = None
    try:
        conn = get_db_connection(config)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_history
                (image_name, predicted_class, confidence_pct, bin_recommendation)
            VALUES (%s, %s, %s, %s)
            """,
            (image_name, predicted_class.lower(), round(float(confidence_pct), 2), bin_recommendation),
        )
        conn.commit()
        cursor.close()
        return True, "Logged prediction to MySQL."
    except Error as err:
        return False, f"Could not log to MySQL: {err}"
    finally:
        if conn and conn.is_connected():
            conn.close()


def fetch_prediction_history(limit=50, config=None):
    """Returns recent predictions from MySQL as a pandas DataFrame."""
    conn = None
    try:
        conn = get_db_connection(config)
        query = """
            SELECT id, image_name, predicted_class, confidence_pct,
                   bin_recommendation, predicted_at
            FROM prediction_history
            ORDER BY predicted_at DESC
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
        return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "id",
                "image_name",
                "predicted_class",
                "confidence_pct",
                "bin_recommendation",
                "predicted_at",
            ]
        )
    finally:
        if conn and conn.is_connected():
            conn.close()


def fetch_catalog_summary(config=None):
    """Returns class counts from waste_images_catalog in MySQL."""
    conn = None
    try:
        conn = get_db_connection(config)
        query = """
            SELECT material_label AS Material,
                   dataset_split AS Split,
                   COUNT(*) AS Image_Count
            FROM waste_images_catalog
            GROUP BY material_label, dataset_split
            ORDER BY material_label, dataset_split
        """
        return pd.read_sql(query, conn)
    except Exception:
        return pd.DataFrame(columns=["Material", "Split", "Image_Count"])
    finally:
        if conn and conn.is_connected():
            conn.close()


if __name__ == "__main__":
    ok, message = init_database()
    print(message)