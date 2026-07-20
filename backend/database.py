import sqlite3
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "patients.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            medication_type TEXT NOT NULL,
            dosage REAL NOT NULL,
            education_level TEXT NOT NULL,
            income_level TEXT NOT NULL,
            social_support TEXT NOT NULL,
            disease_severity TEXT NOT NULL,
            num_comorbidities INTEGER NOT NULL,
            insurance_coverage TEXT NOT NULL,
            previous_adherence INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            adherence_probability REAL NOT NULL,
            adherence_class TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            notification_strategy TEXT NOT NULL,
            shap_values TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            prediction_id INTEGER NOT NULL,
            notification_type TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT DEFAULT 'pendiente',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id),
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    """)
    conn.commit()
    conn.close()


def insert_patient(data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO patients (name, age, gender, medication_type, dosage,
            education_level, income_level, social_support, disease_severity,
            num_comorbidities, insurance_coverage, previous_adherence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["name"], data["age"], data["gender"], data["medication_type"],
        data["dosage"], data["education_level"], data["income_level"],
        data["social_support"], data["disease_severity"],
        data["num_comorbidities"], data["insurance_coverage"],
        data["previous_adherence"]
    ))
    conn.commit()
    patient_id = cursor.lastrowid
    conn.close()
    return patient_id


def insert_prediction(patient_id: int, adherence_prob: float, adherence_class: str,
                      risk_level: str, notification_strategy: str, shap_values: str = None) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (patient_id, adherence_probability, adherence_class,
            risk_level, notification_strategy, shap_values)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (patient_id, adherence_prob, adherence_class, risk_level,
          notification_strategy, shap_values))
    conn.commit()
    prediction_id = cursor.lastrowid
    conn.close()
    return prediction_id


def insert_notification(patient_id: int, prediction_id: int,
                        notification_type: str, message: str) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO notifications (patient_id, prediction_id, notification_type, message)
        VALUES (?, ?, ?, ?)
    """, (patient_id, prediction_id, notification_type, message))
    conn.commit()
    notification_id = cursor.lastrowid
    conn.close()
    return notification_id


def get_all_patients():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_patient_by_id(patient_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_predictions_by_patient(patient_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_notifications_by_patient(patient_id: int):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM notifications WHERE patient_id = ? ORDER BY created_at DESC", (patient_id,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_dashboard_stats():
    conn = get_connection()
    cursor = conn.cursor()
    stats = {}

    cursor.execute("SELECT COUNT(*) as total FROM patients")
    stats["total_patients"] = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    stats["total_predictions"] = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as total FROM notifications WHERE status = 'pendiente'")
    stats["pending_notifications"] = cursor.fetchone()["total"]

    cursor.execute("""
        SELECT risk_level, COUNT(*) as count FROM predictions
        GROUP BY risk_level
    """)
    stats["risk_distribution"] = {row["risk_level"]: row["count"] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT adherence_class, COUNT(*) as count FROM predictions
        GROUP BY adherence_class
    """)
    stats["adherence_distribution"] = {row["adherence_class"]: row["count"] for row in cursor.fetchall()}

    conn.close()
    return stats


def get_all_predictions():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.*, pt.name as patient_name
        FROM predictions p
        JOIN patients pt ON p.patient_id = pt.id
        ORDER BY p.created_at DESC
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


init_db()
