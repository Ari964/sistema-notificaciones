import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

CATEGORICAL_COLS = ["Gender", "Medication_Type", "Education_Level", "Income_Level",
                     "Social_Support", "Disease_Severity", "Insurance_Coverage"]
NUMERICAL_COLS = ["Age", "Dosage", "Num_Comorbidities", "Previous_Adherence"]
ENGINEERED_NUM_COLS = [
    "polypharmacy_index", "age_dosage_ratio", "social_risk_score",
    "age_comorbidity_interaction", "severity_numeric",
    "estimated_adherence_score", "frailty_proxy",
]
ALL_NUM_COLS = NUMERICAL_COLS + ENGINEERED_NUM_COLS
ALL_FEATURES = CATEGORICAL_COLS + ALL_NUM_COLS

API_TO_CSV = {
    "gender": "Gender", "medication_type": "Medication_Type",
    "education_level": "Education_Level", "income_level": "Income_Level",
    "social_support": "Social_Support", "disease_severity": "Disease_Severity",
    "insurance_coverage": "Insurance_Coverage",
    "age": "Age", "dosage": "Dosage",
    "num_comorbidities": "Num_Comorbidities", "previous_adherence": "Previous_Adherence",
}

SUPPORT_MAP = {"Nulo": 0, "Bajo": 1, "Moderado": 2, "Alto": 3}
INCOME_MAP = {"Bajo": 0, "Medio-Bajo": 1, "Medio": 2, "Medio-Alto": 3}
EDU_MAP = {"Sin educación formal": 0, "Primaria": 1, "Secundaria": 2, "Superior": 3}
SEVERITY_MAP = {"Leve": 1, "Moderada": 2, "Severa": 3, "Muy severa": 4}


def create_advanced_features(df):
    df = df.copy()

    df["polypharmacy_index"] = df["Num_Comorbidities"] * df["Dosage"] / df["Age"]
    df["age_dosage_ratio"] = df["Age"] / (df["Dosage"] + 1)

    df["social_risk_score"] = (
        df["Social_Support"].map(SUPPORT_MAP).fillna(1) * 0.4 +
        df["Income_Level"].map(INCOME_MAP).fillna(1) * 0.35 +
        df["Education_Level"].map(EDU_MAP).fillna(1) * 0.25
    )

    df["age_comorbidity_interaction"] = df["Age"] * df["Num_Comorbidities"]
    df["severity_numeric"] = df["Disease_Severity"].map(SEVERITY_MAP).fillna(2)

    df["estimated_adherence_score"] = (
        df["Previous_Adherence"] * 0.4 +
        df["severity_numeric"].clip(upper=4) / 4 * 0.3 +
        (df["Num_Comorbidities"] <= 2).astype(int) * 0.3
    )

    df["frailty_proxy"] = pd.cut(
        df["Age"], bins=[59, 69, 79, 89, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    return df


def load_encoders():
    if os.path.exists(ENCODERS_PATH):
        return joblib.load(ENCODERS_PATH)
    return None


def load_scaler():
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def fit_encoders(df: pd.DataFrame):
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(encoders, ENCODERS_PATH)
    return encoders


def fit_scaler(df: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(df[ALL_NUM_COLS].values)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def preprocess_training(df: pd.DataFrame):
    df_feat = create_advanced_features(df)
    encoders = fit_encoders(df_feat)
    scaler = fit_scaler(df_feat)

    df_processed = df_feat.copy()
    for col in CATEGORICAL_COLS:
        df_processed[col] = encoders[col].transform(df_feat[col].astype(str))

    df_processed[ALL_NUM_COLS] = scaler.transform(df_feat[ALL_NUM_COLS].values)

    X = df_processed[ALL_FEATURES].values
    y = df_processed["Adherence"].values if "Adherence" in df_processed.columns else None
    return X, y


def preprocess_single_patient(data: dict):
    encoders = load_encoders()
    scaler = load_scaler()

    if encoders is None or scaler is None:
        raise ValueError("Los encoders y scaler no estan entrenados. Ejecute primero train.py")

    row = {}
    for csv_col, api_key in {
        "Age": "age", "Dosage": "dosage",
        "Num_Comorbidities": "num_comorbidities", "Previous_Adherence": "previous_adherence",
        "Gender": "gender", "Medication_Type": "medication_type",
        "Education_Level": "education_level", "Income_Level": "income_level",
        "Social_Support": "social_support", "Disease_Severity": "disease_severity",
        "Insurance_Coverage": "insurance_coverage",
    }.items():
        row[csv_col] = data.get(api_key, data.get(csv_col, 0))

    row_df = pd.DataFrame([row])
    row_feat = create_advanced_features(row_df)

    processed = {}
    for col in CATEGORICAL_COLS:
        val = str(row_feat[col].iloc[0])
        if val in encoders[col].classes_:
            processed[col] = encoders[col].transform([val])[0]
        else:
            processed[col] = 0

    num_values = row_feat[ALL_NUM_COLS].values
    num_scaled = scaler.transform(num_values)[0]

    feature_vector = np.array(
        [processed[col] for col in CATEGORICAL_COLS] + list(num_scaled)
    )
    return feature_vector.reshape(1, -1, 1).astype(np.float32)
