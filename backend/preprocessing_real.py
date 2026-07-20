import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
REAL_SCALER_PATH = os.path.join(MODELS_DIR, "real_scaler.pkl")
REAL_FEATURES_PATH = os.path.join(MODELS_DIR, "real_features.json")

REAL_NUMERIC_COLS = ["AGE", "ANNUALCONTRIBUTION", "ANNUALCLAIMAMOUNT", "UNITSTOTAL"]
REAL_BOOL_COLS = ["GENDER_M", "SCHEMETYPE_MEDIUM", "SCHEMETYPE_PREMIUM",
                   "DIAGNOSIS_HYPERTENSION", "COVERTYPE_STANDARD",
                   "COMORBIDITY_NO_COMORBIDITY", "COMPLICATIONDEVELOPMENT_NO_COMPLICATION"]


def create_real_features(df):
    df = df.copy()
    df["claim_to_contribution_ratio"] = df["ANNUALCLAIMAMOUNT"] / (df["ANNUALCONTRIBUTION"] + 1)
    df["units_per_claim"] = df["UNITSTOTAL"] / (df["ANNUALCLAIMAMOUNT"] + 1)
    df["age_group"] = pd.cut(df["AGE"], bins=[0, 35, 50, 65, 80, 120], labels=[0, 1, 2, 3, 4]).astype(int)
    df["cost_burden"] = df["ANNUALCONTRIBUTION"] / (df["UNITSTOTAL"] + 1)
    df["total_risk_score"] = (
        df["DIAGNOSIS_HYPERTENSION"].astype(int) * 0.3 +
        (1 - df["COMORBIDITY_NO_COMORBIDITY"].astype(int)) * 0.3 +
        (1 - df["COMPLICATIONDEVELOPMENT_NO_COMPLICATION"].astype(int)) * 0.2 +
        df["GENDER_M"].astype(int) * 0.1 +
        (df["AGE"] > 60).astype(int) * 0.1
    )
    return df


def load_real_scaler():
    if os.path.exists(REAL_SCALER_PATH):
        return joblib.load(REAL_SCALER_PATH)
    return None


def fit_real_scaler(df):
    scaler = StandardScaler()
    feature_cols = get_feature_columns(df)
    num_cols = [c for c in feature_cols if c not in REAL_BOOL_COLS]
    scaler.fit(df[num_cols].values)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, REAL_SCALER_PATH)
    return scaler


def get_feature_columns(df=None):
    base = REAL_NUMERIC_COLS + REAL_BOOL_COLS
    engineered = ["claim_to_contribution_ratio", "units_per_claim", "age_group",
                   "cost_burden", "total_risk_score"]
    if df is not None:
        return base + [c for c in engineered if c in df.columns]
    return base + engineered


def preprocess_real_training(df):
    df_feat = create_real_features(df)
    scaler = fit_real_scaler(df_feat)

    feature_cols = get_feature_columns(df_feat)
    num_cols = [c for c in feature_cols if c not in REAL_BOOL_COLS]

    df_processed = df_feat.copy()
    df_processed[num_cols] = scaler.transform(df_feat[num_cols].values)

    for col in REAL_BOOL_COLS:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(int)

    X = df_processed[feature_cols].values
    y = None
    if "ADHERENCE" in df.columns:
        y = (df["ADHERENCE"] == "ADHERENT").astype(int).values

    return X, y


def preprocess_real_single(data: dict):
    scaler = load_real_scaler()
    if scaler is None:
        raise ValueError("Scaler no entrenado. Ejecute entrenamiento con datos reales primero.")

    row = pd.DataFrame([data])
    row_feat = create_real_features(row)

    feature_cols = get_feature_columns(row_feat)
    num_cols = [c for c in feature_cols if c not in REAL_BOOL_COLS]

    processed = row_feat.copy()
    processed[num_cols] = scaler.transform(row_feat[num_cols].values)

    for col in REAL_BOOL_COLS:
        if col in processed.columns:
            processed[col] = processed[col].astype(int)

    X = processed[feature_cols].values
    return X.reshape(1, -1, 1).astype(np.float32)
