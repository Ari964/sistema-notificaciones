import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
COMPARISON_PATH = os.path.join(MODELS_DIR, "ensemble_comparison.json")


def get_ensemble_models():
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, force_col_wise=True
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            early_stopping=True, random_state=42
        ),
    }


def run_ensemble_comparison(df):
    from sklearn.base import clone
    from backend.preprocessing import create_advanced_features

    df_feat = create_advanced_features(df)

    cat_cols = ["Gender", "Medication_Type", "Education_Level", "Income_Level",
                "Social_Support", "Disease_Severity", "Insurance_Coverage"]
    df_encoded = pd.get_dummies(df_feat, columns=cat_cols, drop_first=True)

    feature_cols = [c for c in df_encoded.columns if c != "Adherence"]
    X = df_encoded[feature_cols].fillna(0).astype(np.float64).values
    y = df_encoded["Adherence"].values.astype(np.int32)

    smote = SMOTE(random_state=42)

    print("\n" + "=" * 60)
    print("  COMPARACION DE MODELOS DE ENSAMBLE")
    print("=" * 60)

    results = {}

    for name, model in get_ensemble_models().items():
        print(f"\n  Evaluando: {name}...")
        try:
            accs, precs, recs, f1s, aucs = [], [], [], [], []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                X_res, y_res = smote.fit_resample(X_train, y_train)
                X_res = X_res.astype(np.float64)
                y_res = y_res.astype(np.int32)

                m = clone(model)
                m.fit(X_res, y_res)
                y_pred = m.predict(X_val)
                y_prob = m.predict_proba(X_val)[:, 1] if hasattr(m, "predict_proba") else y_pred.astype(float)

                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                accs.append(accuracy_score(y_val, y_pred))
                precs.append(precision_score(y_val, y_pred, zero_division=0))
                recs.append(recall_score(y_val, y_pred, zero_division=0))
                f1s.append(f1_score(y_val, y_pred, zero_division=0))
                aucs.append(roc_auc_score(y_val, y_prob))

            results[name] = {
                "accuracy": round(float(np.mean(accs)), 4),
                "precision": round(float(np.mean(precs)), 4),
                "recall": round(float(np.mean(recs)), 4),
                "f1_score": round(float(np.mean(f1s)), 4),
                "auc_roc": round(float(np.mean(aucs)), 4),
                "accuracy_std": round(float(np.std(accs)), 4),
            }
            print(f"    Acc: {results[name]['accuracy']:.4f} "
                  f"(+/-{results[name]['accuracy_std']:.4f}) | "
                  f"AUC: {results[name]['auc_roc']:.4f} | "
                  f"F1: {results[name]['f1_score']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"error": str(e)}

    best_name = max(
        [k for k in results if "error" not in results[k]],
        key=lambda k: results[k].get("auc_roc", 0),
        default="N/A"
    )
    results["best_ensemble_model"] = best_name

    print(f"\n  Mejor modelo de ensamble: {best_name} "
          f"(AUC-ROC: {results[best_name].get('auc_roc', 0):.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(COMPARISON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    return results


def get_ensemble_comparison():
    if os.path.exists(COMPARISON_PATH):
        with open(COMPARISON_PATH, "r") as f:
            return json.load(f)
    return None
