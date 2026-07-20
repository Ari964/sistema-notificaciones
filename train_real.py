import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from backend.preprocessing_real import preprocess_real_training, create_real_features, get_feature_columns
from backend.model import (build_cnn, build_lstm, build_gru, build_cnn_lstm, build_cnn_gru,
                           MODEL_BUILDERS)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
import json

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REAL_METRICS_PATH = os.path.join(MODELS_DIR, "real_metrics.json")
REAL_BEST_MODEL_PATH = os.path.join(MODELS_DIR, "real_best_model.keras")


def train_real_models(X, y, epochs=50, batch_size=32):
    os.makedirs(MODELS_DIR, exist_ok=True)

    smote = SMOTE(random_state=42)
    X_flat = X.reshape(X.shape[0], -1)
    X_res, y_res = smote.fit_resample(X_flat, y)
    X_res = X_res.reshape(X_res.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    input_shape = (X_train.shape[1], 1)
    all_metrics = {}
    best_auc = 0
    best_model_name = ""

    builders = {
        "CNN": build_cnn, "LSTM": build_lstm, "GRU": build_gru,
        "CNN-LSTM": build_cnn_lstm, "CNN-GRU": build_cnn_gru
    }

    for name, builder in builders.items():
        print(f"\n{'='*50}")
        print(f"  Entrenando: {name} (datos reales)")
        print(f"{'='*50}")

        model = builder(input_shape)
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        y_pred_prob = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_prob)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "auc_roc": round(float(auc), 4),
            "confusion_matrix": cm.tolist(),
            "dataset": "real_mendeley"
        }
        all_metrics[name] = metrics
        print(f"  Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    all_metrics["best_model"] = best_model_name
    all_metrics["dataset_info"] = {
        "source": "Mendeley - Medication Adherence Diabetes Hypertension",
        "total_records": int(len(X)),
        "features": int(X.shape[1]),
        "class_distribution": {
            "adherent": int(np.sum(y == 1)),
            "non_adherent": int(np.sum(y == 0))
        }
    }

    with open(REAL_METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    best_builder = builders[best_model_name]
    best_model = best_builder(input_shape)
    best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    best_model.save(REAL_BEST_MODEL_PATH)

    print(f"\n{'='*50}")
    print(f"  MEJOR MODELO: {best_model_name} (AUC-ROC: {best_auc:.4f})")
    print(f"  Guardado en: {REAL_BEST_MODEL_PATH}")
    print(f"{'='*50}")

    return all_metrics


def main():
    data_path = os.path.join(os.path.dirname(__file__), "data",
                              "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx")

    if not os.path.exists(data_path):
        print(f"ERROR: Dataset no encontrado en {data_path}")
        return

    print("Cargando dataset real de Mendeley...")
    df = pd.read_excel(data_path)
    print(f"Registros: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    print(f"Distribucion: {df['ADHERENCE'].value_counts().to_dict()}")

    print("\nPreprocesando...")
    X, y = preprocess_real_training(df)
    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")

    print("\nEntrenando modelos neuronales...")
    metrics = train_real_models(X, y, epochs=50, batch_size=64)

    print("\nEntrenamiento completado.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "best_model"}, indent=2))


if __name__ == "__main__":
    main()
