import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv1D, GRU, Dense, Dropout, Flatten,
                                      BatchNormalization, MaxPooling1D, LSTM, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import joblib

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.keras")


def build_cnn_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


MODEL_BUILDERS = {
    "CNN": build_cnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
    "CNN-LSTM": build_cnn_lstm,
    "CNN-GRU": build_cnn_gru,
}


def train_models(X, y, epochs=50, batch_size=32):
    os.makedirs(MODELS_DIR, exist_ok=True)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X.reshape(X.shape[0], -1), y)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    input_shape = (X_train.shape[1], 1)
    all_metrics = {}
    best_auc = 0
    best_model_name = ""

    for name, builder in MODEL_BUILDERS.items():
        print(f"\nEntrenando modelo: {name}")
        model = builder(input_shape)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        history = model.fit(
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
            "confusion_matrix": cm.tolist()
        }
        all_metrics[name] = metrics
        print(f"  Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    all_metrics["best_model"] = best_model_name
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    best_builder = MODEL_BUILDERS[best_model_name]
    best_model = best_builder(input_shape)
    best_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    best_model.save(BEST_MODEL_PATH)

    print(f"\nMejor modelo: {best_model_name} (AUC-ROC: {best_auc:.4f})")
    return all_metrics


def get_best_model():
    if os.path.exists(BEST_MODEL_PATH):
        return load_model(BEST_MODEL_PATH)
    return None


def predict_adherence(model, X):
    if model is None:
        raise ValueError("No hay modelo entrenado disponible")
    prob = model.predict(X, verbose=0).flatten()[0]
    adherence_class = "Adherente" if prob >= 0.5 else "No Adherente"
    risk_level = classify_risk(prob)
    return float(prob), adherence_class, risk_level


def classify_risk(probability):
    if probability >= 0.8:
        return "Bajo Riesgo"
    elif probability >= 0.5:
        return "Riesgo Moderado"
    else:
        return "Alto Riesgo"


def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None
