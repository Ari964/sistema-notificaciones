import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import json
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OPTUNA_PATH = os.path.join(MODELS_DIR, "optuna_results.json")


def optimize_xgboost(X, y, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", xgb.XGBClassifier(
                **params, eval_metric="logloss", random_state=42
            ))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="xgboost_optimization"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def optimize_lightgbm(X, y, n_trials=50):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
        pipe = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", lgb.LGBMClassifier(
                **params, random_state=42, verbose=-1
            ))
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        study_name="lightgbm_optimization"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def optimize_cnn_gru(X, y, n_trials=30):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Conv1D, GRU, Dense, Dropout,
                                          BatchNormalization, MaxPooling1D, Input)
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam

    def objective(trial):
        n_filters_1 = trial.suggest_categorical("n_filters_1", [32, 64, 128])
        n_filters_2 = trial.suggest_categorical("n_filters_2", [64, 128, 256])
        gru_units_1 = trial.suggest_categorical("gru_units_1", [64, 128, 256])
        gru_units_2 = trial.suggest_categorical("gru_units_2", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dense_units = trial.suggest_categorical("dense_units", [32, 64, 128])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []

        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(
                X_train.reshape(X_train.shape[0], -1), y_train
            )
            X_res = X_res.reshape(X_res.shape[0], X.shape[1], 1)

            model = Sequential([
                Input(shape=(X.shape[1], 1)),
                Conv1D(n_filters_1, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling1D(pool_size=2),
                Conv1D(n_filters_2, kernel_size=3, activation='relu', padding='same'),
                BatchNormalization(),
                GRU(gru_units_1, return_sequences=True),
                Dropout(dropout_rate),
                GRU(gru_units_2, return_sequences=False),
                Dropout(dropout_rate),
                Dense(dense_units, activation='relu'),
                Dropout(min(dropout_rate + 0.1, 0.6)),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='binary_crossentropy', metrics=['accuracy']
            )

            model.fit(
                X_res, y_res,
                validation_data=(
                    X_val.reshape(X_val.shape[0], X.shape[1], 1), y_val
                ),
                epochs=30, batch_size=batch_size, verbose=0,
                callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
            )

            y_pred = model.predict(
                X_val.reshape(X_val.shape[0], X.shape[1], 1), verbose=0
            ).flatten()

            from sklearn.metrics import roc_auc_score
            auc_scores.append(roc_auc_score(y_val, y_pred))

            trial.report(np.mean(auc_scores), len(auc_scores))
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(auc_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
        study_name="cnn_gru_optimization"
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study.best_params, study.best_value


def run_full_optimization(X_flat, y):
    print("\n" + "=" * 60)
    print("  OPTIMIZACION BAYESIANA CON OPTUNA")
    print("=" * 60)

    results = {}

    print("\n[1/3] Optimizando XGBoost...")
    xgb_params, xgb_auc = optimize_xgboost(X_flat, y, n_trials=30)
    results["XGBoost"] = {"best_params": xgb_params, "best_auc_roc": round(xgb_auc, 4)}
    print(f"  XGBoost optimizado: AUC-ROC = {xgb_auc:.4f}")

    print("\n[2/3] Optimizando LightGBM...")
    lgb_params, lgb_auc = optimize_lightgbm(X_flat, y, n_trials=30)
    results["LightGBM"] = {"best_params": lgb_params, "best_auc_roc": round(lgb_auc, 4)}
    print(f"  LightGBM optimizado: AUC-ROC = {lgb_auc:.4f}")

    print("\n[3/3] Optimizando CNN-GRU...")
    X_nn = X_flat.reshape(X_flat.shape[0], X_flat.shape[1], 1)
    cnn_params, cnn_auc = optimize_cnn_gru(X_nn, y, n_trials=15)
    results["CNN-GRU"] = {"best_params": cnn_params, "best_auc_roc": round(cnn_auc, 4)}
    print(f"  CNN-GRU optimizado: AUC-ROC = {cnn_auc:.4f}")

    best_model = max(results, key=lambda k: results[k]["best_auc_roc"])
    results["best_model"] = best_model

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(OPTUNA_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Mejor modelo optimizado: {best_model} "
          f"(AUC-ROC: {results[best_model]['best_auc_roc']:.4f})")
    return results
