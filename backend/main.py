import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime
import numpy as np
import json
import logging
import pandas as pd

from backend.model import get_best_model, predict_adherence, get_metrics
from backend.preprocessing import (preprocess_single_patient, preprocess_training,
                                    load_encoders, load_scaler)
from backend.explainability import compute_shap_values, generate_explanation, get_feature_importance_from_shap
from backend.notifications import generate_notification_strategy
from backend.database import (
    insert_patient, insert_prediction, insert_notification,
    get_all_patients, get_patient_by_id, get_predictions_by_patient,
    get_notifications_by_patient, get_dashboard_stats, get_all_predictions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sistema Inteligente de Notificaciones Electronicas",
    description="API para prediccion de adherencia terapeutica en pacientes adultos mayores",
    version="2.0.0",
    docs_url="/api/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_model_version = "cnn-gru-v2.0"


def get_model():
    global _model
    if _model is None:
        _model = get_best_model()
    return _model


class PatientData(BaseModel):
    name: str
    age: int = Field(..., ge=60, le=100)
    gender: str
    medication_type: str
    dosage: float = Field(..., gt=0)
    education_level: str
    income_level: str
    social_support: str
    disease_severity: str
    num_comorbidities: int = Field(..., ge=0)
    insurance_coverage: str
    previous_adherence: int = Field(..., ge=0, le=1)


class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]


@app.get("/")
def root():
    return {
        "message": "Sistema Inteligente de Notificaciones Electronicas v2.0",
        "version": _model_version,
        "endpoints": {
            "POST /api/v1/predict": "Prediccion en tiempo real",
            "POST /api/v1/predict/batch": "Prediccion batch",
            "GET /api/v1/patients": "Listar pacientes",
            "GET /api/v1/patients/{id}": "Detalle de paciente",
            "GET /api/v1/predictions": "Historial de predicciones",
            "GET /api/v1/dashboard": "Estadisticas del dashboard",
            "GET /api/v1/metrics": "Metricas de modelos neuronales",
            "GET /api/v1/ensemble-comparison": "Comparacion con ensambles",
            "POST /api/v1/train": "Reentrenar modelos (batch)",
            "POST /api/v1/compare": "Comparar modelos (batch)",
            "POST /api/v1/optimize": "Optimizar hiperparametros (batch)",
            "POST /api/v1/fairness-audit": "Auditoria de equidad",
            "GET /api/v1/model/info": "Info del modelo cargado",
            "GET /api/health": "Health check",
        }
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/v1/model/info")
def model_info():
    return {
        "model_version": _model_version,
        "loaded": _model is not None,
        "metrics": get_metrics()
    }


@app.post("/api/v1/predict")
def predict_realtime(patient: PatientData):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado. Ejecute /api/v1/train primero.")

    encoders = load_encoders()
    scaler = load_scaler()
    if encoders is None or scaler is None:
        raise HTTPException(status_code=503, detail="Preprocesadores no disponibles.")

    patient_dict = patient.model_dump()

    try:
        X = preprocess_single_patient(patient_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error de preprocesamiento: {str(e)}")

    prob, adherence_class, risk_level = predict_adherence(model, X)

    patient_id = insert_patient(patient_dict)

    notification_strategy = generate_notification_strategy(risk_level, prob, patient_dict)

    shap_result = compute_shap_values(model, X.reshape(1, -1), X)
    explanation = generate_explanation(shap_result, adherence_class, risk_level)

    prediction_id = insert_prediction(
        patient_id, prob, adherence_class, risk_level,
        notification_strategy["tipo_notificacion"],
        json.dumps(shap_result)
    )

    insert_notification(patient_id, prediction_id,
                       notification_strategy["tipo_notificacion"],
                       notification_strategy["mensaje"])

    feature_importance = get_feature_importance_from_shap(shap_result)

    return {
        "patient_id": patient_id,
        "prediction_id": prediction_id,
        "adherence_probability": round(prob, 4),
        "adherence_class": adherence_class,
        "risk_level": risk_level,
        "notification": {
            "tipo": notification_strategy["tipo_notificacion"],
            "frecuencia": notification_strategy["frecuencia"],
            "canales": notification_strategy["canales"],
            "mensaje": notification_strategy["mensaje"],
            "acciones": notification_strategy["acciones_recomendadas"],
            "max_intentos": notification_strategy.get("max_intentos", 1),
            "escalacion": notification_strategy.get("escalacion", ""),
            "ajustes_dinamicos": notification_strategy.get("ajustes_dinamicos", []),
        },
        "explanation": explanation,
        "feature_importance": feature_importance,
        "model_version": _model_version,
    }


@app.post("/api/v1/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    results = []
    for i, patient in enumerate(request.patients):
        try:
            result = predict_realtime(patient)
            result["patient_index"] = i
            results.append(result)
        except Exception as e:
            results.append({"error": str(e), "patient_index": i})
    return {"predictions": results, "total": len(results), "successes": len([r for r in results if "error" not in r])}


@app.get("/api/v1/patients")
def list_patients():
    return {"patients": get_all_patients()}


@app.get("/api/v1/patients/{patient_id}")
def get_patient(patient_id: int):
    patient = get_patient_by_id(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    predictions = get_predictions_by_patient(patient_id)
    notifications = get_notifications_by_patient(patient_id)
    return {"patient": patient, "predictions": predictions, "notifications": notifications}


@app.get("/api/v1/predictions")
def list_predictions():
    return {"predictions": get_all_predictions()}


@app.get("/api/v1/dashboard")
def dashboard():
    return get_dashboard_stats()


@app.get("/api/v1/metrics")
def metrics():
    m = get_metrics()
    if m is None:
        raise HTTPException(status_code=404, detail="Metricas no disponibles. Ejecute /api/v1/train primero.")
    return m


@app.get("/api/v1/ensemble-comparison")
def ensemble_comparison():
    from backend.ml_comparator import get_ensemble_comparison
    result = get_ensemble_comparison()
    if result is None:
        raise HTTPException(status_code=404, detail="Comparacion no disponible. Ejecute /api/v1/compare primero.")
    return result


@app.post("/api/v1/train")
def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_training_pipeline)
    return {"message": "Reentrenamiento iniciado en background", "status": "queued", "timestamp": datetime.now().isoformat()}


def _run_training_pipeline():
    logger.info("Iniciando pipeline de reentrenamiento...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medication_adherence.csv")
    if not os.path.exists(data_path):
        logger.error("Dataset no encontrado.")
        return

    df = pd.read_csv(data_path)
    X, y = preprocess_training(df)
    X_nn = X.reshape(X.shape[0], X.shape[1], 1)

    from backend.model import train_models
    train_models(X_nn, y)

    global _model
    _model = None
    logger.info("Reentrenamiento completado.")


@app.post("/api/v1/compare")
def trigger_comparison(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_comparison_pipeline)
    return {"message": "Comparacion de ensambles iniciada en background", "status": "queued"}


def _run_comparison_pipeline():
    logger.info("Iniciando comparacion de modelos de ensamble...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medication_adherence.csv")
    df = pd.read_csv(data_path)

    from backend.ml_comparator import run_ensemble_comparison
    run_ensemble_comparison(df)
    logger.info("Comparacion de ensambles completada.")


@app.post("/api/v1/optimize")
def trigger_optimization(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_optimization_pipeline)
    return {"message": "Optimizacion bayesiana iniciada en background", "status": "queued"}


def _run_optimization_pipeline():
    logger.info("Iniciando optimizacion con Optuna...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medication_adherence.csv")
    df = pd.read_csv(data_path)

    from backend.preprocessing import preprocess_training
    X, y = preprocess_training(df)

    from backend.hyperparameter_optimizer import run_full_optimization
    run_full_optimization(X, y)
    logger.info("Optimizacion completada.")


@app.post("/api/v1/fairness-audit")
def trigger_fairness_audit():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "medication_adherence.csv")
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")

    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    df = pd.read_csv(data_path)
    X, y = preprocess_training(df)
    X_nn = X.reshape(X.shape[0], X.shape[1], 1)

    y_pred_prob = model.predict(X_nn, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    from backend.fairness_audit import run_full_fairness_audit
    report = run_full_fairness_audit(df, y, y_pred, y_pred_prob)

    return report
