import numpy as np
import os
import json
import shap
import joblib
from tensorflow.keras.models import load_model

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

FEATURE_NAMES = [
    "Genero", "Tipo Medicamento", "Nivel Educativo", "Nivel Ingreso",
    "Apoyo Social", "Gravedad Enfermedad", "Cobertura Seguro",
    "Edad", "Dosis", "N. Comorbilidades", "Adherencia Previa",
    "Indice Polifarmacia", "Ratio Edad-Dosis", "Score Riesgo Social",
    "Interaccion Edad-Comorb", "Severidad Numerica",
    "Score Adherencia Estimado", "Proxy Fragilidad",
]


def compute_shap_values(model, X_train_background, X_instance):
    try:
        X_bg = X_train_background.reshape(X_train_background.shape[0], -1)
        X_inst = X_instance.reshape(1, -1)

        n_background = min(20, X_bg.shape[0])
        background = shap.sample(X_bg, n_background) if X_bg.shape[0] > 0 else X_bg

        def predict_fn(x):
            return model.predict(x, verbose=0)

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_inst, nsamples=35)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = float(expected_val[0])
        else:
            expected_val = float(expected_val)

        return {
            "values": shap_vals.flatten().tolist(),
            "feature_names": FEATURE_NAMES,
            "expected_value": expected_val
        }
    except Exception as e:
        return {"error": str(e), "values": [], "feature_names": FEATURE_NAMES}


def get_feature_importance_from_shap(shap_result):
    if "error" in shap_result or not shap_result.get("values"):
        return []

    values = shap_result["values"]
    names = shap_result["feature_names"]

    importance_list = []
    for name, val in zip(names, values):
        importance_list.append({
            "feature": name,
            "importance": abs(float(val)),
            "direction": "positivo" if val > 0 else "negativo",
            "value": float(val)
        })

    importance_list.sort(key=lambda x: x["importance"], reverse=True)
    return importance_list


def generate_explanation(shap_result, adherence_class, risk_level):
    importance = get_feature_importance_from_shap(shap_result)
    top_factors = importance[:5]

    explanation = {
        "resumen": f"El paciente fue clasificado como {adherence_class} con {risk_level}.",
        "factores_principales": [],
        "recomendacion_clinica": ""
    }

    for factor in top_factors:
        explanation["factores_principales"].append(
            f"{factor['feature']}: impacto {factor['direction']} ({factor['value']:+.4f})"
        )

    if risk_level == "Alto Riesgo":
        explanation["recomendacion_clinica"] = (
            "Se recomienda intervencion inmediata: contactar al paciente, "
            "simplificar el esquema farmacologico y considerar apoyo socioeconomico."
        )
    elif risk_level == "Riesgo Moderado":
        explanation["recomendacion_clinica"] = (
            "Se sugiere seguimiento cercano con recordatorios personalizados "
            "y evaluacion de barreras de adherencia."
        )
    else:
        explanation["recomendacion_clinica"] = (
            "El paciente muestra buena adherencia. Mantener seguimiento rutinario "
            "y reforzar conductas positivas."
        )

    return explanation
