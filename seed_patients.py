import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
import numpy as np
from backend.model import get_best_model, predict_adherence
from backend.preprocessing import preprocess_single_patient
from backend.explainability import compute_shap_values, get_feature_importance_from_shap
from backend.notifications import generate_notification_strategy
from backend.database import (
    init_db, insert_patient, insert_prediction, insert_notification
)

random.seed(42)
np.random.seed(42)

NAMES_M = [
    "Carlos Mendoza", "Luis Quispe", "Miguel Torres", "Pedro Sanchez",
    "Jorge Ramirez", "Roberto Flores", "Miguel Angel Herrera", "Juan Perez",
    "Francisco Diaz", "Antonio Morales", "Ricardo Vargas", "Enrique Castro",
    "Alberto Ramos", "Fernando Lopez", "Raul Guerrero", "Hector Salazar",
    "Arturo Castillo", "Gonzalo Mendez", "Victor Rojas", "Sergio Medina",
    "Alfredo Silva", "Ernesto Cruz", "Oscar Navarro", "Rogelio Paredes",
    "Walter Campos", "Edgar Luna", "Byron Soto", "Cesar Pinto",
    "Rolando Chavez", "Eduardo Reyes", "Pablo Aguilar", "Manuel Gutierrez",
    "Alejandro Espinoza", "Daniel Fernandez", "Andres Caceres",
]

NAMES_F = [
    "Maria Garcia", "Rosa Lopez", "Carmen Perez", "Ana Martinez",
    "Teresa Rodriguez", "Claudia Fernandez", "Patricia Gonzalez",
    "Lucia Torres", "Sandra Ramirez", "Gloria Ramos", "Monica Flores",
    "Beatriz Castro", "Laura Morales", "Silvia Vargas", "Nancy Herrera",
    "Diana Ortiz", "Elena Salazar", "Martha Castillo", "Janet Mendoza",
    "Karina Rojas", "Irene Medina", "Olga Silva", "Gladys Cruz",
    "Milagros Navarro", "Yolanda Paredes", "Lourdes Campos", "Hilda Luna",
    "Ruth Soto", "Pilar Pinto", "Cecilia Chavez", "Nidia Reyes",
    "Flor Aguilar", "Delia Gutierrez", "Luz Espinoza", "Zoila Fernandez",
]

MEDICATIONS = [
    "Antihipertensivo", "Antidiabetico", "Anticoagulante",
    "Hipolipemiante", "Antidepresivo", "Analgesico"
]

patient_profiles = []

# Perfil 1: Adultos jovenes (60-69) con buen acceso
for i in range(25):
    patient_profiles.append({
        "name": random.choice(NAMES_M + NAMES_F),
        "age": random.randint(60, 69),
        "gender": random.choice(["Masculino", "Femenino"]),
        "medication_type": random.choice(MEDICATIONS),
        "dosage": round(random.uniform(2, 15), 1),
        "education_level": random.choice(["Secundaria", "Superior"]),
        "income_level": random.choice(["Medio", "Medio-Alto"]),
        "social_support": random.choice(["Moderado", "Alto"]),
        "disease_severity": random.choice(["Leve", "Moderada"]),
        "num_comorbidities": random.randint(1, 3),
        "insurance_coverage": random.choice(["Essalud", "Privado"]),
        "previous_adherence": random.choice([1, 1, 1, 0]),
    })

# Perfil 2: Adultos medios (70-79) con riesgo moderado
for i in range(30):
    patient_profiles.append({
        "name": random.choice(NAMES_M + NAMES_F),
        "age": random.randint(70, 79),
        "gender": random.choice(["Masculino", "Femenino"]),
        "medication_type": random.choice(MEDICATIONS),
        "dosage": round(random.uniform(3, 20), 1),
        "education_level": random.choice(["Primaria", "Secundaria"]),
        "income_level": random.choice(["Bajo", "Medio-Bajo", "Medio"]),
        "social_support": random.choice(["Bajo", "Moderado"]),
        "disease_severity": random.choice(["Moderada", "Severa"]),
        "num_comorbidities": random.randint(2, 5),
        "insurance_coverage": random.choice(["Essalud", "SIS", "SIS"]),
        "previous_adherence": random.choice([1, 1, 0, 0]),
    })

# Perfil 3: Adultos mayores (80+) con alto riesgo
for i in range(25):
    patient_profiles.append({
        "name": random.choice(NAMES_M + NAMES_F),
        "age": random.randint(80, 95),
        "gender": random.choice(["Masculino", "Femenino"]),
        "medication_type": random.choice(MEDICATIONS),
        "dosage": round(random.uniform(1, 12), 1),
        "education_level": random.choice(["Sin educacion formal", "Primaria"]),
        "income_level": random.choice(["Bajo", "Medio-Bajo"]),
        "social_support": random.choice(["Nulo", "Bajo"]),
        "disease_severity": random.choice(["Severa", "Muy severa"]),
        "num_comorbidities": random.randint(3, 7),
        "insurance_coverage": random.choice(["Ninguno", "SIS", "Essalud"]),
        "previous_adherence": random.choice([1, 0, 0, 0]),
    })

# Perfil 4: Casos criticos (polifarmacia, sin seguro, apoyo nulo)
for i in range(15):
    patient_profiles.append({
        "name": random.choice(NAMES_M + NAMES_F),
        "age": random.randint(75, 92),
        "gender": random.choice(["Masculino", "Femenino"]),
        "medication_type": random.choice(["Anticoagulante", "Antidepresivo"]),
        "dosage": round(random.uniform(5, 25), 1),
        "education_level": "Sin educacion formal",
        "income_level": "Bajo",
        "social_support": "Nulo",
        "disease_severity": random.choice(["Severa", "Muy severa"]),
        "num_comorbidities": random.randint(5, 8),
        "insurance_coverage": "Ninguno",
        "previous_adherence": 0,
    })

# Perfil 5: Casos estables con buena adherencia
for i in range(15):
    patient_profiles.append({
        "name": random.choice(NAMES_M + NAMES_F),
        "age": random.randint(60, 75),
        "gender": random.choice(["Masculino", "Femenino"]),
        "medication_type": random.choice(["Antihipertensivo", "Hipolipemiante"]),
        "dosage": round(random.uniform(2, 10), 1),
        "education_level": random.choice(["Secundaria", "Superior"]),
        "income_level": random.choice(["Medio", "Medio-Alto"]),
        "social_support": "Alto",
        "disease_severity": "Leve",
        "num_comorbidities": random.randint(1, 2),
        "insurance_coverage": random.choice(["Privado", "Essalud"]),
        "previous_adherence": 1,
    })

# Perfil 6: Mujeres adultas mayores con depresion
for i in range(10):
    patient_profiles.append({
        "name": random.choice(NAMES_F),
        "age": random.randint(68, 88),
        "gender": "Femenino",
        "medication_type": "Antidepresivo",
        "dosage": round(random.uniform(10, 50), 1),
        "education_level": random.choice(["Primaria", "Secundaria"]),
        "income_level": random.choice(["Bajo", "Medio-Bajo"]),
        "social_support": random.choice(["Nulo", "Bajo", "Moderado"]),
        "disease_severity": random.choice(["Moderada", "Severa"]),
        "num_comorbidities": random.randint(2, 5),
        "insurance_coverage": random.choice(["SIS", "Essalud"]),
        "previous_adherence": random.choice([1, 0, 0]),
    })

random.shuffle(patient_profiles)

def main():
    print("=" * 60)
    print("  POBLANDO BASE DE DATOS CON PACIENTES DIVERSOS")
    print("=" * 60)

    init_db()

    model = get_best_model()
    if model is None:
        print("ERROR: Modelo no encontrado. Ejecuta train.py primero.")
        return

    total = len(patient_profiles)
    alto_riesgo = 0
    moderado = 0
    bajo = 0

    for i, profile in enumerate(patient_profiles):
        try:
            X = preprocess_single_patient(profile)
            prob, adherence_class, risk_level = predict_adherence(model, X)

            patient_id = insert_patient(profile)

            notification = generate_notification_strategy(risk_level, prob, profile)

            shap_result = compute_shap_values(model, X.reshape(1, -1), X)

            prediction_id = insert_prediction(
                patient_id, prob, adherence_class, risk_level,
                notification["tipo_notificacion"],
                str(shap_result)
            )

            insert_notification(
                patient_id, prediction_id,
                notification["tipo_notificacion"],
                notification["mensaje"]
            )

            if "Alto" in risk_level:
                alto_riesgo += 1
            elif "Moderado" in risk_level:
                moderado += 1
            else:
                bajo += 1

            icon = "!!! " if "Alto" in risk_level else ("!  " if "Moderado" in risk_level else "   ")
            print(f"  [{i+1}/{total}] {icon}{profile['name']:25s} | "
                  f"{profile['age']}y {profile['gender'][:1]} | "
                  f"Prob={prob:.1%} | {risk_level}")

        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR: {profile['name']} - {e}")

    print("\n" + "=" * 60)
    print("  POBLADO COMPLETADO")
    print("=" * 60)
    print(f"\n  Total pacientes: {total}")
    print(f"  Alto Riesgo:     {alto_riesgo} ({alto_riesgo/total:.0%})")
    print(f"  Riesgo Moderado: {moderado} ({moderado/total:.0%})")
    print(f"  Bajo Riesgo:     {bajo} ({bajo/total:.0%})")
    print(f"\n  Base de datos: data/patients.db")
    print(f"  Abre Streamlit para ver el dashboard: streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
