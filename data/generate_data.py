import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

age = np.random.randint(60, 95, N)
gender = np.random.choice(["Masculino", "Femenino"], N, p=[0.45, 0.55])
medication_type = np.random.choice(
    ["Antihipertensivo", "Antidiabético", "Anticoagulante", "Hipolipemiante", "Antidepresivo", "Analgésico"],
    N, p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08]
)
dosage = np.round(np.random.uniform(0.5, 20.0, N), 1)
education_level = np.random.choice(
    ["Sin educación formal", "Primaria", "Secundaria", "Superior"],
    N, p=[0.10, 0.35, 0.35, 0.20]
)
income_level = np.random.choice(
    ["Bajo", "Medio-Bajo", "Medio", "Medio-Alto"],
    N, p=[0.30, 0.35, 0.25, 0.10]
)
social_support = np.random.choice(
    ["Nulo", "Bajo", "Moderado", "Alto"],
    N, p=[0.10, 0.25, 0.40, 0.25]
)
disease_severity = np.random.choice(
    ["Leve", "Moderada", "Severa", "Muy severa"],
    N, p=[0.20, 0.40, 0.30, 0.10]
)
num_comorbidities = np.random.randint(1, 8, N)
insurance_coverage = np.random.choice(
    ["Ninguno", "Essalud", "SIS", "Privado"],
    N, p=[0.15, 0.35, 0.35, 0.15]
)
previous_adherence = np.random.choice([0, 1], N, p=[0.35, 0.65])

adherence_prob = np.zeros(N)

for i in range(N):
    p = 0.5
    if age[i] < 70:
        p += 0.05
    elif age[i] > 80:
        p -= 0.10

    if gender[i] == "Femenino":
        p += 0.03

    if medication_type[i] in ["Anticoagulante", "Antidepresivo"]:
        p -= 0.08

    if dosage[i] > 10:
        p -= 0.05

    edu_map = {"Sin educación formal": -0.10, "Primaria": -0.05, "Secundaria": 0.02, "Superior": 0.08}
    p += edu_map.get(education_level[i], 0)

    inc_map = {"Bajo": -0.12, "Medio-Bajo": -0.05, "Medio": 0.03, "Medio-Alto": 0.08}
    p += inc_map.get(income_level[i], 0)

    sup_map = {"Nulo": -0.15, "Bajo": -0.08, "Moderado": 0.03, "Alto": 0.10}
    p += sup_map.get(social_support[i], 0)

    sev_map = {"Leve": 0.08, "Moderada": 0.0, "Severa": -0.08, "Muy severa": -0.15}
    p += sev_map.get(disease_severity[i], 0)

    if num_comorbidities[i] >= 4:
        p -= 0.10
    elif num_comorbidities[i] <= 2:
        p += 0.05

    ins_map = {"Ninguno": -0.10, "Essalud": 0.03, "SIS": 0.02, "Privado": 0.06}
    p += ins_map.get(insurance_coverage[i], 0)

    if previous_adherence[i] == 1:
        p += 0.15
    else:
        p -= 0.10

    p = np.clip(p + np.random.normal(0, 0.08), 0.05, 0.95)
    adherence_prob[i] = p

adherence = (adherence_prob >= 0.5).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Medication_Type": medication_type,
    "Dosage": dosage,
    "Education_Level": education_level,
    "Income_Level": income_level,
    "Social_Support": social_support,
    "Disease_Severity": disease_severity,
    "Num_Comorbidities": num_comorbidities,
    "Insurance_Coverage": insurance_coverage,
    "Previous_Adherence": previous_adherence,
    "Adherence": adherence
})

output_path = os.path.join(os.path.dirname(__file__), "medication_adherence.csv")
df.to_csv(output_path, index=False)
print(f"Dataset generado: {output_path}")
print(f"Registros: {len(df)}")
print(f"Distribución Adherencia: {df['Adherence'].value_counts().to_dict()}")
print(f"\nColumnas: {list(df.columns)}")
print(f"\nPrimeras filas:")
print(df.head())
