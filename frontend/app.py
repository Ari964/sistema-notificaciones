import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from frontend.xai_views import (render_shap_waterfall, render_shap_comparison_bar,
                                 render_patient_risk_gauge, render_model_comparison_chart,
                                 render_confusion_matrices)

BACKEND_URL = "http://localhost:8000"
API = f"{BACKEND_URL}/api/v1"

st.set_page_config(
    page_title="Sistema de Notificaciones Electronicas Inteligentes v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        color: white; padding: 20px 30px; border-radius: 10px; margin-bottom: 20px;
    }
    .risk-alto { color: #d32f2f; font-weight: bold; font-size: 1.2em; }
    .risk-moderado { color: #f57c00; font-weight: bold; font-size: 1.2em; }
    .risk-bajo { color: #388e3c; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)


def api_get(endpoint):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=30)
        return r.json() if r.status_code == 200 else None
    except:
        return None


def api_post(endpoint, data):
    try:
        r = requests.post(f"{API}{endpoint}", json=data, timeout=120)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


with st.sidebar:
    st.image("https://img.icons8.com/color/96/brain.png", width=80)
    st.title("Menu Principal")
    page = st.radio(
        "Navegacion",
        ["🏠 Dashboard", "📝 Registro + Prediccion", "📊 Historial Predicciones",
         "👥 Pacientes", "🧠 Comparar Modelos", "⚙️ Optimizar Hiperparametros",
         "⚖️ Auditoria de Equidad", "🔔 Notificaciones"]
    )
    st.divider()
    health = api_get("/predict") if False else None
    st.caption("Sistema Inteligente de Notificaciones v2.0")
    st.caption("FastAPI + Streamlit + Deep Learning + XAI")


if page == "🏠 Dashboard":
    st.markdown('<div class="main-header"><h1>Sistema Inteligente de Notificaciones Electronicas</h1><p>Redes Neuronales + Ensembles + XAI para Adherencia en Adultos Mayores</p></div>', unsafe_allow_html=True)

    stats = api_get("/dashboard")
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Pacientes", stats.get("total_patients", 0))
        with c2: st.metric("Total Predicciones", stats.get("total_predictions", 0))
        with c3: st.metric("Notif. Pendientes", stats.get("pending_notifications", 0))
        with c4:
            alto = stats.get("risk_distribution", {}).get("Alto Riesgo", 0)
            st.metric("Pacientes Alto Riesgo", alto)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Distribucion por Nivel de Riesgo")
            risk_dist = stats.get("risk_distribution", {})
            if risk_dist:
                colors = {"Alto Riesgo": "#d32f2f", "Riesgo Moderado": "#f57c00", "Bajo Riesgo": "#388e3c"}
                fig = px.bar(x=list(risk_dist.keys()), y=list(risk_dist.values()),
                            color=list(risk_dist.keys()), color_discrete_map=colors)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.subheader("Distribucion de Adherencia")
            adh_dist = stats.get("adherence_distribution", {})
            if adh_dist:
                fig = px.pie(values=list(adh_dist.values()), names=list(adh_dist.keys()),
                            color_discrete_sequence=["#388e3c", "#d32f2f"])
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay datos disponibles. Registre pacientes primero.")

    model_info = api_get("/model/info")
    if model_info:
        st.divider()
        st.subheader("Estado del Modelo")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Version", model_info.get("model_version", "N/A"))
        with c2: st.metric("Cargado", "Si" if model_info.get("loaded") else "No")
        with c3:
            m = model_info.get("metrics", {})
            best = m.get("best_model", "N/A") if m else "N/A"
            st.metric("Mejor Modelo", best)


elif page == "📝 Registro + Prediccion":
    st.header("📝 Registro de Paciente y Prediccion en Tiempo Real")

    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Nombre completo*", value="Maria Garcia Lopez")
            age = st.number_input("Edad*", min_value=60, max_value=100, value=72)
            gender = st.selectbox("Genero*", ["Masculino", "Femenino"])
            medication_type = st.selectbox("Tipo de Medicamento*", [
                "Antihipertensivo", "Antidiabetico", "Anticoagulante",
                "Hipolipemiante", "Antidepresivo", "Analgesico"])
        with c2:
            dosage = st.number_input("Dosis (mg)*", min_value=0.5, max_value=50.0, value=10.0, step=0.5)
            education_level = st.selectbox("Nivel Educativo*", [
                "Sin educacion formal", "Primaria", "Secundaria", "Superior"])
            income_level = st.selectbox("Nivel de Ingreso*", [
                "Bajo", "Medio-Bajo", "Medio", "Medio-Alto"])
            social_support = st.selectbox("Apoyo Social*", ["Nulo", "Bajo", "Moderado", "Alto"])
        with c3:
            disease_severity = st.selectbox("Gravedad de Enfermedad*", [
                "Leve", "Moderada", "Severa", "Muy severa"])
            num_comorbidities = st.number_input("N. Comorbilidades*", min_value=0, max_value=10, value=2)
            insurance_coverage = st.selectbox("Cobertura de Seguro*", [
                "Ninguno", "Essalud", "SIS", "Privado"])
            previous_adherence = st.selectbox("Adherencia Previa*", [
                (1, "Si"), (0, "No")], format_func=lambda x: x[1])

        submitted = st.form_submit_button("Predecir Adherencia", use_container_width=True, type="primary")

    if submitted:
        patient_data = {
            "name": name, "age": age, "gender": gender,
            "medication_type": medication_type, "dosage": dosage,
            "education_level": education_level, "income_level": income_level,
            "social_support": social_support, "disease_severity": disease_severity,
            "num_comorbidities": num_comorbidities,
            "insurance_coverage": insurance_coverage,
            "previous_adherence": previous_adherence[0]
        }

        with st.spinner("Procesando prediccion..."):
            result = api_post("/predict", patient_data)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success("Prediccion completada exitosamente")

            c1, c2 = st.columns([1, 2])
            with c1:
                render_patient_risk_gauge(result["adherence_probability"], result["risk_level"])
            with c2:
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.metric("Prob. Adherencia", f"{result['adherence_probability']:.1%}")
                with mc2: st.metric("Clasificacion", result["adherence_class"])
                with mc3: st.metric("Version Modelo", result.get("model_version", "N/A"))

            st.divider()
            col_n, col_x = st.columns(2)
            with col_n:
                st.subheader("Estrategia de Notificacion")
                notif = result["notification"]
                st.info(f"**Tipo:** {notif['tipo']}")
                st.info(f"**Frecuencia:** {notif['frecuencia']}")
                st.info(f"**Canales:** {', '.join(notif['canales'])}")
                st.info(f"**Max Intentos:** {notif.get('max_intentos', 1)}")
                st.info(f"**Escalacion:** {notif.get('escalacion', 'N/A')}")
                if notif.get("ajustes_dinamicos"):
                    st.warning("**Ajustes Dinamicos:**")
                    for adj in notif["ajustes_dinamicos"]:
                        st.write(f"  - {adj}")
                st.write("**Mensaje Personalizado:**")
                st.write(notif["mensaje"])
                st.write("**Acciones Recomendadas:**")
                for accion in notif["acciones"]:
                    st.write(f"- {accion}")

            with col_x:
                st.subheader("Explicacion del Modelo (XAI)")
                explanation = result["explanation"]
                st.write(f"**Resumen:** {explanation['resumen']}")
                st.write("**Factores Principales:**")
                for factor in explanation["factores_principales"]:
                    st.write(f"  - {factor}")
                st.write(f"**Recomendacion Clinica:** {explanation['recomendacion_clinica']}")

            if result.get("feature_importance"):
                st.divider()
                render_shap_comparison_bar(result["feature_importance"])


elif page == "📊 Historial Predicciones":
    st.header("Historial de Predicciones")

    predictions = api_get("/predictions")
    if predictions and predictions.get("predictions"):
        preds = predictions["predictions"]
        df = pd.DataFrame(preds)

        display_cols = []
        for col in ["patient_name", "adherence_probability", "adherence_class", "risk_level", "created_at"]:
            if col in df.columns:
                display_cols.append(col)

        if display_cols:
            display_df = df[display_cols].copy()
            col_rename = {"patient_name": "Paciente", "adherence_probability": "Prob. Adherencia",
                         "adherence_class": "Clasificacion", "risk_level": "Nivel Riesgo", "created_at": "Fecha"}
            display_df.columns = [col_rename.get(c, c) for c in display_df.columns]
            if "Prob. Adherencia" in display_df.columns:
                display_df["Prob. Adherencia"] = display_df["Prob. Adherencia"].apply(
                    lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else x)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Distribucion de Riesgo")
                risk_counts = df["risk_level"].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                            color_discrete_map={"Alto Riesgo": "#d32f2f", "Riesgo Moderado": "#f57c00", "Bajo Riesgo": "#388e3c"})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.subheader("Distribucion de Probabilidades")
                fig = px.histogram(df, x="adherence_probability", nbins=20, color="risk_level",
                                 color_discrete_map={"Alto Riesgo": "#d32f2f", "Riesgo Moderado": "#f57c00", "Bajo Riesgo": "#388e3c"})
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay predicciones registradas aun.")


elif page == "👥 Pacientes":
    st.header("Gestion de Pacientes")

    patients = api_get("/patients")
    if patients and patients.get("patients"):
        plist = patients["patients"]
        df = pd.DataFrame(plist)

        search = st.text_input("Buscar paciente por nombre")
        if search:
            df = df[df["name"].str.contains(search, case=False, na=False)]

        st.write(f"Total: {len(df)} pacientes")
        display_df = df[["id", "name", "age", "gender", "medication_type", "created_at"]].rename(
            columns={"id": "ID", "name": "Nombre", "age": "Edad", "gender": "Genero",
                     "medication_type": "Medicamento", "created_at": "Registro"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        patient_ids = [p["id"] for p in plist]
        selected_id = st.selectbox("Seleccionar paciente para ver detalles", patient_ids,
                                   format_func=lambda x: next((p["name"] for p in plist if p["id"] == x), str(x)))

        if selected_id:
            detail = api_get(f"/patients/{selected_id}")
            if detail:
                st.divider()
                st.subheader(f"Detalle: {detail['patient']['name']}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Edad:** {detail['patient']['age']}")
                    st.write(f"**Genero:** {detail['patient']['gender']}")
                    st.write(f"**Medicamento:** {detail['patient']['medication_type']}")
                with c2:
                    st.write(f"**Dosis:** {detail['patient']['dosage']} mg")
                    st.write(f"**Educacion:** {detail['patient']['education_level']}")
                    st.write(f"**Ingresos:** {detail['patient']['income_level']}")
                with c3:
                    st.write(f"**Comorbilidades:** {detail['patient']['num_comorbidities']}")
                    st.write(f"**Seguro:** {detail['patient']['insurance_coverage']}")
                    st.write(f"**Adherencia Previa:** {'Si' if detail['patient']['previous_adherence'] else 'No'}")

                if detail.get("predictions"):
                    st.subheader("Predicciones")
                    for pred in detail["predictions"]:
                        risk = pred["risk_level"]
                        icon = "🔴" if "Alto" in risk else ("🟢" if "Bajo" in risk else "🟡")
                        st.write(f"{icon} **{pred['created_at']}** - {pred['adherence_class']} "
                               f"(Prob: {pred['adherence_probability']:.1%}) - {risk}")

                if detail.get("notifications"):
                    st.subheader("Notificaciones")
                    for notif in detail["notifications"]:
                        st.info(f"**{notif['notification_type']}** ({notif['created_at']})\n{notif['message']}")
    else:
        st.info("No hay pacientes registrados.")


elif page == "🧠 Comparar Modelos":
    st.header("Comparacion de Modelos de Ensamble vs Redes Neuronales")

    st.info("Compara modelos clasicos (XGBoost, Random Forest, LightGBM) contra las arquitecturas neuronales (CNN-GRU, LSTM, etc.)")

    if st.button("Ejecutar Comparacion de Ensambles", type="primary", use_container_width=True):
        with st.spinner("Comparando modelos... Esto puede tomar 2-5 minutos."):
            result = api_post("/compare", {})
        if result and "error" not in result:
            st.success(result.get("message", "Comparacion completada"))
        elif result:
            st.error(f"Error: {result.get('error', 'Desconocido')}")

    comparison = api_get("/ensemble-comparison")
    if comparison:
        st.divider()
        st.subheader("Resultados de la Comparacion de Ensambles")
        best = comparison.get("best_ensemble_model", "N/A")
        st.success(f"Mejor modelo de ensamble: **{best}**")

        model_names = [k for k in comparison.keys() if k != "best_ensemble_model"]
        if model_names:
            rows = []
            for name in model_names:
                m = comparison[name]
                if isinstance(m, dict) and "accuracy" in m:
                    rows.append({
                        "Modelo": name, "Accuracy": m["accuracy"], "Precision": m["precision"],
                        "Recall": m["recall"], "F1-Score": m["f1_score"], "AUC-ROC": m["auc_roc"]
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

                fig = go.Figure()
                for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
                    fig.add_trace(go.Bar(name=metric, x=df["Modelo"], y=df[metric],
                                        text=df[metric].round(3), textposition='auto'))
                fig.update_layout(barmode='group', height=500, yaxis_title="Valor")
                st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Metricas de Modelos Neuronales (CNN, LSTM, GRU, CNN-LSTM, CNN-GRU)")
    nn_metrics = api_get("/metrics")
    if nn_metrics:
        render_model_comparison_chart(nn_metrics)
        st.divider()
        st.subheader("Matrices de Confusion (Neuronales)")
        render_confusion_matrices(nn_metrics)
    else:
        st.warning("Metricas neuronales no disponibles. Ejecute el entrenamiento primero.")


elif page == "⚙️ Optimizar Hiperparametros":
    st.header("Optimizacion Bayesiana de Hiperparametros (Optuna)")

    st.write("Busqueda bayesiana para encontrar la mejor configuracion de hiperparametros en XGBoost, LightGBM y CNN-GRU.")

    n_trials = st.slider("Numero de trials por modelo", 10, 60, 20)

    if st.button("Iniciar Optimizacion", type="primary", use_container_width=True):
        with st.spinner(f"Optimizando {n_trials} configurations por modelo... Esto puede tardar 10-30 minutos."):
            result = api_post("/optimize", {"n_trials": n_trials})
        if result and "error" not in result:
            st.success(result.get("message", "Optimizacion completada"))
        elif result:
            st.error(f"Error: {result.get('error', 'Desconocido')}")

    optuna_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "optuna_results.json")
    if os.path.exists(optuna_path):
        import json
        with open(optuna_path, "r") as f:
            optuna_results = json.load(f)

        st.divider()
        st.subheader("Resultados de la Optimizacion")

        model_keys = [k for k in optuna_results.keys() if k != "best_model"]
        if model_keys:
            rows = []
            for name in model_keys:
                m = optuna_results[name]
                if isinstance(m, dict):
                    rows.append({
                        "Modelo": name,
                        "AUC-ROC Optimizado": m.get("best_auc_roc", 0),
                    })
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            best_opt = optuna_results.get("best_model", "N/A")
            st.success(f"Mejor modelo optimizado: **{best_opt}**")

            for name in model_keys:
                with st.expander(f"Mejores hiperparametros: {name}"):
                    st.json(optuna_results[name].get("best_params", {}))


elif page == "⚖️ Auditoria de Equidad":
    st.header("Auditoria de Equidad y Control de Sesgo")

    st.write("Analiza si el modelo presenta sesgo discriminatorio por genero, nivel socioeconomico o rango de edad.")

    if st.button("Ejecutar Auditoria Completa", type="primary", use_container_width=True):
        with st.spinner("Ejecutando auditoria de equidad..."):
            result = api_post("/fairness-audit", {})
        if result and "error" not in result:
            st.session_state["fairness_report"] = result
            st.success("Auditoria completada")
        elif result:
            st.error(f"Error: {result.get('error', 'Desconocido')}")

    report = st.session_state.get("fairness_report")

    if report:
        st.divider()
        st.subheader("1. Disparate Impact (4/5ths Rule)")
        di = report.get("disparate_impact", {})
        group_accs = di.get("group_accuracies", {})
        di_by_attr = di.get("by_attribute", {})

        for attr, accs in group_accs.items():
            with st.expander(f"{attr} - Precision por grupo"):
                for group, acc in accs.items():
                    st.write(f"  **{group}**: Accuracy = {acc:.4f}")

        for attr, comparisons in di_by_attr.items():
            st.write(f"**{attr}:**")
            for pair, metrics in comparisons.items():
                flag = "⚠️ DISCRIMINACION" if metrics["flagged"] else "✅ OK"
                st.write(f"  {pair}: Ratio = {metrics['disparate_impact_ratio']:.4f} [{flag}]")

        st.divider()
        st.subheader("2. Equalized Odds (TPR/FPR por grupo)")
        eo = report.get("equalized_odds", {})
        for attr, groups in eo.items():
            if isinstance(groups, dict) and attr != "max_tpr_disparity":
                with st.expander(f"{attr}"):
                    for group, metrics in groups.items():
                        if isinstance(metrics, dict) and "tpr_recall" in metrics:
                            st.write(f"  **{group}**: TPR={metrics['tpr_recall']:.4f} "
                                   f"FPR={metrics['fpr']:.4f} (n={metrics['support']})")
                    if "max_tpr_disparity" in groups:
                        st.write(f"  Disparidad maxima TPR: {groups['max_tpr_disparity']:.4f}")

        st.divider()
        st.subheader("3. Mitigaciones Recomendadas")
        mitigations = report.get("mitigations", [])
        if mitigations:
            for m in mitigations:
                st.warning(f"**{m.get('attribute', 'N/A')}** - {m.get('issue', 'N/A')}")
                for rec in m.get("recommendations", []):
                    st.write(f"  - {rec}")
        else:
            st.success("No se detectaron sesgos significativos. El modelo cumple con equidad.")


elif page == "🔔 Notificaciones":
    st.header("Centro de Notificaciones Inteligentes")

    stats = api_get("/dashboard")
    if stats:
        c1, c2 = st.columns(2)
        with c1: st.metric("Notificaciones Pendientes", stats.get("pending_notifications", 0))
        with c2: st.metric("Total Predicciones", stats.get("total_predictions", 0))

    st.divider()
    patients = api_get("/patients")
    if patients and patients.get("patients"):
        for patient in patients["patients"]:
            detail = api_get(f"/patients/{patient['id']}")
            if detail and detail.get("notifications"):
                with st.expander(f"{patient['name']} ({len(detail['notifications'])} notificaciones)"):
                    for notif in detail["notifications"]:
                        ntype = notif.get("notification_type", "")
                        if "Urgente" in ntype:
                            st.error(f"**{ntype}** - {notif['created_at']}")
                        elif "Seguimiento" in ntype:
                            st.warning(f"**{ntype}** - {notif['created_at']}")
                        else:
                            st.info(f"**{ntype}** - {notif['created_at']}")
                        st.write(notif["message"])
                        st.divider()
    else:
        st.info("No hay notificaciones. Registre pacientes primero.")
