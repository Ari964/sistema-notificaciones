import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def render_shap_waterfall(shap_values: dict, risk_level: str):
    if not shap_values.get("values"):
        st.warning("Valores SHAP no disponibles")
        return

    values = np.array(shap_values["values"])
    names = shap_values["feature_names"]

    st.subheader("Impacto de Variables en la Prediccion (SHAP)")

    sorted_idx = np.argsort(np.abs(values))[::-1][:10]
    features_sorted = [names[i] for i in sorted_idx]
    values_sorted = [values[i] for i in sorted_idx]
    colors = ["#d32f2f" if v < 0 else "#388e3c" for v in values_sorted]

    fig = go.Figure(go.Bar(
        x=values_sorted,
        y=features_sorted,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values_sorted],
        textposition='outside'
    ))
    fig.update_layout(
        title=f"Variables que influyen en la clasificacion - Riesgo: {risk_level}",
        xaxis_title="Impacto SHAP (positivo = favorece adherencia)",
        yaxis=dict(autorange="reversed"),
        height=500,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Interpretacion para el medico:**")
    top_3 = [(names[i], values[i]) for i in sorted_idx[:3]]
    for fname, fval in top_3:
        direction = "aumenta" if fval > 0 else "reduce"
        st.markdown(
            f"- **{fname}**: {direction} la probabilidad de adherencia "
            f"en **{abs(fval):.4f}** unidades"
        )


def render_shap_comparison_bar(importance_list: list):
    if not importance_list:
        return

    df = pd.DataFrame(importance_list[:10])
    fig = px.bar(
        df, x="importance", y="feature", orientation="h",
        color="direction",
        color_discrete_map={"positivo": "#388e3c", "negativo": "#d32f2f"},
        labels={"importance": "Importancia Absoluta", "feature": "Variable"},
        title="Top 10 Variables Mas Influyentes"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_patient_risk_gauge(probability: float, risk_level: str):
    color_map = {"Bajo Riesgo": "#388e3c", "Riesgo Moderado": "#f57c00", "Alto Riesgo": "#d32f2f"}
    color = color_map.get(risk_level, "#666")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 36}},
        title={"text": f"Nivel: {risk_level}", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#ffebee"},
                {"range": [50, 80], "color": "#fff3e0"},
                {"range": [80, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": probability * 100
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison_chart(metrics_dict: dict):
    model_names = [k for k in metrics_dict.keys() if k not in ("best_model", "best_ensemble_model")]
    if not model_names:
        return

    rows = []
    for name in model_names:
        m = metrics_dict[name]
        if isinstance(m, dict) and "accuracy" in m:
            rows.append({
                "Modelo": name,
                "Accuracy": m.get("accuracy", 0),
                "Precision": m.get("precision", 0),
                "Recall": m.get("recall", 0),
                "F1-Score": m.get("f1_score", 0),
                "AUC-ROC": m.get("auc_roc", 0),
            })

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig = go.Figure()
    for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]:
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Modelo"],
            y=df[metric],
            text=df[metric].round(3),
            textposition='auto'
        ))
    fig.update_layout(barmode='group', height=500, yaxis_title="Valor", xaxis_title="Modelo")
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrices(metrics_dict: dict):
    model_names = [k for k in metrics_dict.keys()
                   if k not in ("best_model", "best_ensemble_model")
                   and isinstance(metrics_dict[k], dict)
                   and "confusion_matrix" in metrics_dict[k]]
    if not model_names:
        return

    cols = st.columns(min(len(model_names), 3))
    for i, name in enumerate(model_names):
        cm = metrics_dict[name]["confusion_matrix"]
        with cols[i % len(cols)]:
            st.write(f"**{name}**")
            fig = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicho", y="Real"),
                           x=["No Adherente", "Adherente"],
                           y=["No Adherente", "Adherente"],
                           color_continuous_scale="Blues")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
