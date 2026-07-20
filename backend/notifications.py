from datetime import datetime, timedelta


RISK_STRATEGIES = {
    "Alto Riesgo": {
        "tipo": "Notificacion Urgente",
        "frecuencia_horas": 8,
        "canales": ["SMS", "Llamada telefonica", "Push App"],
        "max_intentos": 3,
        "escalacion": "Contactar medico tratante en 24h",
        "template_mensaje": (
            "Recordatorio urgente: {nombre}, es hora de tomar su medicamento "
            "({medicamento}, {dosis}mg). Su nivel de adherencia requiere atencion "
            "inmediata. Si tiene dificultades, responda este mensaje."
        ),
    },
    "Riesgo Moderado": {
        "tipo": "Notificacion de Seguimiento",
        "frecuencia_horas": 24,
        "canales": ["SMS", "Push App"],
        "max_intentos": 2,
        "escalacion": "Reevaluar en 3 dias",
        "template_mensaje": (
            "Hola {nombre}, le recordamos tomar su {medicamento} ({dosis}mg). "
            "Si necesita ayuda con su tratamiento, responda INFO."
        ),
    },
    "Bajo Riesgo": {
        "tipo": "Notificacion Informativa",
        "frecuencia_horas": 168,
        "canales": ["Push App"],
        "max_intentos": 1,
        "escalacion": "Ninguna",
        "template_mensaje": (
            "Buenas noticias {nombre}! Siga manteniendo su rutina de "
            "{medicamento}. Proximo recordatorio la proxima semana."
        ),
    },
}


def classify_risk(probability: float) -> str:
    if probability >= 0.8:
        return "Bajo Riesgo"
    elif probability >= 0.5:
        return "Riesgo Moderado"
    else:
        return "Alto Riesgo"


def generate_notification_strategy(risk_level, adherence_prob, patient_data):
    strategy = RISK_STRATEGIES.get(risk_level, RISK_STRATEGIES["Riesgo Moderado"])

    nombre = patient_data.get("name", "Paciente")
    medicamento = patient_data.get("medication_type", "medicamento")
    dosis = patient_data.get("dosage", 0)

    mensaje = strategy["template_mensaje"].format(
        nombre=nombre, medicamento=medicamento, dosis=dosis
    )

    adjustments = []
    age = patient_data.get("age", 70)
    comorbidities = patient_data.get("num_comorbidities", 1)
    severity = patient_data.get("disease_severity", "Moderada")

    if age >= 80:
        adjustments.append("Priorizar llamada telefonica por edad avanzada.")
    if comorbidities >= 4:
        adjustments.append("Coordinar esquema de polifarmacia simplificado.")
    if severity in ["Severa", "Muy severa"]:
        adjustments.append("Escalar a equipo de salud en caso de no respuesta.")
    if patient_data.get("social_support") == "Nulo":
        adjustments.append("Activar red de apoyo comunitario.")
    if patient_data.get("income_level") == "Bajo":
        adjustments.append("Verificar acceso a medicamentos programados.")

    if adjustments:
        mensaje += " ADICIONAL: " + " ".join(adjustments)

    next_notification = datetime.now() + timedelta(hours=strategy["frecuencia_horas"])

    return {
        "tipo_notificacion": strategy["tipo"],
        "frecuencia": f"Cada {strategy['frecuencia_horas']}h",
        "canales": strategy["canales"],
        "max_intentos": strategy["max_intentos"],
        "escalacion": strategy["escalacion"],
        "mensaje": mensaje,
        "siguiente_notificacion": next_notification.isoformat(),
        "ajustes_dinamicos": adjustments,
        "acciones_recomendadas": _get_actions(risk_level, patient_data),
    }


def _get_actions(risk_level, data):
    actions = {
        "Alto Riesgo": [
            "Contactar telefonicamente en las proximas 4 horas",
            "Programar cita de seguimiento en 48 horas",
            "Evaluar simplificacion del esquema farmacologico",
            "Verificar acceso y disponibilidad de medicamentos",
            "Coordinar apoyo familiar o social",
        ],
        "Riesgo Moderado": [
            "Enviar recordatorio personalizado de medicacion",
            "Programar seguimiento semanal",
            "Evaluar barreras de adherencia",
            "Refuerzo educativo sobre tratamiento",
        ],
        "Bajo Riesgo": [
            "Mantener seguimiento rutinario mensual",
            "Refuerzo positivo de conductas",
            "Proxima revision en 30 dias",
        ],
    }
    return actions.get(risk_level, actions["Riesgo Moderado"])


def format_notification_for_display(notification_data):
    lines = [
        f"Tipo: {notification_data['tipo_notificacion']}",
        f"Frecuencia: {notification_data['frecuencia']}",
        f"Canales: {', '.join(notification_data['canales'])}",
        f"Max. Intentos: {notification_data.get('max_intentos', 'N/A')}",
        f"Escalacon: {notification_data.get('escalacion', 'N/A')}",
        "",
        f"Mensaje:",
        f"  {notification_data['mensaje']}",
    ]
    if notification_data.get("ajustes_dinamicos"):
        lines.append("")
        lines.append("Ajustes dinamicos:")
        for adj in notification_data["ajustes_dinamicos"]:
            lines.append(f"  - {adj}")
    lines.append("")
    lines.append("Acciones Recomendadas:")
    for i, accion in enumerate(notification_data.get('acciones_recomendadas', []), 1):
        lines.append(f"  {i}. {accion}")
    return "\n".join(lines)
