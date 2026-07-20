import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from itertools import combinations


SUPPORT_MAP = {"Nulo": 0, "Bajo": 1, "Moderado": 2, "Alto": 3}
INCOME_MAP = {"Bajo": 0, "Medio-Bajo": 1, "Medio": 2, "Medio-Alto": 3}


def create_sensitive_groups(df):
    groups = {}
    groups["Genero"] = {
        "Masculino": df["Gender"] == "Masculino",
        "Femenino": df["Gender"] == "Femenino",
    }
    groups["Nivel Ingreso"] = {
        "Bajo": df["Income_Level"] == "Bajo",
        "Medio": df["Income_Level"].isin(["Medio-Bajo", "Medio"]),
        "Alto": df["Income_Level"] == "Medio-Alto",
    }
    groups["Rango Edad"] = {
        "60-69": df["Age"].between(60, 69),
        "70-79": df["Age"].between(70, 79),
        "80+": df["Age"] >= 80,
    }
    return groups


def compute_disparate_impact(y_true, y_pred, sensitive_groups):
    results = {}
    group_accuracies = {}

    for attr_name, group_labels in sensitive_groups.items():
        results[attr_name] = {}
        group_accuracies[attr_name] = {}

        for group_name, mask in group_labels.items():
            if mask.sum() == 0:
                continue
            acc = accuracy_score(y_true[mask], y_pred[mask])
            group_accuracies[attr_name][group_name] = round(float(acc), 4)

        groups = list(group_accuracies[attr_name].keys())
        for g1, g2 in combinations(groups, 2):
            acc1 = group_accuracies[attr_name][g1]
            acc2 = group_accuracies[attr_name][g2]
            ratio = min(acc1, acc2) / max(acc1, acc2) if max(acc1, acc2) > 0 else 0
            results[attr_name][f"{g1} vs {g2}"] = {
                "disparate_impact_ratio": round(float(ratio), 4),
                "flagged": ratio < 0.8,
            }

    return results, group_accuracies


def compute_equalized_odds(y_true, y_pred, sensitive_groups):
    results = {}

    for attr_name, group_labels in sensitive_groups.items():
        results[attr_name] = {}

        for group_name, mask in group_labels.items():
            if mask.sum() == 0:
                continue

            y_t = y_true[mask]
            y_p = y_pred[mask]

            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            results[attr_name][group_name] = {
                "tpr_recall": round(float(tpr), 4),
                "fpr": round(float(fpr), 4),
                "support": int(mask.sum()),
            }

        tprs = [v["tpr_recall"] for v in results[attr_name].values()]
        if tprs:
            results[attr_name]["max_tpr_disparity"] = round(float(max(tprs) - min(tprs)), 4)

    return results


def compute_calibration_by_group(y_true, y_pred_proba, sensitive_groups, n_bins=10):
    results = {}

    for attr_name, group_labels in sensitive_groups.items():
        results[attr_name] = {}

        for group_name, mask in group_labels.items():
            if mask.sum() == 0:
                continue

            y_t = y_true[mask]
            y_p = y_pred_proba[mask]

            bin_edges = np.linspace(0, 1, n_bins + 1)
            ece = 0.0

            for i in range(n_bins):
                in_bin = (y_p >= bin_edges[i]) & (y_p < bin_edges[i + 1])
                if in_bin.sum() > 0:
                    bin_acc = y_t[in_bin].mean()
                    bin_conf = y_p[in_bin].mean()
                    ece += in_bin.sum() / len(y_t) * abs(bin_acc - bin_conf)

            results[attr_name][group_name] = {
                "ece": round(float(ece), 4),
                "n_samples": int(mask.sum()),
            }

    return results


def run_full_fairness_audit(df, y_true, y_pred, y_pred_proba):
    sensitive = create_sensitive_groups(df)

    report = {"disparate_impact": {}, "equalized_odds": {}, "calibration": {}, "mitigations": []}

    di_results, group_accs = compute_disparate_impact(y_true, y_pred, sensitive)
    report["disparate_impact"] = {
        "by_attribute": di_results,
        "group_accuracies": group_accs,
    }

    eo_results = compute_equalized_odds(y_true, y_pred, sensitive)
    report["equalized_odds"] = eo_results

    cal_results = compute_calibration_by_group(y_true, y_pred_proba, sensitive)
    report["calibration"] = cal_results

    mitigations = suggest_mitigations(report)
    report["mitigations"] = mitigations

    return report


def suggest_mitigations(audit_results):
    mitigations = []

    di = audit_results.get("disparate_impact", {}).get("by_attribute", {})
    for attr, comparisons in di.items():
        for pair, metrics in comparisons.items():
            if metrics.get("flagged"):
                mitigations.append({
                    "attribute": attr,
                    "groups": pair,
                    "issue": "Disparate impact detectado (ratio < 0.8)",
                    "recommendations": [
                        "Revisar features que codifiquen indirectamente la variable sensible",
                        "Aplicar re-weighing o resampling estratificado por grupo",
                        "Considerar adversarial debiasing durante entrenamiento",
                        "Validar con datos de poblaciones subrepresentadas",
                    ]
                })

    for attr, groups in audit_results.get("equalized_odds", {}).items():
        disparity = groups.get("max_tpr_disparity", 0)
        if isinstance(disparity, (int, float)) and disparity > 0.1:
            mitigations.append({
                "attribute": attr,
                "issue": f"Disparidad TPR alta ({disparity:.4f})",
                "recommendations": [
                    "Ajustar umbral de decision por subgrupo",
                    "Entrenar modelos separados por subpoblacion",
                    "Aumentar representacion en datos de entrenamiento",
                ]
            })

    return mitigations
