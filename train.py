import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd


def main():
    print("=" * 60)
    print("  PIPELINE COMPLETO DE ENTRENAMIENTO Y COMPARACION")
    print("  Sistema Inteligente de Notificaciones Electronicas v2.0")
    print("=" * 60)

    data_path = os.path.join(os.path.dirname(__file__), "data", "medication_adherence.csv")

    # Paso 1: Generar dataset si no existe
    if not os.path.exists(data_path):
        print("\n[1/6] Generando dataset sintetico...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "data", "generate_data.py")])
    else:
        print("\n[1/6] Dataset existente encontrado.")

    # Paso 2: Cargar datos
    print("\n[2/6] Cargando dataset...")
    df = pd.read_csv(data_path)
    print(f"  Registros: {len(df)}")
    print(f"  Distribucion: {df['Adherence'].value_counts().to_dict()}")

    # Paso 3: Preprocesamiento con feature engineering avanzado
    print("\n[3/6] Preprocesando con feature engineering avanzado...")
    from backend.preprocessing import preprocess_training, create_advanced_features
    df_feat = create_advanced_features(df)
    print(f"  Features originales: 11 -> Features con ingenieria: {df_feat.shape[1] - 1}")

    X, y = preprocess_training(df)
    print(f"  Shape de X: {X.shape}")

    # Paso 4: Entrenar modelos neuronales
    print("\n[4/6] Entrenando modelos neuronales (CNN, LSTM, GRU, CNN-LSTM, CNN-GRU)...")
    from backend.model import train_models
    X_nn = X.reshape(X.shape[0], X.shape[1], 1)
    nn_metrics = train_models(X_nn, y, epochs=50, batch_size=32)

    # Paso 5: Comparar con modelos de ensamble
    print("\n[5/6] Comparando con modelos de ensamble (XGBoost, RF, LightGBM, etc.)...")
    from backend.ml_comparator import run_ensemble_comparison
    ensemble_results = run_ensemble_comparison(df)

    # Paso 6: Auditoria de equidad
    print("\n[6/6] Ejecutando auditoria de equidad...")
    from backend.fairness_audit import run_full_fairness_audit
    from backend.preprocessing import load_encoders, load_scaler
    from backend.model import get_best_model

    model = get_best_model()
    if model is not None:
        y_pred_prob = model.predict(X_nn, verbose=0).flatten()
        y_pred = (y_pred_prob >= 0.5).astype(int)
        audit = run_full_fairness_audit(df, y, y_pred, y_pred_prob)
        print(f"  Mitigaciones recomendadas: {len(audit.get('mitigations', []))}")

    # Resumen final
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETADO")
    print("=" * 60)

    print(f"\n  Mejor modelo neuronal: {nn_metrics.get('best_model', 'N/A')}")
    best_ensemble = ensemble_results.get("best_ensemble_model", "N/A")
    best_ensemble_auc = ensemble_results.get(best_ensemble, {}).get("auc_roc", 0)
    print(f"  Mejor modelo de ensamble: {best_ensemble} (AUC-ROC: {best_ensemble_auc:.4f})")

    print("\n  Metricas neuronales:")
    for name, m in nn_metrics.items():
        if name != "best_model":
            print(f"    {name}: Acc={m['accuracy']:.4f} AUC={m['auc_roc']:.4f} F1={m['f1_score']:.4f}")

    print("\n  Metricas de ensamble:")
    for name in ensemble_results:
        if name != "best_ensemble_model":
            m = ensemble_results[name]
            if isinstance(m, dict) and "accuracy" in m:
                print(f"    {name}: Acc={m['accuracy']:.4f} AUC={m['auc_roc']:.4f} F1={m['f1_score']:.4f}")

    print("\n  Archivos generados en /models/:")
    print("    - best_model.keras (mejor modelo neuronal)")
    print("    - metrics.json (metricas neuronales)")
    print("    - ensemble_comparison.json (comparacion de ensambles)")
    print("    - encoders.pkl, scaler.pkl (preprocesadores)")


if __name__ == "__main__":
    main()
