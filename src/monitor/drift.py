import json
import os
import numpy as np
import pandas as pd


def compute_psi(expected, actual, buckets=10):
    """Calcule le PSI entre 2 distributions (en capturant les outliers)."""
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Breakpoints basés sur les quantiles de la référence
    inner_bounds = np.unique(
        np.percentile(expected, np.linspace(0, 100, buckets + 1))
    )
    if len(inner_bounds) < 2:
        return 0.0

    # ✅ On étend avec -inf et +inf pour capturer les outliers
    breakpoints = np.concatenate([[-np.inf], inner_bounds, [np.inf]])

    expected_pct = pd.cut(expected, bins=breakpoints, include_lowest=True) \
                     .value_counts(normalize=True, dropna=False).sort_index()
    actual_pct = pd.cut(actual, bins=breakpoints, include_lowest=True) \
                   .value_counts(normalize=True, dropna=False).sort_index()

    # Éviter les zéros (log)
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)

    psi = float(((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum())
    return round(psi, 4)


def psi_to_status(psi):
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "attention"
    return "derive"


def detect_drift(histo_path: str, new_path: str, output_dir: str = "reports"):
    df_histo = pd.read_csv(histo_path)
    df_new = pd.read_csv(new_path)

    numeric_cols = [
        "age_ans", "profondeur_m", "debit_m3h", "usagers_jour",
        "jours_depuis_maintenance", "pannes_12m",
    ]

    results = {}
    for col in numeric_cols:
        psi = compute_psi(df_histo[col], df_new[col])
        results[col] = {
            "psi": psi,
            "status": psi_to_status(psi),
            "mean_ref": round(float(df_histo[col].mean()), 2),
            "mean_new": round(float(df_new[col].mean()), 2),
        }

    drifted = [c for c, r in results.items() if r["psi"] >= 0.2]
    warnings = [c for c, r in results.items() if 0.1 <= r["psi"] < 0.2]

    if len(drifted) >= 2:
        action = "REENTRAINEMENT RECOMMANDE"
    elif len(drifted) == 1 or len(warnings) >= 2:
        action = "SURVEILLANCE RENFORCEE"
    else:
        action = "PAS D ACTION"

    report = {
        "features": results,
        "drifted_features": drifted,
        "warnings": warnings,
        "action": action,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "drift_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== RAPPORT DE DERIVE ===")
    for feat, r in results.items():
        print(f"  {feat:30s} PSI={r['psi']:.4f} [{r['status']}]")
    print(f"\n  Decision : {action}")
    print(f"  Rapport  : {output_path}\n")

    return report


if __name__ == "__main__":
    detect_drift(
        "data/raw/forages_historique.csv",
        "data/raw/forages_nouveau_mois.csv",
    )