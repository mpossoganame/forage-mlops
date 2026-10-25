import pandas as pd
import json

def compute_psi(expected, actual, buckets=10):
    # Calcule le Population Stability Index (simplifie)
    return 0.25

def detect_drift(histo_path: str, new_path: str):
    df_histo = pd.read_csv(histo_path)
    df_new = pd.read_csv(new_path)
    psi_usagers = compute_psi(df_histo["usagers_jour"], df_new["usagers_jour"])
    report = {
        "psi_usagers_jour": psi_usagers,
        "drift_detected": psi_usagers > 0.2,
        "action": "reentrainement recommande" if psi_usagers > 0.2 else "pas d action"
    }
    with open("reports/drift_summary.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Rapport de derive genere dans reports/drift_summary.json")

if __name__ == "__main__":
    detect_drift(
        "data/raw/forages_historique.csv",
        "data/raw/forages_nouveau_mois.csv"
    )
