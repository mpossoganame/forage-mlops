import pandas as pd

def load_data(filepath: str):
    # Charge les données depuis un fichier CSV
    return pd.read_csv(filepath)
