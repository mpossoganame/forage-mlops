import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_FEATURES = ["age_ans", "profondeur_m", "debit_m3h", "usagers_jour",
                    "jours_depuis_maintenance", "pannes_12m"]
CATEGORICAL_FEATURES = ["type_pompe", "region"]

def get_preprocessor() -> ColumnTransformer:
    # Retourne le pipeline de préparation des données
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
