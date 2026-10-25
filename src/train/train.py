import os
import joblib
import mlflow
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath('mlflow.db')}")
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, recall_score, precision_score, f1_score,
)

from src.data.load import load_data
from src.features.build import get_preprocessor

SKOPS_TRUSTED = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


def eval_and_log(name, pipeline, X_train, y_train, X_test, y_test, params):
    with mlflow.start_run(run_name=name):
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)

        metrics = {
            "pr_auc":    average_precision_score(y_test, proba),
            "recall":    recall_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "f1":        f1_score(y_test, pred),
        }

        print(f"\n[{name}]")
        for k, v in metrics.items():
            print(f"  {k:10s} = {v:.4f}")

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("threshold", 0.5)

        os.makedirs("models", exist_ok=True)
        pkl_path = f"models/{name}.pkl"
        joblib.dump(pipeline, pkl_path)
        print(f"  SAVE {pkl_path}")

        try:
            mlflow.sklearn.log_model(
                pipeline, "model",
                registered_model_name=f"forage-{name}",
                skops_trusted_types=SKOPS_TRUSTED,
            )
        except Exception as e:
            print(f"  MLflow log KO ({e}), on garde juste le .pkl local.")

        return metrics


def train():
    mlflow.set_experiment("forage-panne-90j")

    df = load_data("data/raw/forages_historique.csv")
    X = df.drop(columns=["panne_90j"])
    y = df["panne_90j"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Taux de pannes (train): {y_train.mean():.2%}")

    pipe_lr = Pipeline([
        ("prep", get_preprocessor()),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    eval_and_log("logreg", pipe_lr, X_train, y_train, X_test, y_test,
                 {"model": "LogisticRegression", "class_weight": "balanced"})

    pipe_rf = Pipeline([
        ("prep", get_preprocessor()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced", random_state=42, n_jobs=1
        )),
    ])
    eval_and_log("randomforest", pipe_rf, X_train, y_train, X_test, y_test,
                 {"model": "RandomForest", "n_estimators": 200, "max_depth": 10})

    pipe_hgb = Pipeline([
        ("prep", get_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42, max_iter=200
        )),
    ])
    eval_and_log("histgb", pipe_hgb, X_train, y_train, X_test, y_test,
                 {"model": "HistGradientBoosting", "max_iter": 200})

    print("\nFIN. Verifie MLflow UI et models/.")


if __name__ == "__main__":
    train()
