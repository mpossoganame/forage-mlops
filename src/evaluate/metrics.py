from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score
)

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
