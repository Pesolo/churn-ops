"""
Baseline Logistic Regression training script for Telco Customer Churn.
Logs params, metrics, confusion matrix, and the full pipeline to
DagsHub-hosted MLflow.
"""
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)

from preprocessing import (
    fix_total_charges, drop_identifier, encode_target,
    simplify_service_columns, encode_binary_columns,
    split_data, build_preprocessor
)


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = fix_total_charges(df)
    df = drop_identifier(df)
    df = encode_target(df)
    df = simplify_service_columns(df)
    df = encode_binary_columns(df)
    return df


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("churn-baseline")

    df = load_and_clean("../data/raw/telco_churn.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        ))
    ])

    with mlflow.start_run(run_name="logreg-baseline"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", 1.0)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        mlflow.sklearn.log_model(pipeline, "model")

        print("Metrics:", metrics)


if __name__ == "__main__":
    main()