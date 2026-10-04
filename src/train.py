"""
Baseline model training script for Telco Customer Churn.
Supports Logistic Regression and Random Forest via --model flag.
Logs params, metrics, confusion matrix, and the full pipeline to
DagsHub-hosted MLflow.
"""
import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models import infer_signature
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


def build_pipeline(model_type: str) -> Pipeline:
    """
    Builds the full preprocessing + classifier pipeline.
    Note: RF doesn't need StandardScaler, but we reuse build_preprocessor()
    unchanged for both models so the comparison is apples-to-apples
    (same train/test features feeding both). This is a deliberate choice,
    not an oversight.
    """
    if model_type == "logreg":
        classifier = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000, random_state=42
        )
    elif model_type == "rf":
        # Conservative defaults deliberately chosen over sklearn's
        # unregularized defaults (n_estimators=100, max_depth=None) to
        # reduce overfit risk on a 7043-row dataset. Tune from evidence
        # (train/test gap) after this first run, not before.
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def log_model_params(model_type: str, classifier) -> None:
    mlflow.log_param("model_type", model_type)
    if model_type == "logreg":
        mlflow.log_param("C", classifier.C)
        mlflow.log_param("class_weight", classifier.class_weight)
    elif model_type == "rf":
        mlflow.log_param("n_estimators", classifier.n_estimators)
        mlflow.log_param("max_depth", classifier.max_depth)
        mlflow.log_param("min_samples_leaf", classifier.min_samples_leaf)
        mlflow.log_param("class_weight", classifier.class_weight)


def log_feature_importances(pipeline: Pipeline) -> None:
    """
    Extracts feature names post-one-hot-encoding from the ColumnTransformer
    and logs RF feature importances as a bar chart artifact.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, max(4, len(imp_df) * 0.3)))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("Importance")
    plt.title("RF Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importances.png")
    mlflow.log_artifact("feature_importances.png")
    plt.close()


def run(model_type: str = "logreg"):
    """
    Core training logic, callable directly (e.g. from a notebook via
    train.run(model_type="rf")) without touching sys.argv.
    """
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("churn-baseline")

    df = load_and_clean("../data/raw/telco_churn.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline(model_type)

    with mlflow.start_run(run_name=f"{model_type}-baseline"):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        log_model_params(model_type, pipeline.named_steps["classifier"])
        mlflow.log_metrics(metrics)

        # Also log train-set roc_auc so we can eyeball overfit gap for RF
        y_train_proba = pipeline.predict_proba(X_train)[:, 1]
        mlflow.log_metric("train_roc_auc", roc_auc_score(y_train, y_train_proba))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
        disp.plot()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        if model_type == "rf":
            log_feature_importances(pipeline)

        # Signature is inferred from the *raw* input (X_train, pre-pipeline)
        # and the pipeline's own predictions, since the pipeline includes
        # preprocessing internally. Without this, the registry showed
        # Inputs (0)/Outputs (0) with no dtype/shape validation at serving
        # time -- exactly the gap that produced a raw sklearn stack trace
        # instead of a clear error when X_test wasn't preprocessed correctly
        # in an earlier debugging session.
        signature = infer_signature(X_train, y_pred)
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            pipeline,
            "model",
            signature=signature,
            input_example=input_example,
        )
        print("Metrics:", metrics)


def main():
    """CLI entry point. Not used when calling train.run() from a notebook."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["logreg", "rf"], default="logreg",
        help="Which model to train (default: logreg)"
    )
    args = parser.parse_args()
    run(model_type=args.model)


if __name__ == "__main__":
    main()