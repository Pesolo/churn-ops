"""
Preprocessing functions for the Telco Customer Churn dataset.
"""
import pandas as pd


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    TotalCharges is loaded as object/string dtype because 11 rows contain
    blank strings instead of numbers. These blanks correspond exactly to
    customers with tenure == 0 (brand-new customers, nothing billed yet),
    so we impute 0 rather than drop the rows -- MonthlyCharges * 0 tenure
    is a derived value, not a guess.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df

def drop_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """customerID has zero predictive value -- pure row identifier."""
    df = df.copy()
    return df.drop(columns=["customerID"])


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Churn: Yes -> 1, No -> 0. 1 = churned (the event we're predicting)."""
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def simplify_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Columns like OnlineSecurity, TechSupport, StreamingTV etc. have a
    'No internet service' value that is redundant with InternetService == 'No'.
    Collapsing it to 'No' avoids one-hot encoding creating near-duplicate
    columns that carry the same information as InternetService.
    """
    df = df.copy()
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in service_cols:
        df[col] = df[col].replace("No internet service", "No")
    # MultipleLines has an analogous quirk tied to PhoneService
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    return df

def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixed Yes/No -> 1/0 mapping for binary columns. Not fit from data
    (no leakage risk), unlike one-hot encoding or scaling.
    """
    df = df.copy()
    binary_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified split on Churn -- required because of the ~73/27 class
    imbalance we found earlier. A plain random split risks skewing that
    ratio differently between train and test.
    """
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def build_preprocessor():
    """
    ColumnTransformer bundles one-hot + scaling into a single object that
    can be fit once on X_train and reused (never refit) on X_test.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical_cols = ["InternetService", "PaymentMethod", "Contract"]
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ], remainder="passthrough")