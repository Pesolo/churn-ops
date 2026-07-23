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