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