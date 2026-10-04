"""Shared fixtures: a deterministic, raw-shaped copy of the Telco CSV."""
import pandas as pd
import pytest

from preprocessing import (
    fix_total_charges, drop_identifier, encode_target,
    simplify_service_columns, encode_binary_columns,
)


def _pick(levels, i, offset):
    return levels[(i + offset) % len(levels)]


@pytest.fixture
def raw_df():
    """100 rows, all 21 raw columns, exactly 27 churners, exactly 1 blank TotalCharges."""
    yn = ["Yes", "No"]
    svc = ["Yes", "No", "No internet service"]
    rows = []
    for i in range(100):
        tenure = 0 if i == 0 else 1 + (i % 72)
        monthly = round(20 + i * 0.7, 2)
        total = " " if tenure == 0 else str(round(monthly * tenure, 2))
        rows.append({
            "customerID": f"C{i:04d}",
            "gender": _pick(["Male", "Female"], i, 0),
            "SeniorCitizen": i % 2,
            "Partner": _pick(yn, i, 1),
            "Dependents": _pick(yn, i, 2),
            "tenure": tenure,
            "PhoneService": _pick(yn, i, 3),
            "MultipleLines": _pick(["Yes", "No", "No phone service"], i, 0),
            "InternetService": _pick(["DSL", "Fiber optic", "No"], i, 1),
            "OnlineSecurity": _pick(svc, i, 0),
            "OnlineBackup": _pick(svc, i, 1),
            "DeviceProtection": _pick(svc, i, 2),
            "TechSupport": _pick(svc, i, 0),
            "StreamingTV": _pick(svc, i, 1),
            "StreamingMovies": _pick(svc, i, 2),
            "Contract": _pick(["Month-to-month", "One year", "Two year"], i, 2),
            "PaperlessBilling": _pick(yn, i, 0),
            "PaymentMethod": _pick(
                ["Electronic check", "Mailed check",
                 "Bank transfer (automatic)", "Credit card (automatic)"], i, 0),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            # 37 is coprime with 100, so this hits exactly 27 values below 27
            "Churn": "Yes" if (i * 37) % 100 < 27 else "No",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def clean_df(raw_df):
    """Raw frame after the full cleaning chain (assumed order; see train.py)."""
    df = fix_total_charges(raw_df)
    df = drop_identifier(df)
    df = encode_target(df)
    df = simplify_service_columns(df)
    df = encode_binary_columns(df)
    return df