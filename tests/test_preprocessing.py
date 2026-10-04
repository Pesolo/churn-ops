import numpy as np
import pandas as pd

from preprocessing import (
    fix_total_charges, drop_identifier, encode_target,
    simplify_service_columns, encode_binary_columns,
    split_data, build_preprocessor,
)

SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
               "MultipleLines"] + SERVICE_COLS


def _dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


# ---- fixture sanity ----
def test_fixture_shape_and_balance(raw_df):
    assert raw_df.shape == (100, 21)
    assert (raw_df["Churn"] == "Yes").sum() == 27
    assert (raw_df["TotalCharges"].str.strip() == "").sum() == 1


# ---- fix_total_charges ----
def test_blank_total_charges_becomes_zero():
    out = fix_total_charges(pd.DataFrame({"TotalCharges": ["29.85", " ", "1889.5"]}))
    assert out["TotalCharges"].tolist() == [29.85, 0.0, 1889.5]


def test_numeric_strings_convert():
    out = fix_total_charges(pd.DataFrame({"TotalCharges": ["1", "2.5"]}))
    assert out["TotalCharges"].tolist() == [1.0, 2.5]


def test_total_charges_is_float():
    out = fix_total_charges(pd.DataFrame({"TotalCharges": ["29.85", " "]}))
    assert out["TotalCharges"].dtype == "float64"


def test_fix_total_charges_does_not_mutate_input():
    df = pd.DataFrame({"TotalCharges": ["29.85", " "]})
    fix_total_charges(df)
    assert df["TotalCharges"].tolist() == ["29.85", " "]


def test_garbage_total_charges_becomes_zero_too():
    # Characterization: ANY non-numeric value becomes 0, not just blanks.
    out = fix_total_charges(pd.DataFrame({"TotalCharges": ["abc"]}))
    assert out["TotalCharges"].tolist() == [0.0]


# ---- drop_identifier ----
def test_drop_identifier_removes_column_keeps_rest():
    df = pd.DataFrame({"customerID": ["a", "b"], "x": [1, 2]})
    out = drop_identifier(df)
    assert list(out.columns) == ["x"]
    assert out["x"].tolist() == [1, 2]


def test_drop_identifier_does_not_mutate_input():
    df = pd.DataFrame({"customerID": ["a"], "x": [1]})
    drop_identifier(df)
    assert "customerID" in df.columns


# ---- encode_target ----
def test_encode_target_yes_no():
    out = encode_target(pd.DataFrame({"Churn": ["Yes", "No"]}))
    assert out["Churn"].tolist() == [1, 0]


def test_encode_target_lowercase_becomes_nan():
    # Characterization: .map() silently turns unmapped values into NaN.
    out = encode_target(pd.DataFrame({"Churn": ["Yes", "yes"]}))
    assert out["Churn"].iloc[0] == 1
    assert pd.isna(out["Churn"].iloc[1])


# ---- simplify_service_columns ----
def test_simplify_collapses_no_service_values(raw_df):
    out = simplify_service_columns(raw_df)
    for col in SERVICE_COLS:
        assert set(out[col].unique()) <= {"Yes", "No"}
    assert set(out["MultipleLines"].unique()) <= {"Yes", "No"}


def test_simplify_leaves_other_values_untouched(raw_df):
    out = simplify_service_columns(raw_df)
    assert out["InternetService"].equals(raw_df["InternetService"])
    for col in SERVICE_COLS + ["MultipleLines"]:
        assert (out[col] == "Yes").equals(raw_df[col] == "Yes")


def test_simplify_does_not_mutate_input(raw_df):
    simplify_service_columns(raw_df)
    assert "No internet service" in raw_df["OnlineSecurity"].values


# ---- encode_binary_columns ----
def test_binary_columns_are_0_1_without_nan(raw_df):
    out = encode_binary_columns(simplify_service_columns(raw_df))
    for col in BINARY_COLS:
        assert not out[col].isna().any()
        assert set(out[col].unique()) <= {0, 1}


def test_gender_mapping(raw_df):
    out = encode_binary_columns(simplify_service_columns(raw_df))
    assert (out["gender"] == 1).equals(raw_df["gender"] == "Male")


def test_wrong_call_order_gives_nan(raw_df):
    # Characterization: skipping simplify_service_columns leaves NaN behind.
    out = encode_binary_columns(raw_df)
    assert out["OnlineSecurity"].isna().any()


# ---- split_data ----
def test_split_sizes(clean_df):
    X_train, X_test, y_train, y_test = split_data(clean_df)
    assert len(X_train) == 80 and len(X_test) == 20
    assert len(y_train) == 80 and len(y_test) == 20


def test_split_removes_target_from_features(clean_df):
    X_train, X_test, _, _ = split_data(clean_df)
    assert "Churn" not in X_train.columns
    assert "Churn" not in X_test.columns


def test_split_has_no_row_overlap(clean_df):
    X_train, X_test, _, _ = split_data(clean_df)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_preserves_class_ratio(clean_df):
    _, _, y_train, y_test = split_data(clean_df)
    assert abs(y_train.mean() - 0.27) < 0.05
    assert abs(y_test.mean() - 0.27) < 0.05


def test_split_is_reproducible(clean_df):
    a = split_data(clean_df)
    b = split_data(clean_df)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_split_changes_with_seed(clean_df):
    a = split_data(clean_df, random_state=42)
    b = split_data(clean_df, random_state=7)
    assert set(a[1].index) != set(b[1].index)


# ---- build_preprocessor ----
def test_preprocessor_output_shape(clean_df):
    X = clean_df.drop(columns=["Churn"])
    out = _dense(build_preprocessor().fit_transform(X))
    assert out.shape == (100, 23)  # 7 one-hot + 3 scaled + 13 passthrough


def test_preprocessor_has_no_nan(clean_df):
    X = clean_df.drop(columns=["Churn"])
    out = _dense(build_preprocessor().fit_transform(X))
    assert not np.isnan(out).any()


def test_scaled_columns_are_standardized(clean_df):
    X = clean_df.drop(columns=["Churn"])
    out = _dense(build_preprocessor().fit_transform(X))
    scaled = out[:, 7:10]  # columns after the 7 one-hot ones
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-9)


def test_unseen_category_encodes_like_dropped_level(clean_df):
    # Characterization: with drop="first" + handle_unknown="ignore", an unseen
    # level becomes all zeros, identical to the dropped first level ("DSL").
    X = clean_df.drop(columns=["Churn"])
    fitted = build_preprocessor().fit(X)
    known = X.iloc[[0]].copy()
    known["InternetService"] = "DSL"
    unseen = X.iloc[[0]].copy()
    unseen["InternetService"] = "Satellite"
    a = _dense(fitted.transform(known))
    b = _dense(fitted.transform(unseen))
    assert (b[0, :2] == 0).all()
    assert np.array_equal(a, b)