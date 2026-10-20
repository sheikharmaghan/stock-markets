import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="PSX Stock Prediction", page_icon="📈", layout="wide")

MODEL_PATH = "psx_decision_tree_model.pkl"

TARGET_CANDIDATES = [
    "close", "closing_price", "close_price", "closing price",
    "close price", "adj_close", "adj close", "adjusted_close",
]
LEAKAGE_WORDS = [
    "change", "change_percent", "change_pct", "daily_return", "return_percent",
]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def find_target(columns):
    normalized = {c.lower().strip().replace(" ", "_"): c for c in columns}
    for cand in TARGET_CANDIDATES:
        if cand in normalized:
            return normalized[cand]
    return None


def get_feature_lists(model):
    """Read expected input columns and their types from the saved pipeline."""
    pre = model.named_steps["preprocessor"]
    numeric, categorical = [], []
    for name, _, cols in pre.transformers_:
        if name == "numeric":
            numeric = list(cols)
        elif name == "categorical":
            categorical = list(cols)
    expected = list(getattr(pre, "feature_names_in_", numeric + categorical))
    return expected, numeric, categorical


def prepare_input(df, expected, numeric, categorical, target_col):
    """Apply the same cleaning steps used during training."""
    X = df.copy()
    X.columns = X.columns.str.strip()
    X = X.replace([np.inf, -np.inf], np.nan)

    if target_col and target_col in X.columns:
        X = X.drop(columns=[target_col])

    # Date/time columns -> year, month, day (same as training)
    for col in list(X.columns):
        if "date" in col.lower() or "time" in col.lower():
            parsed = pd.to_datetime(X[col], errors="coerce")
            if parsed.notna().mean() >= 0.5:
                X[col + "_year"] = parsed.dt.year
                X[col + "_month"] = parsed.dt.month
                X[col + "_day"] = parsed.dt.day
                X = X.drop(columns=[col])

    # Drop leakage columns (same as training)
    leak = [
        c for c in X.columns
        if any(w in c.lower().replace(" ", "_") for w in LEAKAGE_WORDS)
    ]
    X = X.drop(columns=leak)

    # Make sure every column the model expects exists
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = np.nan if c in numeric else pd.Series([np.nan] * len(X), dtype="object")

    X = X[expected]

    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        X[c] = X[c].astype("object")

    return X, missing


# ---------------------------------------------------------------- UI
st.title("📈 Pakistan Stock Market Prediction")
st.write(
    "Upload a CSV containing the same input features used to train the "
    "Decision Tree model. The app will predict closing prices."
)

try:
    model = load_model()
except Exception as e:
    st.error("Model loading failed. Check the model file and package versions.")
    st.code(str(e))
    st.stop()

expected, numeric, categorical = get_feature_lists(model)

with st.expander("Columns the model expects"):
    st.write(expected)

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()

    raw.columns = raw.columns.str.strip()
    st.subheader("Uploaded data")
    st.dataframe(raw.head(20), use_container_width=True)

    target_col = find_target(raw.columns)
    X, missing = prepare_input(raw, expected, numeric, categorical, target_col)

    if missing:
        st.warning(
            "These columns were not found in your CSV and were filled with "
            "empty values (predictions may be less accurate): "
            + ", ".join(missing)
        )

    try:
        preds = model.predict(X)
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.stop()

    result = raw.copy()
    result["Predicted_Closing_Price"] = preds

    st.subheader("Predictions")
    st.dataframe(result, use_container_width=True)

    # If the CSV also has the real closing price, show accuracy
    if target_col:
        actual = pd.to_numeric(raw[target_col], errors="coerce")
        ok = actual.notna()
        if ok.sum() > 0:
            from sklearn.metrics import (
                mean_absolute_error, mean_squared_error, r2_score,
            )
            a, p = actual[ok], preds[ok.values]
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mean_absolute_error(a, p):.4f}")
            c2.metric("RMSE", f"{np.sqrt(mean_squared_error(a, p)):.4f}")
            c3.metric("R² Score", f"{r2_score(a, p):.4f}")

            chart_df = pd.DataFrame(
                {"Actual": a.values, "Predicted": p}
            ).reset_index(drop=True)
            st.subheader("Actual vs Predicted")
            st.line_chart(chart_df)
    else:
        st.subheader("Predicted closing price")
        st.line_chart(pd.DataFrame({"Predicted": preds}))

    st.download_button(
        "Download predictions as CSV",
        result.to_csv(index=False).encode("utf-8"),
        file_name="psx_predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Please upload a CSV file to get predictions.")
