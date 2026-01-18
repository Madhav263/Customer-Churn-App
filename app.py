import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

DATASET_PATH = "telco-churn.csv"
MODEL_PATH = "model.pkl"
FEATURES_PATH = "features.pkl"

st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📉 Customer Churn Prediction & Analysis System")

# -----------------------------
# UTILITY: Normalize Churn Values
# -----------------------------
def normalize_churn(series):
    """
    Convert any possible churn value to 0/1
    supported: yes/no, y/n, stayed/churned, 1/0, true/false
    """
    series = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "1": 1, "true": 1, "churned": 1,
        "no": 0, "n": 0, "0": 0, "false": 0, "stayed": 0
    }
    series = series.map(mapping)
    return series

# -----------------------------
# TRAIN & SAVE MODEL
# -----------------------------
def train_and_save_model():
    df = pd.read_csv(DATASET_PATH)

    if "Churn" not in df.columns:
        st.error("❌ Dataset must contain 'Churn' column")
        st.stop()

    df["Churn"] = normalize_churn(df["Churn"])
    df = df[df["Churn"].isin([0, 1])]

    if df.shape[0] == 0:
        st.error("❌ No valid rows found in 'Churn' column after mapping.\n"
                 "Allowed values: yes/no, y/n, 1/0, true/false, stayed/churned")
        st.stop()

    # Fill numeric NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaN
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    if X.shape[0] == 0:
        st.error("❌ No valid data left for training after preprocessing")
        st.stop()

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    return model, list(X.columns)

# -----------------------------
# LOAD OR TRAIN
# -----------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
    st.warning("⚠️ Model not found. Training automatically...")
    model, expected_features = train_and_save_model()
    st.success("✅ Model trained successfully")
else:
    model = joblib.load(MODEL_PATH)
    expected_features = joblib.load(FEATURES_PATH)

# -----------------------------
# FILE UPLOAD & PREDICTION
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    if "Churn" in df.columns:
        df = df.drop("Churn", axis=1)

    # Fill numeric NaN
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical NaN
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Encode & align features
    df = pd.get_dummies(df)
    df = df.reindex(columns=expected_features, fill_value=0)

    if st.button("🔮 Predict Churn"):
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)

        # Individual predictions
        result = pd.DataFrame({
            "Churn Probability": probs,
            "Prediction": np.where(preds == 1, "Will Churn ❌", "Will Stay ✅")
        })
        st.subheader("📊 Individual Predictions")
        st.dataframe(result)

        # Summary stats
        total_users = len(df)
        total_churn = preds.sum()
        total_stay = total_users - total_churn
        avg_churn_prob = probs.mean()

        st.subheader("📈 Summary Stats")
        st.markdown(f"- **Total Users:** {total_users}")
        st.markdown(f"- **Predicted to Churn:** {total_churn}")
        st.markdown(f"- **Predicted to Stay:** {total_stay}")
        st.markdown(f"- **Average Churn Probability:** {avg_churn_prob:.2f}")

        # High-risk analysis
        high_risk_idx = np.where(probs >= 0.7)[0]  # threshold 70%
        st.subheader("⚠️ High-Risk Users (>70% probability)")
        if len(high_risk_idx) > 0:
            high_risk = result.iloc[high_risk_idx].copy()
            # Suggested solution
            high_risk["Suggested Action"] = "Offer discount/loyalty program/personalized support"
            # Estimated chance of retention if solution applied
            high_risk["Retention Probability"] = 0.5 + 0.5 * (1 - high_risk["Churn Probability"])
            st.dataframe(high_risk)
        else:
            st.info("No high-risk users detected (all <70% probability)")

else:
    st.info("⬆️ Upload a CSV file to start prediction")













