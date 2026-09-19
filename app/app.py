import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

MODEL_PATH = os.path.join('model', 'fraud_model.keras')
SCALER_PATH = os.path.join('model', 'scaler.pkl')

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    bundle = joblib.load(SCALER_PATH)
    return model, bundle['scaler'], bundle['threshold'], bundle['feature_columns']


model, scaler, threshold, feature_columns = load_artifacts()

st.title("💳 Credit Card Fraud Detection")
st.caption("ANN-based fraud detector trained on the ULB Credit Card Fraud dataset (SMOTE + class weighting).")

if model is None:
    st.error(
        "No trained model found. Please run `python model/train.py` first "
        "(after placing `creditcard.csv` inside the `data/` folder) to generate "
        "`model/fraud_model.keras` and `model/scaler.pkl`."
    )
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔢 Manual Entry", "📁 Batch Upload (CSV)", "ℹ️ About"])


def predict(df_features: pd.DataFrame):
    """df_features must contain exactly `feature_columns`, in order, unscaled Time/Amount."""
    df = df_features.copy()
    df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
    X = df[feature_columns].values
    probs = model.predict(X, verbose=0).ravel()
    preds = (probs >= threshold).astype(int)
    return preds, probs


# Manual entry
with tab1:
    st.subheader("Check a single transaction")
    st.write(
        "Enter the transaction's `Time`, `Amount`, and the 28 PCA-transformed "
        "features (`V1`–`V28`) exactly as they appear in the dataset. "
        "If you don't have real values handy, use **Load Random Test Example** below."
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        use_sample = st.button("🎲 Load Random Test Example")

    if use_sample and os.path.exists(os.path.join('data', 'creditcard.csv')):
        sample_row = pd.read_csv(os.path.join('data', 'creditcard.csv')).sample(1).iloc[0]
        for c in feature_columns:
            st.session_state[f"input_{c}"] = float(sample_row[c])
        if 'Class' in sample_row:
            st.session_state['_last_sample_true_label'] = (
                'Fraud' if sample_row['Class'] == 1 else 'Legitimate'
            )

    if st.session_state.get('_last_sample_true_label'):
        st.info(f"Loaded a real example — true label: "
                f"{st.session_state['_last_sample_true_label']} (hidden from the model).")

    with st.form("manual_predict_form"):
        cols = st.columns(4)
        inputs = {}
        for i, feat in enumerate(feature_columns):
            with cols[i % 4]:
                inputs[feat] = st.number_input(
                    feat, format="%.6f", key=f"input_{feat}"
                )
        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        row = pd.DataFrame([inputs])[feature_columns]
        preds, probs = predict(row)
        pred, prob = preds[0], probs[0]

        st.divider()
        if pred == 1:
            st.error(f"🚨 **FRAUD DETECTED** — probability: {prob:.4f}")
        else:
            st.success(f"✅ **Legitimate transaction** — fraud probability: {prob:.4f}")
        st.progress(min(float(prob), 1.0))

# Batch CSV upload
with tab2:
    st.subheader("Check multiple transactions from a CSV")
    st.write(
        "Upload a CSV containing the columns "
        f"`{', '.join(feature_columns)}` (an optional `Class` column is ignored)."
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in feature_columns if c not in batch_df.columns]
            if missing:
                st.error(f"CSV is missing required columns: {missing}")
            else:
                preds, probs = predict(batch_df)
                result_df = batch_df.copy()
                result_df['Fraud_Prediction'] = preds
                result_df['Fraud_Probability'] = probs

                n_fraud = int(preds.sum())
                st.write(f"**{n_fraud} of {len(result_df)}** transactions flagged as fraud.")

                fraud_rows = result_df[result_df['Fraud_Prediction'] == 1]

                st.markdown(f"**🚨 Flagged fraud transactions ({len(fraud_rows)}):**")
                if len(fraud_rows) > 0:
                    st.dataframe(
                        fraud_rows.style.apply(
                            lambda r: ['background-color: #ffcccc' for _ in r], axis=1
                        ),
                        use_container_width=True
                    )
                else:
                    st.write("None found in this file.")

                with st.expander(f"View all {len(result_df)} transactions (unstyled)"):
                    st.dataframe(result_df, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download results as CSV", csv_out,
                                    "fraud_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process file: {e}")

# About
with tab3:
    st.subheader("About this project")
    st.markdown(f"""
    - **Model:** Artificial Neural Network (64-32-16-1, dropout + batch normalization)
    - **Dataset:** [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
      (284,807 transactions, 492 fraud cases — ~0.17%)
    - **Imbalance handling:** SMOTE oversampling (training set only) + class weighting
    - **Decision threshold:** {threshold:.4f} (tuned via Youden's J statistic on the ROC curve,
      instead of the default 0.5, to balance catching fraud vs. false alarms)
    - **Note:** `V1`–`V28` are PCA-transformed features from the original dataset;
      their real-world meaning is anonymized for confidentiality.
    """)