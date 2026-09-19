<<<<<<< HEAD
# 💳 Credit Card Fraud Detection

An ANN-based system for detecting fraudulent credit card transactions, with a Streamlit
web app for interactive predictions.

## Problem Statement
Credit card fraud is rare but costly — in this dataset, fraudulent transactions make up
only ~0.17% of all transactions. A naive model can score 99.8% accuracy while catching
almost no fraud. This project builds and deploys a neural network that specifically
addresses this class imbalance, and wraps it in a usable app so predictions can be checked
interactively rather than only in a notebook.

## Dataset
[Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions made by European cardholders in September 2013, 492 labeled as fraud.
Features `V1`–`V28` are PCA-transformed for confidentiality; `Time` and `Amount` are raw.

> The raw CSV (~144MB) is **not included in this repo** — download it from the link above
> and place it at `data/creditcard.csv` before training.

## Approach
1. **Preprocessing:** drop duplicates, scale `Time`/`Amount` with MinMaxScaler
2. **Split:** stratified 70/15/15 train/validation/test
3. **Imbalance handling:** SMOTE oversampling on the training set only (avoids leakage into
   validation/test), plus class weighting during training
4. **Model:** Feedforward ANN (64 → 32 → 16 → 1) with BatchNorm, Dropout, EarlyStopping,
   and learning-rate reduction on plateau
5. **Threshold tuning:** instead of the default 0.5 cutoff, the decision threshold is chosen
   via Youden's J statistic on the ROC curve, since minimizing missed fraud (false negatives)
   matters more than a generic accuracy score here
6. **Evaluation:** accuracy, precision, recall, F1, specificity, AUC-ROC, average precision

## Project Structure
```
fraud-detection-app/
├── data/                    # place creditcard.csv here (not committed)
├── model/
│   ├── train.py             # training script
│   ├── fraud_model.keras    # saved trained model (generated)
│   └── scaler.pkl           # saved scaler + threshold + feature list (generated)
├── app.py                   # Streamlit app
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
Download `creditcard.csv` from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:
```
data/creditcard.csv
```

### 3. Train the model
```bash
python model/train.py
```
This prints dataset stats, training progress, and final test-set metrics, saves a results
plot to `model/fraud_detection_results.png`, and saves the trained model + scaler for the app.

Optional flags:
```bash
python model/train.py --epochs 50 --batch_size 32 --data path/to/creditcard.csv
```

### 4. Run the app
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. Two ways to check transactions:
- **Manual Entry** — type in feature values (or load a random real example from the dataset)
- **Batch Upload** — upload a CSV of transactions and download predictions for all of them

## Results
_Fill in after running `model/train.py` on your machine — see the printed metrics and
`model/fraud_detection_results.png`._

| Metric | Score |
|---|---|
| Accuracy | |
| Precision | |
| Recall | |
| F1-Score | |
| AUC-ROC | |

## Future Approach
- Try gradient-boosted models (XGBoost/LightGBM) as a lighter-weight comparison to the ANN
- Add SHAP-based explainability so flagged transactions come with a reason
- Move from a single static threshold to a cost-sensitive decision rule
- Real-time/streaming inference instead of batch-only prediction
- Model monitoring for concept drift as spending patterns change over time

## Authors
Muhammad Saim (23-CS-23), Zaeem Mehmood (23-CS-151) — UET Taxila
=======
# Zynvex_Fraud_Detection_Model
Credit card fraud detection app built with TensorFlow &amp; Streamlit — ANN model trained with SMOTE on the ULB fraud dataset (0.95 AUC-ROC).
<<<<<<< HEAD
>>>>>>> 75fa5ccd04dfababe3092b577f96062b133b01fb
=======
Download the dataset from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place it at `data/creditcard.csv` before running `model/train.py`.
>>>>>>> 555aa21c32c1ed64591cb1d3d9040619f2e0c23e
