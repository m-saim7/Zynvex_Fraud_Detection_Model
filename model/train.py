# ============================================================
#  Credit Card Fraud Detection — ANN Model (Training Script)
#  Refactored from Google Colab notebook for standalone/local use
# ============================================================
#
#  HOW TO USE:
#  1. Place creditcard.csv inside the ../data/ folder
#     (dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
#  2. From the project root, run:  python model/train.py
#  3. Trained model -> model/fraud_model.keras
#     Scaler         -> model/scaler.pkl
#     Results plot   -> model/fraud_detection_results.png
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed, just save the figure
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'creditcard.csv')
MODEL_OUT_PATH = os.path.join(SCRIPT_DIR, 'fraud_model.keras')
SCALER_OUT_PATH = os.path.join(SCRIPT_DIR, 'scaler.pkl')
PLOT_OUT_PATH = os.path.join(SCRIPT_DIR, 'fraud_detection_results.png')


def parse_args():
    parser = argparse.ArgumentParser(description='Train the fraud detection ANN.')
    parser.add_argument('--data', type=str, default=DEFAULT_DATA_PATH,
                         help='Path to creditcard.csv')
    parser.add_argument('--epochs', type=int, default=50,
                         help='Max training epochs (EarlyStopping may stop sooner)')
    parser.add_argument('--batch_size', type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"TensorFlow version : {tf.__version__}")
    print(f"GPU available      : {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("Libraries loaded ✓\n")

    # ── Load dataset ────────────────────────────────────────
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Dataset not found at '{args.data}'.\n"
            f"Download it from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place creditcard.csv inside the data/ folder, or pass --data <path>."
        )

    df = pd.read_csv(args.data)

    print("=" * 50)
    print("          DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total rows       : {len(df):,}")
    print(f"Total columns    : {df.shape[1]}")
    print(f"Missing values   : {df.isnull().sum().sum()}")
    print(f"Duplicate rows   : {df.duplicated().sum():,}")
    fraud = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]
    print(f"Legitimate (0)   : {len(legit):,}  ({len(legit)/len(df)*100:.3f}%)")
    print(f"Fraudulent  (1)  : {len(fraud):,}   ({len(fraud)/len(df)*100:.3f}%)")
    print("=" * 50)

    # ── Preprocessing ───────────────────────────────────────
    df = df.drop_duplicates()
    print(f"\nAfter removing duplicates: {len(df):,} rows")

    scaler = MinMaxScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
    print("Amount & Time normalized ✓")

    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    feature_columns = df.drop('Class', axis=1).columns.tolist()

    # ── Train / Val / Test split (70/15/15) ─────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
    )

    print(f"\nData Split (Stratified 70/15/15):")
    print(f"  Train      : {len(X_train):,} samples  | fraud: {y_train.sum()}")
    print(f"  Validation : {len(X_val):,} samples  | fraud: {y_val.sum()}")
    print(f"  Test       : {len(X_test):,} samples  | fraud: {y_test.sum()}")

    # ── SMOTE (training set only) ───────────────────────────
    print(f"\nBefore SMOTE — fraud: {y_train.sum():,}  |  legit: {(y_train==0).sum():,}")
    smote = SMOTE(sampling_strategy=0.5, random_state=SEED, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print(f"After  SMOTE — fraud: {y_train_sm.sum():,}  |  legit: {(y_train_sm==0).sum():,}")

    idx = np.random.permutation(len(X_train_sm))
    X_train_sm = X_train_sm[idx]
    y_train_sm = y_train_sm[idx]

    # ── Build ANN ────────────────────────────────────────────
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_sm.shape[1],),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation='relu', kernel_initializer='he_normal'),

        Dense(1, activation='sigmoid')
    ], name='ANN_FraudDetector')

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    print("\nModel Architecture:")
    model.summary()

    # ── Train ────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=4, min_lr=1e-6, verbose=1)
    ]

    cw_vals = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_sm)
    class_weights = {0: cw_vals[0], 1: cw_vals[1]}

    print("\nStarting training...\n")
    history = model.fit(
        X_train_sm, y_train_sm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    print(f"\nTraining done — {len(history.history['loss'])} epochs completed ✓")

    # ── Evaluate ─────────────────────────────────────────────
    y_prob = model.predict(X_test, verbose=0).ravel()

    fpr_arr, tpr_arr, thresholds = roc_curve(y_test, y_prob)
    j_scores = tpr_arr - fpr_arr
    best_thresh = float(thresholds[np.argmax(j_scores)])
    print(f"\nOptimal threshold (Youden's J): {best_thresh:.4f}")

    y_pred = (y_prob >= best_thresh).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp)

    print()
    print("=" * 55)
    print("     FINAL RESULTS — TEST SET")
    print("=" * 55)
    print(f"  Accuracy         : {acc*100:.4f}%")
    print(f"  Precision        : {prec*100:.4f}%")
    print(f"  Recall           : {rec*100:.4f}%")
    print(f"  F1-Score         : {f1*100:.4f}%")
    print(f"  Specificity      : {spec*100:.4f}%")
    print(f"  AUC-ROC          : {auc:.6f}")
    print(f"  Avg Precision    : {ap:.6f}")
    print("-" * 55)
    print(f"  True  Positives  : {tp}   (Fraud correctly caught)")
    print(f"  True  Negatives  : {tn:,}  (Legit correctly passed)")
    print(f"  False Positives  : {fp}   (Legit wrongly flagged)")
    print(f"  False Negatives  : {fn}   (Fraud missed)")
    print("=" * 55)

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Legitimate', 'Fraud'], digits=4))

    # ── Plots ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('ANN Credit Card Fraud Detection — Results',
                 fontsize=14, fontweight='bold')

    ep = range(1, len(history.history['loss']) + 1)

    ax = axes[0, 0]
    ax.plot(ep, history.history['loss'], 'b-o', ms=3, label='Train')
    ax.plot(ep, history.history['val_loss'], 'r--s', ms=3, label='Validation')
    ax.set_title('Loss (Binary Cross-Entropy)', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, history.history['accuracy'], 'b-o', ms=3, label='Train')
    ax.plot(ep, history.history['val_accuracy'], 'r--s', ms=3, label='Validation')
    ax.set_title('Accuracy over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(ep, history.history['auc'], 'b-o', ms=3, label='Train')
    ax.plot(ep, history.history['val_auc'], 'r--s', ms=3, label='Validation')
    ax.set_title('AUC-ROC over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'],
                linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

    ax = axes[1, 1]
    ax.plot(fpr_arr, tpr_arr, color='#1565C0', lw=2, label=f'AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr_arr[best_idx], tpr_arr[best_idx], s=80,
               color='red', zorder=5, label=f'Best threshold={best_thresh:.3f}')
    ax.fill_between(fpr_arr, tpr_arr, alpha=0.08, color='#1565C0')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Specificity']
    values = [acc, prec, rec, f1, auc, spec]
    colors = ['#1565C0', '#2E7D32', '#F57F17', '#6A1B9A', '#AD1457', '#00695C']
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylim(0, 1.13)
    ax.set_title('All Performance Metrics', fontweight='bold')
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_OUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {PLOT_OUT_PATH}")

    # ── Save model + scaler + threshold (for the app) ───────
    model.save(MODEL_OUT_PATH)
    joblib.dump({
        'scaler': scaler,
        'threshold': best_thresh,
        'feature_columns': feature_columns
    }, SCALER_OUT_PATH)
    print(f"Model saved to  {MODEL_OUT_PATH}")
    print(f"Scaler + threshold + feature list saved to {SCALER_OUT_PATH}")
    print("\nAll done. You can now run the Streamlit app: streamlit run app.py")


if __name__ == '__main__':
    main()
