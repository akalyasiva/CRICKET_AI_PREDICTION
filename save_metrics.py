"""
save_metrics.py
===============
Run ONCE after training:
    python save_metrics.py

Loads the saved models, evaluates on test data,
saves all metrics + probability arrays to  models/metrics.json.
The Streamlit pages read that JSON — zero model loading at page open.
"""
import os, json, random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "datasets", "CRICKET.csv")
OUT_PATH  = os.path.join(MODEL_DIR, "metrics.json")

SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

BUCKET_NAMES = ["Low (<100)", "Below Par (100-139)", "Par (140-169)",
                "Good (170-199)", "Excellent (200+)"]

def score_category(score):
    if   score < 100: return "Low (<100)"
    elif score < 140: return "Below Par (100-139)"
    elif score < 170: return "Par (140-169)"
    elif score < 200: return "Good (170-199)"
    else:             return "Excellent (200+)"

print("Loading artefacts...")
scaler   = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))

print("Rebuilding test set (same split as training)...")
df = pd.read_csv(DATA_PATH)
df['extras_type']    = df['extras_type'].fillna('No_Extra')
df['dismissal_kind'] = df['dismissal_kind'].fillna('No_Dismissal')
df['ball_number']    = df['over'] * 6 + df['ball']
df = df.sort_values(['match_id','inning','ball_number'])
df['current_score']  = df.groupby(['match_id','inning'])['total_runs'].cumsum()
df['wickets_fallen'] = df.groupby(['match_id','inning'])['is_wicket'].cumsum()
df['overs_completed']= df['ball_number'] / 6.0
df['run_rate']       = df['current_score'] / (df['overs_completed'] + 1e-6)
df['remaining_overs']= 20 - df['overs_completed']

final  = df.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
inn1   = final[final['inning']==1][['match_id','total_runs']]
inn2   = final[final['inning']==2][['match_id','total_runs']]
mg     = inn1.merge(inn2, on='match_id', suffixes=('_1','_2'))
mg['inning2_win'] = (mg['total_runs_2'] > mg['total_runs_1']).astype(int)
df     = df.merge(mg[['match_id','inning2_win']], on='match_id')
df['win'] = np.where(df['inning']==2, df['inning2_win'], 1-df['inning2_win'])

for col in ['batting_team','bowling_team']:
    le = encoders[col]
    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)

features = ['inning','batting_team','bowling_team','ball_number',
            'current_score','wickets_fallen','run_rate','remaining_overs']
X_scaled = scaler.transform(df[features].values)
y        = df['win'].values

SEQ = 10
X_seq, y_seq = [], []
for match in df['match_id'].unique():
    idx  = df[df['match_id']==match].index.tolist()
    Xm   = X_scaled[[df.index.get_loc(i) for i in idx]]
    ym   = y[[df.index.get_loc(i) for i in idx]]
    for i in range(len(idx) - SEQ):
        X_seq.append(Xm[i:i+SEQ])
        y_seq.append(ym[i+SEQ-1])

X_lstm = np.array(X_seq)
y_lstm = np.array(y_seq)
_, X_test, _, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=SEED)
print(f"Test set: {len(X_test)} samples")

print("Loading trained models...")
lstm_model   = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"),   compile=False)
bilstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_model.h5"), compile=False)

def evaluate(model, name):
    print(f"  Evaluating {name}...")
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)

    # Projected scores from last time-step features
    proj_scores = []
    for i in range(len(X_test)):
        raw     = scaler.inverse_transform(X_test[i, -1, :].reshape(1,-1))[0]
        curr_s  = float(raw[4]); rr = float(raw[6]); rem = float(raw[7]); wk = float(raw[5])
        proj    = float(np.clip(curr_s + rr * rem * (1 - wk * 0.02), 50, 300))
        proj_scores.append(proj)

    # Per-bucket accuracy
    bucket_acc = {}
    for bn in BUCKET_NAMES:
        idx_b = [i for i,s in enumerate(proj_scores) if score_category(s) == bn]
        if idx_b:
            ba = float(accuracy_score(y_test[idx_b], preds[idx_b])) * 100
            bucket_acc[bn] = round(ba, 2)

    return {
        "accuracy":   round(float(accuracy_score(y_test, preds))  * 100, 2),
        "precision":  round(float(precision_score(y_test, preds, zero_division=0)) * 100, 2),
        "recall":     round(float(recall_score(y_test, preds, zero_division=0))    * 100, 2),
        "f1":         round(float(f1_score(y_test, preds, zero_division=0))        * 100, 2),
        "probs":      probs[:600].tolist(),   # 600 samples for histogram
        "bucket_acc": bucket_acc,
    }

lstm_metrics   = evaluate(lstm_model,   "LSTM")
bilstm_metrics = evaluate(bilstm_model, "BiLSTM")

out = {
    "lstm":       lstm_metrics,
    "bilstm":     bilstm_metrics,
    "seq_length": SEQ,
}

with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nSaved → {OUT_PATH}")
print(f"LSTM   Accuracy: {lstm_metrics['accuracy']}%  F1: {lstm_metrics['f1']}%")
print(f"BiLSTM Accuracy: {bilstm_metrics['accuracy']}%  F1: {bilstm_metrics['f1']}%")
print("\nDone. The Streamlit app now reads metrics instantly from the JSON.")