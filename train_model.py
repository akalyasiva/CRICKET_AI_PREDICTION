print("CRICKET AI TRAINING STARTED")

import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "CRICKET.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Dataset Path:", DATA_PATH)


# ------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

df = pd.read_csv(DATA_PATH)
print("Dataset Loaded:", df.shape)


# ------------------------------------------------
# MISSING VALUE HANDLING
# ------------------------------------------------

df['extras_type']    = df['extras_type'].fillna('No_Extra')
df['dismissal_kind'] = df['dismissal_kind'].fillna('No_Dismissal')


# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------

df['ball_number'] = df['over'] * 6 + df['ball']

df = df.sort_values(['match_id', 'inning', 'ball_number'])

df['current_score']  = df.groupby(['match_id', 'inning'])['total_runs'].cumsum()
df['wickets_fallen'] = df.groupby(['match_id', 'inning'])['is_wicket'].cumsum()

TOTAL_OVERS = 20

df['overs_completed'] = df['ball_number'] / 6.0
df['run_rate']        = df['current_score'] / (df['overs_completed'] + 1e-6)
df['remaining_overs'] = TOTAL_OVERS - df['overs_completed']


# ------------------------------------------------
# TARGET CREATION (WIN / LOSS)
# ------------------------------------------------

final_scores = df.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()

inning1 = final_scores[final_scores['inning'] == 1][['match_id', 'total_runs']]
inning2 = final_scores[final_scores['inning'] == 2][['match_id', 'total_runs']]

merged = inning1.merge(inning2, on='match_id', suffixes=('_1', '_2'))
merged['inning2_win'] = (merged['total_runs_2'] > merged['total_runs_1']).astype(int)

df = df.merge(merged[['match_id', 'inning2_win']], on='match_id')
df['win'] = np.where(df['inning'] == 2, df['inning2_win'], 1 - df['inning2_win'])


# ------------------------------------------------
# TEAM ENCODING
# ------------------------------------------------

label_encoders = {}

for col in ['batting_team', 'bowling_team']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))


# ------------------------------------------------
# FEATURE SELECTION
# ------------------------------------------------

features = [
    'inning',
    'batting_team',
    'bowling_team',
    'ball_number',
    'current_score',
    'wickets_fallen',
    'run_rate',
    'remaining_overs'
]

X = df[features].values
y = df['win'].values


# ------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "feature_scaler.pkl"))


# ------------------------------------------------
# SEQUENCE CREATION
# ------------------------------------------------

SEQUENCE_LENGTH = 10

X_seq = []
y_seq = []

for match in df['match_id'].unique():
    match_rows = df[df['match_id'] == match]
    X_match   = X_scaled[match_rows.index]
    y_match   = y[match_rows.index]
    for i in range(len(match_rows) - SEQUENCE_LENGTH):
        X_seq.append(X_match[i:i + SEQUENCE_LENGTH])
        y_seq.append(y_match[i + SEQUENCE_LENGTH - 1])

X_lstm = np.array(X_seq)
y_lstm = np.array(y_seq)

print("Sequence Shape:", X_lstm.shape)


# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_lstm, test_size=0.2, random_state=SEED
)


# ------------------------------------------------
# SCORE CATEGORY FUNCTION
# ------------------------------------------------

def score_category(score):
    if   score < 100: return "Low (<100)"
    elif score < 140: return "Below Par (100-139)"
    elif score < 170: return "Par (140-169)"
    elif score < 200: return "Good (170-199)"
    else:             return "Excellent (200+)"


# ------------------------------------------------
# LSTM MODEL
# ------------------------------------------------

print("\nTRAINING LSTM MODEL")

inputs = Input(shape=(SEQUENCE_LENGTH, 8))
x      = LSTM(64, return_sequences=True)(inputs)
x      = Dropout(0.2)(x)
x      = LSTM(32)(x)
x      = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

lstm_model = Model(inputs, output)
lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

lstm_model.fit(
    X_train, y_train,
    epochs=25, batch_size=64, validation_split=0.1,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

pred_lstm     = lstm_model.predict(X_test)
pred_lstm_bin = (pred_lstm > 0.5).astype(int)

print("\n----- LSTM RESULTS -----")
print("Accuracy :", accuracy_score(y_test, pred_lstm_bin))
print("Precision:", precision_score(y_test, pred_lstm_bin))
print("Recall   :", recall_score(y_test, pred_lstm_bin))
print("F1 Score :", f1_score(y_test, pred_lstm_bin))
print(classification_report(y_test, pred_lstm_bin))

# ── Score category distribution on test set ──────────────────────────────────
# Projected scores from LSTM win-probability proxy
proj_scores_lstm = []
for i in range(len(X_test)):
    # recover approximate current score from last time-step (index 4 = current_score)
    last_step   = X_test[i, -1, :]                     # shape (8,)
    curr_s_raw  = scaler.inverse_transform(last_step.reshape(1, -1))[0][4]
    rr_raw      = scaler.inverse_transform(last_step.reshape(1, -1))[0][6]
    rem_raw     = scaler.inverse_transform(last_step.reshape(1, -1))[0][7]
    wk_raw      = scaler.inverse_transform(last_step.reshape(1, -1))[0][5]
    proj        = float(np.clip(curr_s_raw + rr_raw * rem_raw * (1 - wk_raw * 0.02), 50, 300))
    proj_scores_lstm.append(proj)

cat_counts_lstm = {}
for s in proj_scores_lstm:
    c = score_category(s)
    cat_counts_lstm[c] = cat_counts_lstm.get(c, 0) + 1

print("\nLSTM — Projected Score Category Distribution on Test Set:")
for cat, cnt in sorted(cat_counts_lstm.items()):
    print(f"  {cat}: {cnt}")

lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
print("LSTM MODEL SAVED")


# ------------------------------------------------
# BILSTM MODEL
# ------------------------------------------------

print("\nTRAINING BILSTM MODEL")

inputs = Input(shape=(SEQUENCE_LENGTH, 8))
x      = Bidirectional(LSTM(64, return_sequences=True))(inputs)
x      = Dropout(0.3)(x)
x      = Bidirectional(LSTM(32))(x)
x      = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

bilstm_model = Model(inputs, output)
bilstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

bilstm_model.fit(
    X_train, y_train,
    epochs=60, batch_size=64, validation_split=0.1,
    callbacks=[EarlyStopping(patience=7, restore_best_weights=True)]
)

pred_bilstm     = bilstm_model.predict(X_test)
pred_bilstm_bin = (pred_bilstm > 0.5).astype(int)

print("\n----- BILSTM RESULTS -----")
print("Accuracy :", accuracy_score(y_test, pred_bilstm_bin))
print("Precision:", precision_score(y_test, pred_bilstm_bin))
print("Recall   :", recall_score(y_test, pred_bilstm_bin))
print("F1 Score :", f1_score(y_test, pred_bilstm_bin))
print(classification_report(y_test, pred_bilstm_bin))

# ── Score category distribution on test set ──────────────────────────────────
proj_scores_bilstm = []
for i in range(len(X_test)):
    last_step  = X_test[i, -1, :]
    curr_s_raw = scaler.inverse_transform(last_step.reshape(1, -1))[0][4]
    rr_raw     = scaler.inverse_transform(last_step.reshape(1, -1))[0][6]
    rem_raw    = scaler.inverse_transform(last_step.reshape(1, -1))[0][7]
    wk_raw     = scaler.inverse_transform(last_step.reshape(1, -1))[0][5]
    proj       = float(np.clip(curr_s_raw + rr_raw * rem_raw * (1 - wk_raw * 0.02), 50, 300))
    proj_scores_bilstm.append(proj)

cat_counts_bilstm = {}
for s in proj_scores_bilstm:
    c = score_category(s)
    cat_counts_bilstm[c] = cat_counts_bilstm.get(c, 0) + 1

print("\nBiLSTM — Projected Score Category Distribution on Test Set:")
for cat, cnt in sorted(cat_counts_bilstm.items()):
    print(f"  {cat}: {cnt}")

bilstm_model.save(os.path.join(MODEL_DIR, "bilstm_model.h5"))
print("BILSTM MODEL SAVED")

print("\nTRAINING COMPLETE")