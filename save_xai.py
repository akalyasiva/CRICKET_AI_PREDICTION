"""
save_xai.py  —  Run ONCE after  python save_metrics.py
═══════════════════════════════════════════════════════
    python save_xai.py

Saves to  models/xai_cache.json:
  • background_X    — 50 flat test samples for building explainers
  • global_shap     — mean |SHAP| across 30 samples (both models)
  • global_lime     — mean |weight| across 30 samples (both models)

Local SHAP + LIME are computed LIVE in the page for any user-chosen scenario.
"""

import os, json, time
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "datasets", "CRICKET.csv")
OUT_PATH  = os.path.join(MODEL_DIR, "xai_cache.json")

SEQ  = 10
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FEATURE_NAMES = [
    "Inning", "Batting Team", "Bowling Team", "Ball Number",
    "Current Score", "Wickets Fallen", "Run Rate", "Remaining Overs",
]

print("Loading artefacts …")
scaler   = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
lstm_m   = tf.keras.models.load_model(
               os.path.join(MODEL_DIR, "lstm_model.h5"),   compile=False)
bilstm_m = tf.keras.models.load_model(
               os.path.join(MODEL_DIR, "bilstm_model.h5"), compile=False)

print("Rebuilding test set …")
df = pd.read_csv(DATA_PATH)
df['extras_type']    = df['extras_type'].fillna('No_Extra')
df['dismissal_kind'] = df['dismissal_kind'].fillna('No_Dismissal')
df['ball_number']    = df['over'] * 6 + df['ball']
df = df.sort_values(['match_id', 'inning', 'ball_number'])
df['current_score']  = df.groupby(['match_id','inning'])['total_runs'].cumsum()
df['wickets_fallen'] = df.groupby(['match_id','inning'])['is_wicket'].cumsum()
df['overs_completed']= df['ball_number'] / 6.0
df['run_rate']       = df['current_score'] / (df['overs_completed'] + 1e-6)
df['remaining_overs']= 20 - df['overs_completed']

final = df.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
inn1  = final[final['inning']==1][['match_id','total_runs']]
inn2  = final[final['inning']==2][['match_id','total_runs']]
mg    = inn1.merge(inn2, on='match_id', suffixes=('_1','_2'))
mg['inning2_win'] = (mg['total_runs_2'] > mg['total_runs_1']).astype(int)
df    = df.merge(mg[['match_id','inning2_win']], on='match_id')
df['win'] = np.where(df['inning']==2, df['inning2_win'], 1 - df['inning2_win'])

for col in ['batting_team', 'bowling_team']:
    le     = encoders[col]
    df[col] = df[col].apply(
        lambda x: le.transform([x])[0] if x in le.classes_ else 0)

feats = ['inning','batting_team','bowling_team','ball_number',
         'current_score','wickets_fallen','run_rate','remaining_overs']
X_sc  = scaler.transform(df[feats].values)
y     = df['win'].values

X_seq, y_seq = [], []
for mid in df['match_id'].unique():
    idx = df[df['match_id']==mid].index.tolist()
    Xm  = X_sc[[df.index.get_loc(i) for i in idx]]
    ym  = y[[df.index.get_loc(i) for i in idx]]
    for i in range(len(idx) - SEQ):
        X_seq.append(Xm[i:i+SEQ])
        y_seq.append(ym[i+SEQ-1])

X_all = np.array(X_seq); y_all = np.array(y_seq)
_, X_test, _, _ = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED)

X_flat = X_test[:, -1, :]   # (N, 8) last timestep

from utils.shap_engine import create_shap_explainer, global_shap_importance
from utils.lime_engine import create_lime_explainer, global_lime_importance

N_BG   = 50    # background samples for KernelExplainer
N_GLOB = 30    # samples for global importance

cache = {
    "feature_names": FEATURE_NAMES,
    # Background samples — page reloads explainers from these
    "background_X":  X_flat[:N_BG].tolist(),
}

for model, mname in [(lstm_m, "lstm"), (bilstm_m, "bilstm")]:
    print(f"\n── {mname.upper()} ──")
    e = {}

    print("  Building KernelExplainer + global SHAP …")
    t0       = time.time()
    shap_exp = create_shap_explainer(model, X_flat[:N_BG])
    g_sh     = global_shap_importance(shap_exp, X_flat, n_samples=N_GLOB)
    print(f"  ✓ SHAP {time.time()-t0:.0f}s")
    e["global_shap"] = {
        "mean_abs_shap": g_sh["mean_abs_shap"].tolist(),
        "all_shap_vals": g_sh["all_shap_vals"].tolist(),
        "X_samples":     X_flat[:N_GLOB].tolist(),
    }

    print("  Building LIME explainer + global LIME …")
    t0       = time.time()
    lime_exp = create_lime_explainer(X_flat)
    g_lm     = global_lime_importance(lime_exp, model, X_flat, n_samples=N_GLOB)
    print(f"  ✓ LIME {time.time()-t0:.0f}s")
    e["global_lime"] = {
        "mean_abs_weight": g_lm["mean_abs_weight"].tolist(),
        "all_weights":     g_lm["all_weights"].tolist(),
    }

    cache[mname] = e
    print(f"  {mname.upper()} done ✓")

with open(OUT_PATH, "w") as f:
    json.dump(cache, f, indent=2)

print(f"\n✅  Saved → {OUT_PATH}")
print("   The analytics page reads global values from this file.")
print("   Local SHAP + LIME are computed live per user scenario.")