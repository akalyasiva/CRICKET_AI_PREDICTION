"""
utils/prediction_engine.py
Models are loaded ONCE at import time via @st.cache_resource in the page.
Call load_models() from the page — it returns all four objects.
Never retrain. Never reload.
"""
import os, numpy as np, joblib
import tensorflow as tf

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SEQ       = 10   # must match train_model.py SEQUENCE_LENGTH

BUCKET_NAMES = ["Low (<100)", "Below Par (100-139)", "Par (140-169)",
                "Good (170-199)", "Excellent (200+)"]

def _bucket(score):
    if   score < 100: return 0, "Low (<100)"
    elif score < 140: return 1, "Below Par (100-139)"
    elif score < 170: return 2, "Par (140-169)"
    elif score < 200: return 3, "Good (170-199)"
    else:             return 4, "Excellent (200+)"

# ── This is what the page calls inside @st.cache_resource ─────────────────────
def load_models():
    """Load all four artefacts from disk. Call once, cache the result."""
    for fname in ["lstm_model.h5", "bilstm_model.h5",
                  "feature_scaler.pkl", "label_encoders.pkl"]:
        if not os.path.exists(os.path.join(MODEL_DIR, fname)):
            raise FileNotFoundError(
                f"Missing: models/{fname}  — run python train_model.py first.")
    lstm   = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"),   compile=False)
    bilstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "bilstm_model.h5"), compile=False)
    scaler = joblib.load(os.path.join(MODEL_DIR, "feature_scaler.pkl"))
    enc    = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    return lstm, bilstm, scaler, enc

def get_teams(enc):
    return sorted(enc["batting_team"].classes_.tolist())

def preprocess(inning, batting, bowling, ball_no, score, wkts, rr, rem, scaler, enc):
    bat_e = enc["batting_team"].transform([batting])[0]
    bow_e = enc["bowling_team"].transform([bowling])[0]
    raw   = np.array([[float(inning), float(bat_e), float(bow_e),
                        float(ball_no), float(score), float(wkts),
                        float(rr), float(rem)]])
    return scaler.transform(raw)   # shape (1, 8)

def predict(X_flat, lstm, bilstm, scaler):
    """
    X_flat : (1, 8)  — output of preprocess()
    Returns dict with win prob + score category.
    No model reloading — uses objects passed in.
    """
    X_seq    = np.repeat(X_flat[:, np.newaxis, :], SEQ, axis=1)  # (1, SEQ, 8)

    lstm_p   = float(lstm.predict(X_seq,   verbose=0)[0][0])
    bilstm_p = float(bilstm.predict(X_seq, verbose=0)[0][0])
    win_p    = (lstm_p + bilstm_p) / 2.0

    # Projected score from raw feature values
    raw       = scaler.inverse_transform(X_flat)[0]
    curr_s, rr_v, rem_v, wk_v = raw[4], raw[6], raw[7], raw[5]
    proj      = float(np.clip(curr_s + rr_v * rem_v * (1 - wk_v * 0.02), 50, 300))
    bkt_i, bkt_name = _bucket(proj)

    return {
        "prediction":        "WIN" if win_p >= 0.5 else "LOSS",
        "win_probability":   win_p,
        "loss_probability":  1.0 - win_p,
        "lstm_probability":  lstm_p,
        "bilstm_probability":bilstm_p,
        "predicted_score":   proj,
        "score_bucket":      bkt_i,
        "score_bucket_name": bkt_name,
    }