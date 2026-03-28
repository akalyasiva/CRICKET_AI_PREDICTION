"""
utils/shap_engine.py
SHAP explainability for the trained LSTM/BiLSTM models.
Uses KernelExplainer so it works with any black-box Keras model.
SEQUENCE_LENGTH must match train_model.py (10).
"""
import shap
import numpy as np

SEQUENCE_LENGTH = 10
FEATURE_NAMES   = [
    "Inning", "Batting Team", "Bowling Team",
    "Ball Number", "Current Score", "Wickets Fallen",
    "Run Rate", "Remaining Overs",
]


def _make_predict_fn(model):
    """Wrap model so KernelExplainer can call it with flat (N, 8) arrays."""
    def predict_fn(X_flat):
        # KernelExplainer passes (N, 8) → model needs (N, SEQ_LEN, 8)
        X_seq = np.repeat(X_flat[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
        preds = model.predict(X_seq, verbose=0)
        return preds.flatten()   # shape (N,)
    return predict_fn


def create_shap_explainer(model, X_background_flat):
    """
    Build a KernelExplainer using background data.

    Parameters
    ----------
    model              : loaded Keras LSTM or BiLSTM model
    X_background_flat  : np.ndarray shape (N, 8) — scaled background samples
                         (use ~30-100 rows for speed)
    Returns
    -------
    shap.KernelExplainer
    """
    predict_fn = _make_predict_fn(model)
    # Use k-means summary to keep background small and fast
    n_clusters = min(20, len(X_background_flat))
    bg         = shap.kmeans(X_background_flat, n_clusters)
    return shap.KernelExplainer(predict_fn, bg)


def local_shap_values(explainer, X_sample_flat):
    """
    Explain a single prediction.

    Parameters
    ----------
    X_sample_flat : np.ndarray shape (1, 8) — single scaled sample

    Returns
    -------
    dict:
        shap_values  : np.ndarray (8,)
        base_value   : float
        feature_names: list[str]
    """
    values = explainer.shap_values(X_sample_flat, nsamples=100)
    if isinstance(values, list):
        sv = np.array(values[0]).flatten()
    else:
        sv = np.array(values).flatten()

    bv = explainer.expected_value
    if isinstance(bv, (list, np.ndarray)):
        bv = float(bv[0])
    else:
        bv = float(bv)

    return {
        "shap_values":  sv[:8],           # guard against shape surprises
        "base_value":   bv,
        "feature_names": FEATURE_NAMES,
    }


def global_shap_importance(explainer, X_test_flat, n_samples=30):
    """
    Compute global (average) feature importance over multiple samples.

    Parameters
    ----------
    X_test_flat : np.ndarray shape (N, 8)
    n_samples   : int — how many rows to use (more = more accurate but slower)

    Returns
    -------
    dict:
        mean_abs_shap : np.ndarray (8,)
        all_shap_vals : np.ndarray (n_samples, 8)
        feature_names : list[str]
    """
    X_sub  = X_test_flat[:n_samples]
    values = explainer.shap_values(X_sub, nsamples=100)
    if isinstance(values, list):
        sv = np.array(values[0])
    else:
        sv = np.array(values)

    if sv.ndim == 3:               # unexpected extra dim
        sv = sv.reshape(sv.shape[0], -1)

    sv = sv[:, :8]                 # keep 8 features

    return {
        "mean_abs_shap": np.mean(np.abs(sv), axis=0),
        "all_shap_vals": sv,
        "feature_names": FEATURE_NAMES,
    }