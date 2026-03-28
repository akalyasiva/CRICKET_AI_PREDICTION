"""
utils/lime_engine.py
LIME explainability for the trained LSTM/BiLSTM cricket models.
SEQUENCE_LENGTH must match train_model.py (10).
"""
import lime
import lime.lime_tabular
import numpy as np

SEQUENCE_LENGTH = 10
FEATURE_NAMES   = [
    "Inning", "Batting Team", "Bowling Team",
    "Ball Number", "Current Score", "Wickets Fallen",
    "Run Rate", "Remaining Overs",
]


def _make_predict_fn(model):
    """
    LIME passes flat (N, 8) arrays.
    Model needs (N, SEQ_LEN, 8).
    Returns (N, 2) → [P(LOSS), P(WIN)] for LIME classifier mode.
    """
    def predict_fn(X_flat):
        X_seq = np.repeat(X_flat[:, np.newaxis, :], SEQUENCE_LENGTH, axis=1)
        preds = model.predict(X_seq, verbose=0).flatten()
        return np.column_stack([1 - preds, preds])
    return predict_fn


def create_lime_explainer(X_train_flat):
    """
    Build a LimeTabularExplainer.

    Parameters
    ----------
    X_train_flat : np.ndarray shape (N, 8) — scaled training/background samples

    Returns
    -------
    lime.lime_tabular.LimeTabularExplainer
    """
    return lime.lime_tabular.LimeTabularExplainer(
        training_data        = X_train_flat,
        feature_names        = FEATURE_NAMES,
        class_names          = ["LOSS", "WIN"],
        mode                 = "classification",
        discretize_continuous= True,
        random_state         = 42,
    )


def local_lime_explanation(explainer, model, X_sample_flat, num_features=8):
    """
    Explain a single prediction with LIME.

    Parameters
    ----------
    X_sample_flat : np.ndarray shape (8,) or (1, 8)

    Returns
    -------
    dict:
        weights      : dict {feature_str: weight}
        intercept    : float
        local_pred   : float — LIME's local WIN probability
        model_pred   : float — model's actual WIN probability
        feature_names: list[str]
        explanation  : raw LIME Explanation object
    """
    predict_fn = _make_predict_fn(model)
    sample_1d  = X_sample_flat.flatten()

    exp = explainer.explain_instance(
        sample_1d,
        predict_fn,
        num_features = num_features,
        num_samples  = 500,
        labels       = (1,)       # explain WIN class
    )

    weights = dict(exp.as_list(label=1))

    # Safely extract local prediction value
    lp = exp.local_pred
    if hasattr(lp, '__len__'):
        local_pred_val = float(lp[1]) if len(lp) > 1 else float(lp[0])
    else:
        local_pred_val = float(lp)

    intercept_val  = float(exp.intercept.get(1, exp.intercept.get(0, 0.0)))
    model_pred_val = float(predict_fn(sample_1d.reshape(1, -1))[0, 1])

    return {
        "weights":       weights,
        "intercept":     intercept_val,
        "local_pred":    local_pred_val,
        "model_pred":    model_pred_val,
        "feature_names": FEATURE_NAMES,
        "explanation":   exp,
    }


def global_lime_importance(explainer, model, X_test_flat, n_samples=30):
    """
    Average |LIME weights| over n_samples to get approximate global importance.

    Returns
    -------
    dict:
        mean_abs_weight: np.ndarray (8,)
        all_weights    : np.ndarray (n_samples, 8)
        feature_names  : list[str]
    """
    predict_fn  = _make_predict_fn(model)
    all_weights = []

    for i in range(min(n_samples, len(X_test_flat))):
        exp = explainer.explain_instance(
            X_test_flat[i], predict_fn,
            num_features=8, num_samples=300, labels=(1,)
        )
        row = np.zeros(8)
        for feat_str, w in exp.as_list(label=1):
            for j, fname in enumerate(FEATURE_NAMES):
                if fname.lower().replace(" ", "") in feat_str.lower().replace(" ", ""):
                    row[j] = w
                    break
        all_weights.append(row)

    arr = np.array(all_weights)
    return {
        "mean_abs_weight": np.mean(np.abs(arr), axis=0),
        "all_weights":     arr,
        "feature_names":   FEATURE_NAMES,
    }