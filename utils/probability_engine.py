"""utils/probability_engine.py — Probability helper functions."""


def confidence_level(prob: float) -> str:
    """Return a plain-English confidence label for a win probability."""
    if prob >= 0.85 or prob <= 0.15:
        return "Very High"
    elif prob >= 0.70 or prob <= 0.30:
        return "High"
    elif prob >= 0.55 or prob <= 0.45:
        return "Moderate"
    else:
        return "Low"


def stability_index(win_prob: float, loss_prob: float) -> float:
    """How decisive the prediction is — 1.0 = perfectly certain, 0.0 = 50-50."""
    return abs(win_prob - loss_prob)


def model_agreement(lstm_pred, bilstm_pred) -> str:
    """Check whether both models agree on WIN or LOSS."""
    lstm_win   = lstm_pred   >= 0.5 if isinstance(lstm_pred,   float) else lstm_pred   == "WIN"
    bilstm_win = bilstm_pred >= 0.5 if isinstance(bilstm_pred, float) else bilstm_pred == "WIN"
    return "Strong Agreement ✅" if lstm_win == bilstm_win else "Weak Agreement ⚠️"