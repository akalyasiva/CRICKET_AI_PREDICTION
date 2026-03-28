"""
utils/commentary_engine.py
Generates plain-English AI commentary from model prediction data.
No model re-training — uses already-computed probabilities and SHAP values.
"""

FEATURE_PLAIN = {
    "Inning":          "which innings it is",
    "Batting Team":    "the batting team",
    "Bowling Team":    "the bowling team",
    "Ball Number":     "how many balls have been bowled",
    "Current Score":   "the runs on the board",
    "Wickets Fallen":  "the number of batters who are out",
    "Run Rate":        "how fast the batting side is scoring",
    "Remaining Overs": "how many overs are still left",
}

FEATURE_NAMES = [
    "Inning", "Batting Team", "Bowling Team",
    "Ball Number", "Current Score", "Wickets Fallen",
    "Run Rate", "Remaining Overs",
]


# ── Core commentary generator ──────────────────────────────────────────────────

def generate_commentary(win_prob, score, wickets, run_rate, remaining_overs,
                        batting_team="the batting team", bowling_team="the bowling side",
                        lstm_prob=None, bilstm_prob=None):
    """
    Returns a plain-English paragraph explaining the current match situation.

    Parameters
    ----------
    win_prob         : float [0-1]  — overall win probability for batting team
    score            : int          — current runs
    wickets          : int          — wickets fallen
    run_rate         : float        — current run rate
    remaining_overs  : float        — overs left
    batting_team     : str
    bowling_team     : str
    lstm_prob        : float | None — LSTM individual probability
    bilstm_prob      : float | None — BiLSTM individual probability
    """
    pct   = round(win_prob * 100, 1)
    lines = []

    # ── Situation summary ──
    if win_prob >= 0.80:
        lines.append(
            f"{batting_team} are in a very strong position right now. "
            f"The AI gives them a {pct}% chance of winning from here."
        )
    elif win_prob >= 0.65:
        lines.append(
            f"{batting_team} have the upper hand, with a {pct}% win chance. "
            f"Things are going their way, but the match is not over yet."
        )
    elif win_prob >= 0.50:
        lines.append(
            f"It is a close contest. {batting_team} just edge it with a {pct}% chance, "
            f"but {bowling_team} can still turn this around."
        )
    elif win_prob >= 0.35:
        lines.append(
            f"{batting_team} are under pressure. The AI gives them only a {pct}% chance — "
            f"{bowling_team} currently have the advantage."
        )
    else:
        lines.append(
            f"{batting_team} are in real trouble. With just a {pct}% chance of winning, "
            f"they need something special to get back into this match."
        )

    # ── Run rate comment ──
    if run_rate >= 10.0:
        lines.append(
            f"They are scoring at {run_rate:.1f} runs per over — a blazing pace that is "
            f"putting a lot of pressure on {bowling_team}."
        )
    elif run_rate >= 7.5:
        lines.append(
            f"A run rate of {run_rate:.1f} per over is healthy and keeping the innings on track."
        )
    elif run_rate >= 5.5:
        lines.append(
            f"The current run rate is {run_rate:.1f} per over — an average pace. "
            f"They will need to push harder in the later overs."
        )
    else:
        lines.append(
            f"A run rate of only {run_rate:.1f} per over is quite low. "
            f"{batting_team} really need to start scoring faster."
        )

    # ── Wickets comment ──
    if wickets == 0:
        lines.append("All ten wickets are still standing — a great position for the batting side.")
    elif wickets <= 2:
        lines.append(
            f"Only {wickets} wicket{'s' if wickets > 1 else ''} down — {batting_team} still have "
            f"plenty of batting firepower left."
        )
    elif wickets <= 5:
        lines.append(
            f"With {wickets} wickets gone, {batting_team} need to be a bit more careful. "
            f"They cannot afford to lose the plot from here."
        )
    elif wickets <= 7:
        lines.append(
            f"{wickets} wickets fallen — {batting_team} are running out of batters. "
            f"Every ball counts now."
        )
    else:
        lines.append(
            f"⚠️ {wickets} wickets down — this is an emergency situation for {batting_team}. "
            f"The tail needs to dig in."
        )

    # ── Remaining overs ──
    if remaining_overs >= 15:
        lines.append(
            f"There are still {remaining_overs:.1f} overs left — a huge amount of cricket to be played."
        )
    elif remaining_overs >= 10:
        lines.append(
            f"With {remaining_overs:.1f} overs to go, the match is entering a critical phase."
        )
    elif remaining_overs >= 5:
        lines.append(
            f"Only {remaining_overs:.1f} overs remain — the death overs are here. "
            f"This is where matches are won and lost."
        )
    else:
        lines.append(
            f"Just {remaining_overs:.1f} overs left! Every single delivery is massive now."
        )

    # ── Model agreement ──
    if lstm_prob is not None and bilstm_prob is not None:
        both_agree = (lstm_prob >= 0.5) == (bilstm_prob >= 0.5)
        if both_agree:
            lines.append(
                f"Both AI models (LSTM: {lstm_prob*100:.0f}%, BiLSTM: {bilstm_prob*100:.0f}%) "
                f"agree on this outcome — that makes the prediction more reliable."
            )
        else:
            lines.append(
                f"Interestingly, the two AI models disagree: LSTM says {lstm_prob*100:.0f}% "
                f"and BiLSTM says {bilstm_prob*100:.0f}%. This is a genuinely uncertain match."
            )

    return " ".join(lines)


def generate_shap_commentary(shap_values, feature_vals, win_prob,
                              batting_team="the batting team"):
    """
    Turn raw SHAP values into plain-English bullet points.

    Parameters
    ----------
    shap_values  : array-like length 8
    feature_vals : array-like length 8 — original (unscaled) feature values
    win_prob     : float [0-1]
    batting_team : str

    Returns
    -------
    list of str — one sentence per important feature
    """
    import numpy as np
    sv    = np.array(shap_values, dtype=float)
    fv    = np.array(feature_vals, dtype=float)
    order = np.argsort(np.abs(sv))[::-1]
    lines = []

    for i in order[:5]:          # top 5 factors only
        name  = FEATURE_NAMES[i]
        plain = FEATURE_PLAIN.get(name, name)
        val   = fv[i] if i < len(fv) else 0.0
        s     = float(sv[i])

        if abs(s) < 0.005:
            continue              # skip near-zero contributions

        direction = "helping" if s > 0 else "hurting"
        strength  = ("a lot" if abs(s) > 0.08
                     else "somewhat" if abs(s) > 0.03
                     else "slightly")

        lines.append(
            f"• {plain.capitalize()} (value: {val:.1f}) is {strength} {direction} "
            f"{batting_team}'s chances."
        )

    if not lines:
        lines.append("No single factor is clearly dominant right now — a balanced situation.")

    return lines


def generate_lime_commentary(lime_weights, batting_team="the batting team"):
    """
    Turn LIME weights dict into a plain-English summary.

    Parameters
    ----------
    lime_weights : dict {feature_condition_str: weight}
    batting_team : str

    Returns
    -------
    str
    """
    if not lime_weights:
        return "LIME could not find a clear dominant factor for this prediction."

    sorted_w = sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    pos = [(f, w) for f, w in sorted_w if w > 0]
    neg = [(f, w) for f, w in sorted_w if w < 0]

    parts = []
    if pos:
        top_pos = pos[0][0]
        parts.append(f"Right now, {top_pos} is the biggest thing working in {batting_team}'s favour.")
    if neg:
        top_neg = neg[0][0]
        parts.append(f"On the other hand, {top_neg} is working against them the most.")

    if len(pos) > len(neg):
        parts.append(f"Overall, more factors are helping ({len(pos)}) than hurting ({len(neg)}) — the AI leans WIN.")
    elif len(neg) > len(pos):
        parts.append(f"Overall, more factors are hurting ({len(neg)}) than helping ({len(pos)}) — the AI leans LOSS.")
    else:
        parts.append("An equal number of factors are helping and hurting — a genuinely close match.")

    return " ".join(parts)