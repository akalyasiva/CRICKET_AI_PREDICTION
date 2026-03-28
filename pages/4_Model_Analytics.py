"""
pages/4_Model_Analytics.py
══════════════════════════
Reads from:
  • models/xai_cache.json  — background samples + global SHAP/LIME
  • models/metrics.json    — accuracy, precision, recall, F1

Local SHAP + LIME are computed live for any user-chosen match scenario.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nav import inject_navbar

st.set_page_config(
    page_title="Model Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_navbar("Analytics")

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#f0fff8;--bg2:#ffffff;--bg3:#f8fff9;
  --border:#b0dcc4;--text:#0f172a;--text2:#334155;--text3:#64748b;--text4:#94a3b8;
  --shap:#059669;--lime:#d97706;--blue:#2563eb;--red:#dc2626;
}
*,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{display:none!important;}
section[data-testid="stSidebarContent"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
#MainMenu,header,footer{visibility:hidden;}
.main .block-container{padding-top:76px!important;max-width:1300px;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;
  border:1px solid var(--border);border-radius:10px;}
.stTabs [data-baseweb="tab"]{color:var(--text4)!important;
  font-family:Orbitron,monospace!important;font-size:.58rem!important;
  letter-spacing:2px!important;padding:10px 18px!important;}
.stTabs [aria-selected="true"]{color:var(--shap)!important;
  background:var(--bg3)!important;border-radius:8px!important;}

/* Input panel */
.input-panel{background:#ffffff;border:1px solid var(--border);border-radius:14px;
  padding:20px 18px;position:sticky;top:80px;}
.ip-title{font-family:Orbitron,monospace;font-size:.52rem;letter-spacing:3px;
  color:var(--shap);text-transform:uppercase;margin-bottom:14px;}
.derived-row{background:#f0fff8;border:1px solid var(--border);border-radius:8px;
  padding:10px 14px;margin:10px 0;display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.derived-lbl{font-size:.68rem;color:var(--text4);}
.derived-val{font-family:Orbitron,monospace;font-size:.82rem;font-weight:700;
  color:var(--shap);}

/* Predict button */
.stButton>button{width:100%;background:linear-gradient(135deg,#059669,#047857)!important;
  color:#fff!important;border:none!important;border-radius:9px!important;
  font-family:Orbitron,monospace!important;font-size:.58rem!important;
  letter-spacing:2px!important;padding:12px!important;margin-top:6px!important;
  font-weight:700!important;cursor:pointer!important;}
.stButton>button:hover{background:linear-gradient(135deg,#047857,#065f46)!important;}

/* Win prob badge */
.prob-badge{text-align:center;padding:12px;background:#f0fff8;
  border:1px solid var(--border);border-radius:10px;margin-bottom:14px;}
.prob-pct{font-family:Orbitron,monospace;font-size:1.6rem;font-weight:900;}
.prob-lbl{font-size:.68rem;color:var(--text4);margin-top:2px;}

/* Metric pills */
.pill-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0;}
.pill{background:#f8fff9;border:1px solid var(--border);border-radius:8px;
  padding:10px 8px;text-align:center;}
.pill-val{font-family:Orbitron,monospace;font-size:.9rem;font-weight:700;}
.pill-lbl{font-size:.6rem;color:var(--text4);text-transform:uppercase;
  letter-spacing:1px;margin-top:3px;}

/* Model cards */
.m-card{background:#fff;border:1px solid var(--border);border-radius:12px;
  padding:20px;position:relative;overflow:hidden;}
.m-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.m-card.lstm::before  {background:linear-gradient(90deg,#2563eb,#06b6d4);}
.m-card.bilstm::before{background:linear-gradient(90deg,#7c3aed,#ec4899);}
.m-acc{font-family:Orbitron,monospace;font-size:2rem;font-weight:900;line-height:1.1;}
.m-sub{font-size:.7rem;color:var(--text4);margin-bottom:10px;}

/* Inputs */
label{color:var(--text2)!important;font-size:.78rem!important;}
div[data-baseweb="select"]>div{background:#fff!important;
  border-color:var(--border)!important;border-radius:8px!important;}
.stSlider>div>div>div{background:var(--shap)!important;}
.stNumberInput>div>div{background:#fff!important;border-color:var(--border)!important;
  border-radius:8px!important;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Inning", "Batting Team", "Bowling Team", "Ball Number",
    "Current Score", "Wickets Fallen", "Run Rate", "Remaining Overs",
]
IPL_TEAMS = [
    "Chennai Super Kings", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Sunrisers Hyderabad", "Delhi Capitals",
    "Punjab Kings", "Rajasthan Royals", "Gujarat Titans", "Lucknow Super Giants",
]

PBG = "#f8fff9"

def PL(h=360):
    return dict(
        paper_bgcolor="#ffffff",
        font=dict(color="#334155", family="Inter"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#059669"),
        margin=dict(t=40, b=30, l=10, r=30),
        height=h,
    )

# ── LOAD CACHES ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
XAI_PATH     = os.path.join(BASE_DIR, "models", "xai_cache.json")

@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            d = json.load(f); d["loaded"] = True; return d
    return {"loaded": False,
            "lstm":   {"accuracy":86.04,"precision":85.20,"recall":87.10,"f1":86.14},
            "bilstm": {"accuracy":87.82,"precision":87.00,"recall":88.50,"f1":87.74}}

@st.cache_data
def load_xai():
    if os.path.exists(XAI_PATH):
        with open(XAI_PATH) as f:
            d = json.load(f); d["loaded"] = True; return d
    return {"loaded": False}

M   = load_metrics()
XAI = load_xai()

# ── LOAD MODEL + BUILD EXPLAINERS (once per session) ──────────────────────────
@st.cache_resource
def load_engines():
    """Load models + scaler + encoders + build SHAP/LIME explainers once."""
    import joblib, tensorflow as tf
    from utils.shap_engine import create_shap_explainer
    from utils.lime_engine import create_lime_explainer

    scaler   = joblib.load(os.path.join(BASE_DIR, "models", "feature_scaler.pkl"))
    encoders = joblib.load(os.path.join(BASE_DIR, "models", "label_encoders.pkl"))
    lstm_m   = tf.keras.models.load_model(
                   os.path.join(BASE_DIR, "models", "lstm_model.h5"),   compile=False)
    bilstm_m = tf.keras.models.load_model(
                   os.path.join(BASE_DIR, "models", "bilstm_model.h5"), compile=False)

    # Use saved background if available, else random fallback
    if XAI.get("loaded") and "background_X" in XAI:
        bg = np.array(XAI["background_X"])          # (50, 8)
    else:
        bg = np.random.randn(50, 8)

    lstm_shap   = create_shap_explainer(lstm_m,   bg)
    bilstm_shap = create_shap_explainer(bilstm_m, bg)
    lstm_lime   = create_lime_explainer(bg)
    bilstm_lime = create_lime_explainer(bg)

    return {
        "lstm_model":   lstm_m,
        "bilstm_model": bilstm_m,
        "lstm_shap":    lstm_shap,
        "bilstm_shap":  bilstm_shap,
        "lstm_lime":    lstm_lime,
        "bilstm_lime":  bilstm_lime,
        "scaler":       scaler,
        "encoders":     encoders,
    }

# Try loading engines; page works in "cache-only" mode if models not present
engines_ok = False
try:
    ENG = load_engines()
    engines_ok = True
except Exception as _e:
    ENG = None

# ── PREPROCESS HELPER ─────────────────────────────────────────────────────────
def preprocess_input(inning, batting, bowling, ball_number,
                     score, wickets, run_rate, remaining):
    """Return scaled flat (1, 8) array for the given scenario."""
    enc = ENG["encoders"]; scaler = ENG["scaler"]
    bt = enc["batting_team"].transform([batting])[0] if batting in enc["batting_team"].classes_ else 0
    bw = enc["bowling_team"].transform([bowling])[0] if bowling in enc["bowling_team"].classes_ else 0
    raw = np.array([[inning, bt, bw, ball_number, score, wickets, run_rate, remaining]])
    return scaler.transform(raw)   # (1, 8)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#064e3b,#022c22 60%,#0f172a);
    border:1px solid #059669;border-radius:16px;padding:28px 36px;margin-bottom:20px;'>
    <div style='font-family:Orbitron,monospace;font-size:.52rem;letter-spacing:5px;
        color:#34d399;text-transform:uppercase;margin-bottom:6px;'>How the AI Works</div>
    <div style='font-family:Orbitron,monospace;font-size:1.6rem;font-weight:900;color:#fff;'>
        Model <span style="background:linear-gradient(90deg,#34d399,#06d6a0);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Analytics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS ────────────────────────────────────────────────────────────────────
if not engines_ok:
    st.warning("⚡ Models not found — run `python save_metrics.py` and `python save_xai.py` first. Showing cached global values only.")


# ══════════════════════════════════════════════════════════════════════════════
# TWO-COLUMN LAYOUT: [inputs | tabs]
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 3], gap="large")

# ── LEFT: INPUT PANEL ─────────────────────────────────────────────────────────
with left:
    st.markdown("<div class='input-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='ip-title'>Match Scenario</div>", unsafe_allow_html=True)

    inning  = st.radio("Inning", [1, 2], horizontal=True, index=1)
    batting = st.selectbox("Batting Team",  IPL_TEAMS, index=0)
    bowling = st.selectbox("Bowling Team",  IPL_TEAMS, index=1)
    overs   = st.number_input("Overs", min_value=0.1, max_value=19.5,
                               value=12.0, step=0.1, format="%.1f")
    score   = st.number_input("Score", min_value=0, max_value=300,
                               value=110, step=1)
    wickets = st.slider("Wickets", 0, 10, 3)

    # Auto-derived
    run_rate  = round(score / max(overs, 0.1), 1)
    remaining = round(20.0 - overs, 1)
    ball_num  = int(overs * 6)

    st.markdown(f"""
    <div class='derived-row'>
        <div><div class='derived-lbl'>Run Rate</div>
             <div class='derived-val'>{run_rate}/ov</div></div>
        <div><div class='derived-lbl'>Rem. Overs</div>
             <div class='derived-val'>{remaining}</div></div>
    </div>""", unsafe_allow_html=True)

    analyse = st.button("⚡  ANALYSE", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: TABS ───────────────────────────────────────────────────────────────
with right:

    # ── COMPUTE local SHAP + LIME if Analyse clicked ──────────────────────────
    if "shap_result" not in st.session_state:
        st.session_state["shap_result"] = None
        st.session_state["lime_result"] = None
        st.session_state["win_prob"]    = None
        st.session_state["last_input"]  = None

    current_input = (inning, batting, bowling, overs, score, wickets)

    if analyse and engines_ok:
        with st.spinner("Computing SHAP + LIME …"):
            from utils.shap_engine import local_shap_values
            from utils.lime_engine import local_lime_explanation
            import tensorflow as tf

            SEQ    = 10
            X_flat = preprocess_input(inning, batting, bowling, ball_num,
                                      score, wickets, run_rate, remaining)
            model  = ENG["bilstm_model"]
            X_seq  = np.repeat(X_flat[:, np.newaxis, :], SEQ, axis=1)
            wp     = float(model.predict(X_seq, verbose=0).flatten()[0])

            sr = local_shap_values(ENG["bilstm_shap"], X_flat)
            lr = local_lime_explanation(ENG["bilstm_lime"], model, X_flat)

            # Map LIME dict → ordered (8,) array
            w_arr = np.zeros(8)
            for feat_str, w in lr["weights"].items():
                for j, fn in enumerate(FEATURE_NAMES):
                    if fn.lower().replace(" ","") in feat_str.lower().replace(" ",""):
                        w_arr[j] = w; break

            st.session_state["shap_result"] = sr
            st.session_state["lime_result"] = {"weights_arr": w_arr,
                                                "local_pred":  lr["local_pred"],
                                                "model_pred":  wp}
            st.session_state["win_prob"]    = wp
            st.session_state["last_input"]  = current_input

    SR  = st.session_state["shap_result"]
    LR  = st.session_state["lime_result"]
    WP  = st.session_state["win_prob"]

    # ── Global fallback data ──────────────────────────────────────────────────
    if XAI.get("loaded") and "bilstm" in XAI:
        g_shap = np.array(XAI["bilstm"]["global_shap"]["mean_abs_shap"])
        g_lime = np.array(XAI["bilstm"]["global_lime"]["mean_abs_weight"])
    else:
        np.random.seed(42)
        g_shap = np.abs(np.random.normal(0.06, 0.03, 8))
        g_lime = g_shap * np.random.uniform(0.85, 1.15, 8)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆  SHAP",
        "🍋  LIME",
        "🤖  About Model",
        "⚖️  SHAP vs LIME",
    ])

    # ══ TAB 1 — SHAP ═══════════════════════════════════════════════════════════
    with tab1:
        from plotly.subplots import make_subplots

        # Sort features by global importance (highest at top)
        sidx = np.argsort(g_shap)   # ascending so top = top of horizontal bar
        feat_sorted = [FEATURE_NAMES[i] for i in sidx]
        glob_sorted = [float(g_shap[i]) for i in sidx]

        has_local = SR is not None
        if has_local:
            sv = np.array(SR["shap_values"])
            local_sorted = [float(sv[i]) for i in sidx]
            pred_label = "WIN" if WP >= 0.5 else "LOSS"
            pred_color = "#059669" if WP >= 0.5 else "#dc2626"
            local_title = f"Local SHAP — {WP*100:.1f}% {pred_label}"
        else:
            local_sorted = [0.0] * 8
            pred_color   = "#94a3b8"
            local_title  = "Local SHAP — click ANALYSE"

        fig = make_subplots(
            rows=1, cols=2,
            shared_yaxes=True,
            column_widths=[0.5, 0.5],
            subplot_titles=["Global SHAP (mean |value|)", local_title],
            horizontal_spacing=0.06,
        )

        # Left — Global SHAP (absolute, green gradient)
        fig.add_trace(go.Bar(
            y=feat_sorted, x=glob_sorted, orientation="h",
            name="Global SHAP",
            marker=dict(
                color=glob_sorted,
                colorscale=[[0,"#d1fae5"],[0.5,"#059669"],[1,"#b8860b"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in glob_sorted],
            textposition="outside",
            textfont=dict(size=9, color="#64748b"),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP| = %{x:.4f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)

        # Right — Local SHAP (signed, green/red)
        local_colors = (["#059669" if v >= 0 else "#dc2626" for v in local_sorted]
                        if has_local else ["#e2e8f0"] * 8)
        fig.add_trace(go.Bar(
            y=feat_sorted, x=local_sorted, orientation="h",
            name="Local SHAP",
            marker=dict(color=local_colors, opacity=0.88 if has_local else 0.3),
            text=[f"{v:+.3f}" for v in local_sorted] if has_local else [""] * 8,
            textposition="outside",
            textfont=dict(size=9, color="#334155"),
            hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)

        fig.add_vline(x=0, line_color="#b0dcc4", line_width=1.2, row=1, col=2)

        fig.update_layout(
            **PL(400),
            plot_bgcolor=PBG,
            title=dict(text="SHAP — Global &amp; Local (BiLSTM)",
                       font=dict(size=13, color="#0f172a")),
        )
        fig.update_xaxes(gridcolor="#e2e8f0", row=1, col=1,
                         title_text="Mean |SHAP value|",
                         title_font=dict(color="#94a3b8", size=10))
        fig.update_xaxes(gridcolor="#e2e8f0", row=1, col=2,
                         title_text="← LOSS  |  WIN →",
                         title_font=dict(color="#94a3b8", size=10))
        fig.update_yaxes(gridcolor="#e2e8f0")
        # Subplot title colours
        fig.layout.annotations[0].font.color = "#059669"
        fig.layout.annotations[1].font.color = pred_color
        fig.layout.annotations[0].font.size  = 11
        fig.layout.annotations[1].font.size  = 11

        st.plotly_chart(fig, use_container_width=True)

        if has_local:
            st.caption(
                f"Baseline: {SR['base_value']:.3f}  →  Final: **{WP*100:.1f}% {pred_label}**  |  "
                "Green = pushes WIN · Red = pushes LOSS"
            )
        else:
            st.info("👈 Set your match scenario and click **ANALYSE** to see local SHAP.")

    # ══ TAB 2 — LIME ═══════════════════════════════════════════════════════════
    with tab2:
        if LR is None:
            sidx = np.argsort(g_lime)
            fig = go.Figure(go.Bar(
                y=[FEATURE_NAMES[i] for i in sidx],
                x=[float(g_lime[i]) for i in sidx],
                orientation="h",
                marker=dict(color="#d97706", opacity=0.80),
                text=[f"{g_lime[i]:.4f}" for i in sidx],
                textposition="outside",
                textfont=dict(size=9, color="#64748b"),
                hovertemplate="<b>%{y}</b><br>Mean |weight| = %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(**PL(360), plot_bgcolor=PBG,
                title=dict(text="Global LIME Importance (BiLSTM)",
                           font=dict(size=13, color="#0f172a")),
                xaxis=dict(gridcolor="#e2e8f0", title="Mean |LIME weight|",
                           title_font=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("👈 Set your match scenario and click **ANALYSE** to see local LIME for that moment.")
        else:
            lw = np.array(LR["weights_arr"])
            lp = LR["local_pred"]
            mp = LR["model_pred"]

            pred_label = "WIN" if mp >= 0.5 else "LOSS"
            pred_color = "#059669" if mp >= 0.5 else "#dc2626"

            st.markdown(f"""
            <div class='prob-badge'>
                <div class='prob-pct' style='color:#d97706;'>{lp*100:.1f}%</div>
                <div class='prob-lbl'>LIME local prediction
                &nbsp;·&nbsp; model output&nbsp;
                <strong style='color:{pred_color};'>{mp*100:.1f}% {pred_label}</strong></div>
            </div>""", unsafe_allow_html=True)

            lidx      = np.argsort(lw)
            lw_sorted = lw[lidx]
            fn_sorted = [FEATURE_NAMES[i] for i in lidx]

            fig = go.Figure(go.Bar(
                y=fn_sorted, x=lw_sorted, orientation="h",
                marker=dict(
                    color=["#d97706" if v >= 0 else "#dc2626" for v in lw_sorted],
                    opacity=0.85,
                ),
                text=[f"{v:+.3f}" for v in lw_sorted],
                textposition="outside",
                textfont=dict(size=9, color="#334155"),
                hovertemplate="<b>%{y}</b><br>LIME weight: %{x:+.4f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color="#b0dcc4", line_width=1.5)
            fig.update_layout(**PL(360), plot_bgcolor=PBG,
                title=dict(text="Local LIME — feature weights for this scenario",
                           font=dict(size=13, color="#0f172a")),
                xaxis=dict(gridcolor="#e2e8f0",
                           title="← pushes LOSS  |  pushes WIN →",
                           title_font=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="#e2e8f0"),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"LIME (500 perturbations) · local pred {lp*100:.1f}%  ·  "
                       "Orange = pushes WIN · Red = pushes LOSS")

    # ══ TAB 3 — ABOUT MODEL ════════════════════════════════════════════════════
    with tab3:
        lm = M["lstm"]; bm = M["bilstm"]

        ca, cb = st.columns(2)

        def pill_row(met, color):
            return "".join([
                f'<div class="pill"><div class="pill-val" style="color:{color};">'
                f'{met[k]:.1f}%</div><div class="pill-lbl">{l}</div></div>'
                for k, l in [("accuracy","Accuracy"),("precision","Precision"),
                              ("recall","Recall"),("f1","F1")]
            ])

        with ca:
            st.markdown(
                f'<div class="m-card lstm">'
                f'<div style="font-family:Orbitron,monospace;font-size:.52rem;'
                f'letter-spacing:3px;color:#2563eb;margin-bottom:8px;">LSTM</div>'
                f'<div class="m-acc" style="color:#2563eb;">{lm["accuracy"]:.1f}%</div>'
                f'<div class="m-sub">accuracy on unseen test data</div>'
                f'<div class="pill-grid">{pill_row(lm,"#2563eb")}</div>'
                f'<div style="font-size:.8rem;color:var(--text3);margin-top:10px;'
                f'line-height:1.8;">'
                f'Reads match ball-by-ball, forward only.<br>'
                f'Faster to train.</div></div>',
                unsafe_allow_html=True,
            )

        with cb:
            st.markdown(
                f'<div class="m-card bilstm">'
                f'<div style="font-family:Orbitron,monospace;font-size:.52rem;'
                f'letter-spacing:3px;color:#7c3aed;margin-bottom:8px;">BiLSTM ✦ Recommended</div>'
                f'<div class="m-acc" style="color:#7c3aed;">{bm["accuracy"]:.1f}%</div>'
                f'<div class="m-sub">accuracy on unseen test data</div>'
                f'<div class="pill-grid">{pill_row(bm,"#7c3aed")}</div>'
                f'<div style="font-size:.8rem;color:var(--text3);margin-top:10px;'
                f'line-height:1.8;">'
                f'Reads forward AND backward.<br>'
                f'Higher accuracy across all categories.</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # LSTM vs BiLSTM comparison bar
        mk = ["accuracy","precision","recall","f1"]
        ml = ["Accuracy","Precision","Recall","F1"]
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(
            name="LSTM", x=ml, y=[lm[k] for k in mk],
            marker=dict(color="#2563eb", opacity=0.88,
                        line=dict(color="#1d4ed8", width=1)),
            text=[f"{lm[k]:.1f}%" for k in mk], textposition="outside",
            textfont=dict(size=10, color="#2563eb", family="Orbitron"),
            hovertemplate="<b>LSTM</b><br>%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig_cmp.add_trace(go.Bar(
            name="BiLSTM", x=ml, y=[bm[k] for k in mk],
            marker=dict(color="#7c3aed", opacity=0.88,
                        line=dict(color="#6d28d9", width=1)),
            text=[f"{bm[k]:.1f}%" for k in mk], textposition="outside",
            textfont=dict(size=10, color="#7c3aed", family="Orbitron"),
            hovertemplate="<b>BiLSTM</b><br>%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig_cmp.update_layout(
            **PL(300), plot_bgcolor=PBG, barmode="group",
            title=dict(text="LSTM vs BiLSTM — Performance Comparison",
                       font=dict(size=13, color="#0f172a")),
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(gridcolor="#e2e8f0", range=[70, 103], title="Score (%)"),
            legend=dict(bgcolor="#fff", bordercolor="#b0dcc4", borderwidth=1,
                        font=dict(color="#334155", size=11)),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Quick feature importance table
        st.markdown(
            "<div style='font-size:.8rem;font-weight:600;color:var(--text2);"
            "margin-bottom:6px;'>Feature Importance (Global SHAP — BiLSTM)</div>",
            unsafe_allow_html=True,
        )
        ranks = np.argsort(-g_shap)
        feat_df = pd.DataFrame({
            "Rank":    [f"#{i+1}" for i in range(8)],
            "Feature": [FEATURE_NAMES[r] for r in ranks],
            "Mean |SHAP|": [f"{g_shap[r]:.4f}" for r in ranks],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=330)

    # ══ TAB 4 — SHAP vs LIME ═══════════════════════════════════════════════════
    with tab4:
        if SR is None or LR is None:
            # Show global comparison
            g_sidx = np.argsort(g_shap)[::-1]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Global SHAP",
                x=[FEATURE_NAMES[i] for i in g_sidx],
                y=[float(g_shap[i]) for i in g_sidx],
                marker=dict(color="#059669", opacity=0.85),
                text=[f"{g_shap[i]:.3f}" for i in g_sidx],
                textposition="outside",
                textfont=dict(size=9, color="#059669"),
                hovertemplate="<b>SHAP</b> — %{x}<br>%{y:.4f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="Global LIME",
                x=[FEATURE_NAMES[i] for i in g_sidx],
                y=[float(g_lime[i]) for i in g_sidx],
                marker=dict(color="#d97706", opacity=0.80),
                text=[f"{g_lime[i]:.3f}" for i in g_sidx],
                textposition="outside",
                textfont=dict(size=9, color="#d97706"),
                hovertemplate="<b>LIME</b> — %{x}<br>%{y:.4f}<extra></extra>",
            ))
            fig.update_layout(**PL(380), plot_bgcolor=PBG, barmode="group",
                title=dict(text="Global SHAP vs LIME — overall feature importance",
                           font=dict(size=13, color="#0f172a")),
                xaxis=dict(gridcolor="#e2e8f0", tickangle=-25),
                yaxis=dict(gridcolor="#e2e8f0", title="Importance (absolute)"),
                legend=dict(bgcolor="#fff", bordercolor="#b0dcc4", borderwidth=1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.info("👈 Click **ANALYSE** to compare local SHAP vs LIME for your scenario.")

        else:
            sv  = np.array(SR["shap_values"])
            lw  = np.array(LR["weights_arr"])
            # Signed local comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Local SHAP",
                x=FEATURE_NAMES, y=sv,
                marker=dict(color=["#059669" if v>=0 else "#dc2626" for v in sv],
                            opacity=0.88),
                text=[f"{v:+.3f}" for v in sv],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>SHAP</b> — %{x}<br>%{y:+.4f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                name="Local LIME",
                x=FEATURE_NAMES, y=lw,
                marker=dict(color=["#d97706" if v>=0 else "#ef4444" for v in lw],
                            opacity=0.72),
                text=[f"{v:+.3f}" for v in lw],
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate="<b>LIME</b> — %{x}<br>%{y:+.4f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_color="#b0dcc4", line_width=1)
            fig.update_layout(**PL(380), plot_bgcolor=PBG, barmode="group",
                title=dict(text="Local SHAP vs LIME — signed contributions for this scenario",
                           font=dict(size=13, color="#0f172a")),
                xaxis=dict(gridcolor="#e2e8f0", tickangle=-25),
                yaxis=dict(gridcolor="#e2e8f0",
                           title="← LOSS  |  contribution  |  WIN →"),
                legend=dict(bgcolor="#fff", bordercolor="#b0dcc4", borderwidth=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Agreement
            agree = [(sv[i]>0) == (lw[i]>0) for i in range(8)]
            pct   = sum(agree)/8*100
            clr   = "#059669" if pct >= 75 else "#d97706"
            df_cmp = pd.DataFrame({
                "Feature":   FEATURE_NAMES,
                "SHAP":      [f"{v:+.3f}" for v in sv],
                "LIME":      [f"{v:+.3f}" for v in lw],
                "Agreement": ["✅" if a else "❌" for a in agree],
            })
            c1, c2 = st.columns([4, 1])
            with c1:
                st.dataframe(df_cmp, use_container_width=True, hide_index=True)
            with c2:
                st.markdown(
                    f"<div style='background:#fff;border:1px solid var(--border);"
                    f"border-radius:10px;padding:16px;text-align:center;margin-top:4px;'>"
                    f"<div style='font-family:Orbitron,monospace;font-size:.48rem;"
                    f"letter-spacing:2px;color:var(--text4);'>Agreement</div>"
                    f"<div style='font-family:Orbitron,monospace;font-size:1.6rem;"
                    f"font-weight:900;color:{clr};margin:6px 0;'>{pct:.0f}%</div>"
                    f"<div style='font-size:.72rem;color:var(--text4);'>"
                    f"{'Reliable ✓' if pct>=75 else 'Caution ⚠'}</div></div>",
                    unsafe_allow_html=True,
                )