"""pages/2_Predict.py — warm peach/cream light theme"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nav import inject_navbar

# ── Exact team lists ───────────────────────────────────────────────────────────
BATTING_TEAMS = [
    "Chennai Super Kings","Deccan Chargers","Delhi Capitals",
    "Gujarat Lions","Gujarat Titans","Kochi Tuskers Kerala",
    "Kolkata Knight Riders","Lucknow Super Giants","Mumbai Indians",
    "Pune Warriors","Punjab Kings","Rajasthan Royals",
    "Rising Pune Supergiants","Royal Challengers Bangalore","Sunrisers Hyderabad",
]
# List 2: Chennai moved to index 1 (after Deccan Chargers), rest same order
BOWLING_TEAMS = [
    "Deccan Chargers","Chennai Super Kings","Delhi Capitals",
    "Gujarat Lions","Gujarat Titans","Kochi Tuskers Kerala",
    "Kolkata Knight Riders","Lucknow Super Giants","Mumbai Indians",
    "Pune Warriors","Punjab Kings","Rajasthan Royals",
    "Rising Pune Supergiants","Royal Challengers Bangalore","Sunrisers Hyderabad",
]

st.set_page_config(page_title="Predict — Cricket AI", page_icon="⚡",
                   layout="wide", initial_sidebar_state="collapsed")
inject_navbar("Predict")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#f2f7f4;--bg2:#ffffff;--border:#c8ddd1;
  --text:#0d1f16;--text2:#1e3d2a;--text3:#4a7a5e;--text4:#7aaa90;
  --acc:#2d7a4f;--blue:#2563eb;--purple:#7c3aed;
}
*,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{display:none!important;}
section[data-testid="stSidebarContent"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
#MainMenu,header,footer{visibility:hidden;}
.main .block-container{padding-top:72px!important;max-width:1280px;margin:0 auto;}
label{color:var(--text2)!important;font-size:.82rem!important;}
.stSelectbox>div>div,div[data-baseweb="select"]>div,.stNumberInput>div>div{
  background:#fff!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:8px!important;}
.stSlider>div>div>div{background:var(--acc)!important;}
.stButton>button{background:linear-gradient(90deg,#2d7a4f,#1a5c38)!important;color:#fff!important;
  font-family:Orbitron,monospace!important;font-size:.62rem!important;font-weight:700!important;
  letter-spacing:3px!important;text-transform:uppercase!important;padding:14px 0!important;
  border-radius:10px!important;border:none!important;width:100%!important;}
.page-hdr{background:linear-gradient(135deg,#0a1f14 0%,#1a4d30 55%,#051008 100%);
  border:1px solid #2d7a4f;border-radius:18px;padding:40px 48px;margin-bottom:28px;
  position:relative;overflow:hidden;}
.page-hdr::before{content:'⚡';position:absolute;font-size:240px;right:-20px;top:-50px;opacity:.04;pointer-events:none;}
.hdr-eye{font-family:Orbitron,monospace;font-size:.56rem;letter-spacing:5px;color:#6ee7a0;
  text-transform:uppercase;margin-bottom:10px;display:block;}
.hdr-title{font-family:Orbitron,monospace;font-size:2rem;font-weight:900;color:#fff;margin-bottom:8px;}
.hdr-title .acc{background:linear-gradient(90deg,#6ee7a0,#34d399);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hdr-sub{color:rgba(255,255,255,.65);font-size:.92rem;}
.inp-panel{background:#fff;border:1.5px solid var(--border);border-radius:14px;
  padding:26px 26px 20px;box-shadow:0 3px 12px rgba(45,122,79,.07);}
.panel-lbl{font-family:Orbitron,monospace;font-size:.52rem;letter-spacing:3px;
  text-transform:uppercase;color:var(--acc);margin-bottom:16px;display:block;}
.res-card{background:#fff;border:1.5px solid var(--border);border-radius:14px;
  padding:22px 24px;box-shadow:0 3px 12px rgba(0,0,0,.04);
  margin-bottom:12px;position:relative;overflow:hidden;}
.res-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.rc-win::before  {background:linear-gradient(90deg,#059669,#10b981);}
.rc-loss::before {background:linear-gradient(90deg,#dc2626,#ef4444);}
.rc-score::before{background:linear-gradient(90deg,#2563eb,#06b6d4);}
.rc-bkt::before  {background:linear-gradient(90deg,#2d7a4f,#34d399);}
.verdict-row{text-align:center;padding:8px 0 14px;}
.v-eye{font-family:Orbitron,monospace;font-size:.52rem;letter-spacing:4px;
  text-transform:uppercase;color:#94a3b8;margin-bottom:8px;}
.v-badge{display:inline-block;padding:8px 36px;border-radius:50px;
  font-family:Orbitron,monospace;font-size:1.8rem;font-weight:900;letter-spacing:3px;}
.v-win {background:linear-gradient(90deg,#059669,#10b981);color:#fff;}
.v-loss{background:linear-gradient(90deg,#dc2626,#ef4444);color:#fff;}
.v-pct {font-family:Orbitron,monospace;font-size:2.6rem;font-weight:900;line-height:1.1;margin-top:8px;}
.split-bar{background:#d4eadb;border-radius:50px;height:12px;overflow:hidden;margin:6px 0;}
.split-fill{height:100%;border-radius:50px;}
.score-num{font-family:Orbitron,monospace;font-size:2.4rem;font-weight:900;color:#2563eb;line-height:1;}
.score-unit{font-size:.72rem;font-weight:500;color:#94a3b8;margin-top:4px;}
.bkt-pill{display:inline-block;padding:6px 20px;border-radius:30px;
  font-family:Orbitron,monospace;font-size:.78rem;font-weight:700;margin-top:8px;}
.sec-lbl{font-family:Orbitron,monospace;font-size:.52rem;letter-spacing:3px;
  text-transform:uppercase;color:#78503a;margin:12px 0 8px;display:block;}
.met-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px;}
.met-box{background:#edf5f0;border:1.5px solid var(--border);border-radius:10px;padding:12px;text-align:center;}
.met-key{font-family:Orbitron,monospace;font-size:.5rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#a0714f;margin-bottom:5px;}
.met-val{font-family:Orbitron,monospace;font-size:.9rem;font-weight:700;color:var(--text);}
.conf-banner{border-radius:12px;padding:14px 20px;margin-bottom:12px;display:flex;align-items:center;gap:12px;}
.conf-real{background:#f0fdf4;border:1.5px solid #86efac;}
.conf-sim {background:#fef9c3;border:1.5px solid #fde047;}
.conf-icon{font-size:1.4rem;}
.conf-text{font-size:.86rem;line-height:1.7;}
.conf-text strong{font-weight:700;}
.sum-strip{background:#edf5f0;border:1.5px solid var(--border);border-radius:10px;
  padding:13px 16px;font-size:.84rem;color:var(--text2);line-height:1.9;margin-top:10px;}
.await{display:flex;flex-direction:column;align-items:center;justify-content:center;height:480px;gap:12px;}
.await-icon{font-size:70px;opacity:.07;}
.await-title{font-family:Orbitron,monospace;font-size:.75rem;font-weight:700;color:#c8ddd1;letter-spacing:4px;text-transform:uppercase;}
.await-sub{font-size:.82rem;color:#7aaa90;text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="page-hdr">
  <span class="hdr-eye">Real-Time Prediction Engine</span>
  <div class="hdr-title">Win · Score · <span class="acc">Category</span></div>
  <div class="hdr-sub">LSTM + BiLSTM ensemble — Win/Loss · Projected score · Score category</div>
</div>
""", unsafe_allow_html=True)

BUCKET_COLORS = ['#94a3b8','#f59e0b','#3b82f6','#10b981','#b8860b']
BUCKET_EMOJIS = ['📉','📊','🎯','✅','🔥']
BUCKET_NAMES  = ["Low (<100)","Below Par (100-139)","Par (140-169)",
                 "Good (170-199)","Excellent (200+)"]

# ── Load trained models ONCE (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading trained models…")
def get_engine():
    try:
        from utils.prediction_engine import load_models, get_teams
        lstm, bilstm, scaler, enc = load_models()
        return lstm, bilstm, scaler, enc, get_teams(enc), True
    except Exception:
        return None, None, None, None, [
            "Chennai Super Kings","Deccan Chargers","Delhi Capitals",
            "Gujarat Lions","Gujarat Titans","Kochi Tuskers Kerala",
            "Kolkata Knight Riders","Lucknow Super Giants","Mumbai Indians",
            "Pune Warriors","Punjab Kings","Rajasthan Royals",
            "Rising Pune Supergiants","Royal Challengers Bangalore","Sunrisers Hyderabad"
        ], False

lstm_m, bilstm_m, scaler, enc, TEAMS, MODEL_OK = get_engine()

def _bucket(s):
    if   s < 100: return 0, "Low (<100)"
    elif s < 140: return 1, "Below Par (100-139)"
    elif s < 170: return 2, "Par (140-169)"
    elif s < 200: return 3, "Good (170-199)"
    else:         return 4, "Excellent (200+)"

def _simulate(score, wkts, overs, inning, rr, rem):
    np.random.seed(int(score + wkts*10 + overs*3))
    wp   = float(np.clip(0.5+(rr-8)*0.04-wkts*0.035+(inning-1.5)*-0.05
                         +np.random.normal(0,.03), .05, .95))
    proj = float(np.clip(score + rr*rem*(1-wkts*0.018), 50, 280))
    bi, bn = _bucket(proj)
    lp = float(np.clip(wp+np.random.normal(0,.025),.05,.95))
    bp = float(np.clip(wp+np.random.normal(0,.025),.05,.95))
    return {"prediction":"WIN" if wp>=.5 else "LOSS",
            "win_probability":wp,"loss_probability":1-wp,
            "lstm_probability":lp,"bilstm_probability":bp,
            "predicted_score":proj,"score_bucket":bi,"score_bucket_name":bn}

# ── Layout ─────────────────────────────────────────────────────────────────────
col_in, col_out = st.columns([1, 1.2], gap="large")

with col_in:
    st.markdown('<div class="inp-panel"><span class="panel-lbl">▸ Match State Inputs</span>',
                unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1: inning = st.selectbox("🏏 Inning", [1, 2])
    with r2: overs  = st.number_input("🕐 Overs Completed", 0.0, 20.0, 10.0,
                                       step=0.1, format="%.1f")
    batting = st.selectbox("🟡 Batting Team", BATTING_TEAMS)
    bowling = st.selectbox("🔵 Bowling Team", [t for t in BOWLING_TEAMS if t != batting])
    r3, r4 = st.columns(2)
    with r3: score = st.number_input("📊 Current Score", 0, 300, 95)
    with r4: wkts  = st.slider("💀 Wickets Fallen", 0, 10, 2)
    st.markdown('</div>', unsafe_allow_html=True)
    predict_btn = st.button("⚡  PREDICT NOW")

with col_out:
    if predict_btn:
        rr  = score / (overs + 1e-6)
        rem = 20.0 - overs
        is_sim = False

        if MODEL_OK:
            try:
                from utils.prediction_engine import preprocess, predict as run_predict
                X_flat = preprocess(inning, batting, bowling,
                                    int(overs*6), score, wkts, rr, rem, scaler, enc)
                result = run_predict(X_flat, lstm_m, bilstm_m, scaler)
            except Exception:
                result = _simulate(score, wkts, overs, inning, rr, rem)
                is_sim = True
        else:
            result = _simulate(score, wkts, overs, inning, rr, rem)
            is_sim = True

        # ── Confirmation / Alert banner ────────────────────────────────────
        if not is_sim:
            st.markdown(
                '<div class="conf-banner conf-real">'
                '<span class="conf-icon">✅</span>'
                '<div class="conf-text">'
                '<strong>Prediction from trained model</strong> — '
                'LSTM + BiLSTM loaded from <code>models/</code> folder. '
                'This is a real prediction, not a simulation.'
                '</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="conf-banner conf-sim">'
                '<span class="conf-icon">⚠️</span>'
                '<div class="conf-text">'
                '<strong>Simulation mode</strong> — trained model not found. '
                'Run <code>python train_model.py</code> then <code>python save_metrics.py</code>. '
                'Results below are <strong>illustrative only</strong>.'
                '</div></div>',
                unsafe_allow_html=True)

        pred     = result["prediction"]
        wp       = result["win_probability"]
        lp       = result["loss_probability"]
        proj     = result.get("predicted_score", score + rr*rem)
        bkt_i    = result.get("score_bucket", 2)
        bkt_name = result.get("score_bucket_name", "Par (140-169)")
        bkt_col  = BUCKET_COLORS[bkt_i]
        bkt_emo  = BUCKET_EMOJIS[bkt_i]
        pct_col  = "#059669" if pred=="WIN" else "#dc2626"
        v_cls    = "v-win"   if pred=="WIN" else "v-loss"
        rc_cls   = "rc-win"  if pred=="WIN" else "rc-loss"

        # ── Verdict card ───────────────────────────────────────────────────
        st.markdown(
            '<div class="res-card ' + rc_cls + '">'
            '<div class="verdict-row">'
            '<div class="v-eye">AI Prediction</div>'
            '<div class="v-badge ' + v_cls + '">' + pred + '</div>'
            '<div class="v-pct" style="color:' + pct_col + ';">' + f'{wp*100:.1f}%' + '</div>'
            '<div style="font-size:.76rem;font-weight:500;color:#94a3b8;margin-top:4px;">'
            'win probability for ' + batting + '</div>'
            '</div>'
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-family:Orbitron,monospace;font-size:.56rem;font-weight:700;color:#059669;">WIN ' + f'{wp*100:.1f}%' + '</span>'
            '<span style="font-family:Orbitron,monospace;font-size:.56rem;font-weight:700;color:#dc2626;">LOSS ' + f'{lp*100:.1f}%' + '</span>'
            '</div>'
            '<div class="split-bar"><div class="split-fill" style="width:' + f'{wp*100:.1f}' + '%;'
            'background:linear-gradient(90deg,#059669,#10b981);"></div></div>'
            '</div>',
            unsafe_allow_html=True)

        # ── Score + Category ───────────────────────────────────────────────
        c_sc, c_bkt = st.columns(2)
        with c_sc:
            st.markdown(
                '<div class="res-card rc-score">'
                '<span class="sec-lbl">Projected Final Score</span>'
                '<div class="score-num">' + str(int(proj)) + '</div>'
                '<div class="score-unit">estimated runs this innings</div>'
                '</div>',
                unsafe_allow_html=True)
        with c_bkt:
            st.markdown(
                '<div class="res-card rc-bkt">'
                '<span class="sec-lbl">Score Category</span>'
                '<div class="bkt-pill" style="background:' + bkt_col + '22;color:' + bkt_col + ';'
                'border:1.5px solid ' + bkt_col + '66;">' + bkt_emo + ' ' + bkt_name + '</div>'
                '</div>',
                unsafe_allow_html=True)

        # ── Win Probability Gauge ──────────────────────────────────────────
        st.markdown('<span class="sec-lbl">▸ Win Probability Gauge</span>',
                    unsafe_allow_html=True)
        gc = "#059669" if pred == "WIN" else "#dc2626"
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=wp * 100,
            number={"suffix": "%", "font": {"color": gc, "size": 30, "family": "Orbitron"}},
            title={"text": "Win Probability",
                   "font": {"color": "#4a7a5e", "family": "Orbitron", "size": 11}},
            gauge={
                "axis":  {"range": [0, 100], "tickcolor": "#c8ddd1",
                           "tickfont": {"color": "#7aaa90", "size": 9}},
                "bar":   {"color": gc, "thickness": .22},
                "bgcolor": "#f2f7f4", "bordercolor": "#c8ddd1",
                "steps": [
                    {"range": [0,  30], "color": "#fee2e2"},
                    {"range": [30, 50], "color": "#fef9c3"},
                    {"range": [50, 70], "color": "#d1fae5"},
                    {"range": [70,100], "color": "#bbf7d0"},
                ],
                "threshold": {"line": {"color": "#2d7a4f", "width": 3},
                               "thickness": .75, "value": 50}
            }
        ))
        fig_g.update_layout(
            paper_bgcolor="#f2f7f4",
            margin=dict(t=28, b=10, l=20, r=20),
            height=200
        )
        st.plotly_chart(fig_g, use_container_width=True)

        # ── Quick stats ────────────────────────────────────────────────────
        conf = ("Very High" if wp>=0.85 or wp<=0.15
                else "High"     if wp>=0.70 or wp<=0.30
                else "Moderate" if wp>=0.60 or wp<=0.40
                else "Low")
        st.markdown(
            '<div class="met-grid">'
            '<div class="met-box"><div class="met-key">Run Rate</div>'
            '<div class="met-val">' + f'{rr:.1f}' + '</div></div>'
            '<div class="met-box"><div class="met-key">Remaining</div>'
            '<div class="met-val">' + f'{rem:.1f} ov' + '</div></div>'
            '<div class="met-box"><div class="met-key">Confidence</div>'
            '<div class="met-val">' + conf + '</div></div>'
            '<div class="met-box"><div class="met-key">Wickets</div>'
            '<div class="met-val">' + f'{wkts}/10' + '</div></div>'
            '</div>',
            unsafe_allow_html=True)

        # ── Plain English summary ──────────────────────────────────────────
        mood = (batting + " are in a very strong position." if wp >= 0.75
                else batting + " have the edge but the match is still open." if wp >= 0.55
                else "This match is very close — could go either way." if wp >= 0.45
                else bowling + " currently have the upper hand.")
        rr_txt = (f"Scoring at {rr:.1f} runs/over — very fast pace." if rr >= 10
                  else f"Run rate of {rr:.1f}/over — decent pace." if rr >= 7
                  else f"Only {rr:.1f} runs/over — needs to accelerate.")
        wk_txt = ("No wickets fallen — all batters still in." if wkts == 0
                  else f"Only {wkts} wicket(s) down — plenty of batting left." if wkts <= 2
                  else f"{wkts} wickets gone — batting side needs to be careful." if wkts <= 5
                  else f"⚠️ {wkts} wickets down — not many batters left.")
        st.markdown(
            '<div class="sum-strip"><strong>AI Summary:</strong> '
            + mood + ' ' + rr_txt + ' ' + wk_txt
            + ' Projected: <strong>' + str(int(proj)) + ' runs</strong> (' + bkt_name + ').</div>',
            unsafe_allow_html=True)

    else:
        # ── Awaiting state ─────────────────────────────────────────────────
        st.markdown("""
        <div class="await">
          <div class="await-icon">⚡</div>
          <div class="await-title">Ready to Predict</div>
          <div class="await-sub">Fill in match details on the left<br>
            and click <strong>Predict Now</strong>.</div>
        </div>""", unsafe_allow_html=True)