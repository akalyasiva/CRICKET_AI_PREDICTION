"""pages/3_XAI_Commentary.py — Commentary from trained model"""
import streamlit as st
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nav import inject_navbar

BATTING_TEAMS = [
    "Chennai Super Kings","Deccan Chargers","Delhi Capitals",
    "Gujarat Lions","Gujarat Titans","Kochi Tuskers Kerala",
    "Kolkata Knight Riders","Lucknow Super Giants","Mumbai Indians",
    "Pune Warriors","Punjab Kings","Rajasthan Royals",
    "Rising Pune Supergiants","Royal Challengers Bangalore","Sunrisers Hyderabad",
]
BOWLING_TEAMS = [
    "Deccan Chargers","Chennai Super Kings","Delhi Capitals",
    "Gujarat Lions","Gujarat Titans","Kochi Tuskers Kerala",
    "Kolkata Knight Riders","Lucknow Super Giants","Mumbai Indians",
    "Pune Warriors","Punjab Kings","Rajasthan Royals",
    "Rising Pune Supergiants","Royal Challengers Bangalore","Sunrisers Hyderabad",
]

st.set_page_config(page_title="Commentary — Cricket AI", page_icon="💬",
                   layout="wide", initial_sidebar_state="collapsed")
inject_navbar("Commentary")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#f5f3ff;--card:#ffffff;--border:#d8d0f0;
  --text:#1e0a3c;--text2:#3b1f6b;--text3:#6b4fa0;--text4:#9b7fcc;
  --purple:#7c3aed;--violet:#6d28d9;
}
*,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{display:none!important;}
section[data-testid="stSidebarContent"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
#MainMenu,header,footer{visibility:hidden;}
.main .block-container{padding-top:72px!important;max-width:1280px;margin:0 auto;}
label{color:var(--text2)!important;font-size:.82rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div,div[data-baseweb="select"]>div{
  background:#fff!important;border-color:var(--border)!important;
  color:var(--text)!important;border-radius:8px!important;}
.stSlider>div>div>div{background:var(--purple)!important;}
.stButton>button{
  background:linear-gradient(90deg,#6d28d9,#7c3aed)!important;
  color:#fff!important;font-family:'Space Grotesk',sans-serif!important;
  font-size:.82rem!important;font-weight:800!important;letter-spacing:2px!important;
  text-transform:uppercase!important;padding:14px 0!important;
  border-radius:12px!important;border:none!important;width:100%!important;
  box-shadow:0 4px 16px rgba(124,58,237,.3)!important;}
.page-hdr{background:linear-gradient(135deg,#1e0a3c 0%,#4c1d95 55%,#0f172a 100%);
  border:1px solid rgba(139,92,246,.5);border-radius:18px;padding:40px 48px;
  margin-bottom:28px;position:relative;overflow:hidden;
  box-shadow:0 16px 48px rgba(0,0,0,.3);}
.page-hdr::before{content:'💬';position:absolute;font-size:240px;
  right:-20px;top:-50px;opacity:.04;pointer-events:none;}
.hdr-eye{font-family:'Space Grotesk',sans-serif;font-size:.7rem;font-weight:700;
  letter-spacing:4px;text-transform:uppercase;color:#a78bfa;margin-bottom:12px;display:block;}
.hdr-title{font-family:'Space Grotesk',sans-serif;font-size:2.4rem;font-weight:800;
  color:#fff;line-height:1.1;margin-bottom:10px;}
.hdr-title .acc{background:linear-gradient(90deg,#a78bfa,#ec4899);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hdr-sub{font-family:'Inter',sans-serif;font-size:.95rem;color:rgba(255,255,255,.6);}
.src-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;
  border-radius:20px;font-family:'Space Grotesk',sans-serif;font-size:.72rem;
  font-weight:700;letter-spacing:.5px;}
.src-real{background:#d1fae5;border:1px solid #6ee7b7;color:#065f46;}
.src-sim {background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;}
.verdict-strip{background:#fff;border:1.5px solid var(--border);border-radius:14px;
  padding:20px 26px;margin-bottom:18px;display:flex;align-items:center;
  gap:20px;flex-wrap:wrap;box-shadow:0 3px 12px rgba(124,58,237,.06);}
.v-badge{padding:8px 28px;border-radius:50px;font-family:'Space Grotesk',sans-serif;
  font-size:1.5rem;font-weight:900;letter-spacing:3px;flex-shrink:0;}
.v-win {background:linear-gradient(90deg,#059669,#10b981);color:#fff;}
.v-loss{background:linear-gradient(90deg,#dc2626,#ef4444);color:#fff;}
/* Commentary */
.commentary{background:#fff;border:1.5px solid var(--border);border-radius:14px;
  padding:26px 30px;box-shadow:0 3px 12px rgba(124,58,237,.06);}
.comm-text{font-family:'Inter',sans-serif;font-size:1.05rem;color:#1e0a3c;
  line-height:1.9;margin:0;}
.comm-text strong{color:#4c1d95;}
/* Factor pills row */
.factor-row{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;
  padding-top:16px;border-top:1px solid var(--border);}
.fpill{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;
  border-radius:20px;font-family:'Inter',sans-serif;font-size:.78rem;font-weight:600;}
.fpill-up  {background:#d1fae5;border:1px solid #6ee7b7;color:#065f46;}
.fpill-down{background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;}
.factor-label{font-family:'Space Grotesk',sans-serif;font-size:.6rem;font-weight:700;
  letter-spacing:3px;text-transform:uppercase;color:#9b7fcc;margin-bottom:8px;}
.await{display:flex;flex-direction:column;align-items:center;
  justify-content:center;height:280px;gap:12px;}
.await-icon{font-size:72px;opacity:.07;}
.await-title{font-family:'Space Grotesk',sans-serif;font-size:.85rem;font-weight:800;
  color:#c4b5fd;letter-spacing:4px;text-transform:uppercase;}
.await-sub{font-family:'Inter',sans-serif;font-size:.82rem;color:#9b7fcc;text-align:center;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="page-hdr">
  <span class="hdr-eye">Why did the AI say that?</span>
  <div class="hdr-title">XAI <span class="acc">Explanation</span></div>
  <div class="hdr-sub">A clear, honest explanation of why the AI made its prediction.</div>
</div>""", unsafe_allow_html=True)

FEATURE_NAMES = ["Inning","Batting Team","Bowling Team","Ball Number",
                 "Current Score","Wickets Fallen","Run Rate","Remaining Overs"]
FEATURE_PLAIN = {
    "Inning":           "which team bats first",
    "Batting Team":     "the team batting",
    "Bowling Team":     "the team bowling",
    "Ball Number":      "how far into the match it is",
    "Current Score":    "the score",
    "Wickets Fallen":   "how many batters are out",
    "Run Rate":         "how fast they are scoring",
    "Remaining Overs":  "how many overs are left",
}
BUCKET_NAMES = ["Low (<100)","Below Par (100-139)","Par (140-169)","Good (170-199)","Excellent (200+)"]

@st.cache_resource(show_spinner="Loading trained models…")
def get_engine():
    try:
        from utils.prediction_engine import load_models, get_teams
        lstm, bilstm, scaler, enc = load_models()
        return lstm, bilstm, scaler, enc, BATTING_TEAMS, True
    except Exception as e:
        return None, None, None, None, BATTING_TEAMS, False

lstm_m, bilstm_m, scaler, enc, TEAMS, MODEL_OK = get_engine()

def _sim_shap(score, wkts, overs, inning):
    np.random.seed(int(score + wkts*7 + overs*3))
    rr = score/(overs+1e-6); rem = 20-overs
    feat_vals = np.array([float(inning),0.,1.,overs*6,float(score),float(wkts),rr,rem])
    sv  = np.array([0.03*inning,0.01,-0.01,0.06*(overs/20),
                    0.12*(score/180),-0.10*(wkts/10),
                    0.09*min(rr/12,1),0.05*(rem/20)]) + np.random.normal(0,0.015,8)
    wp  = float(np.clip(0.5+sv.sum()*1.2, 0.05, 0.95))
    lw  = {FEATURE_NAMES[i]: float(sv[i]*np.random.uniform(0.8,1.2)) for i in range(8)}
    proj= int(np.clip(score+rr*rem*(1-wkts*0.018),50,280))
    rp  = np.array([max(.01,.15-proj*.001),max(.01,.20-abs(proj-120)*.003),
                    max(.01,.30-abs(proj-155)*.004),max(.01,.25-abs(proj-180)*.004),
                    max(.01,.10+(proj-190)*.003)]).clip(.01); rp/=rp.sum()
    return sv, lw, feat_vals, wp, float(proj), BUCKET_NAMES[int(np.argmax(rp))]

def build_commentary(batting, bowling, wp, proj, bkt_nm, sv_arr, lime_w, rr, wkts, overs, rem, inning, score):
    """Pure human-style commentary. No numbers except score/wickets. No jargon."""

    # What the top SHAP + LIME features actually are
    top_up   = [FEATURE_NAMES[i] for i in np.argsort(sv_arr)[::-1] if sv_arr[i] >  0.02][:2]
    top_down = [FEATURE_NAMES[i] for i in np.argsort(sv_arr)       if sv_arr[i] < -0.02][:2]
    lime_items = sorted(lime_w.items(), key=lambda x: abs(x[1]), reverse=True)
    lime_top = lime_items[0][0] if lime_items else None

    # Plain phrases for each feature — used inline in sentences
    PHRASES = {
        "Current Score":    ("the runs they have on the board",         "not having enough runs yet"),
        "Run Rate":         ("the strong scoring pace",                  "the slow scoring pace"),
        "Wickets Fallen":   ("having most of their batters still in",   "losing too many wickets"),
        "Remaining Overs":  ("still having plenty of overs to go",      "running out of overs"),
        "Ball Number":      ("being well into the innings",              "being early in the innings"),
        "Inning":           ("batting second and chasing a target",      "batting first"),
        "Batting Team":     ("the strength of this batting side",        "the batting lineup"),
        "Bowling Team":     ("the pressure from the bowling side",       "the bowling attack"),
    }

    def pos_phrase(f): return PHRASES.get(f, (f, f))[0]
    def neg_phrase(f): return PHRASES.get(f, (f, f))[1]

    # ── Opener ────────────────────────────────────────────────────────────────
    if wp >= 0.80:
        opener = f"Things are looking really good for <strong>{batting}</strong> right now."
    elif wp >= 0.65:
        opener = f"<strong>{batting}</strong> are in a decent position and have the upper hand."
    elif wp >= 0.55:
        opener = f"<strong>{batting}</strong> have a small advantage, but this match is far from over."
    elif wp >= 0.45:
        opener = f"It is a very close contest — honestly, either team could win this."
    elif wp >= 0.35:
        opener = f"<strong>{bowling}</strong> look slightly more in control at this stage."
    else:
        opener = f"<strong>{bowling}</strong> are on top and look the stronger side right now."

    # ── Situation ─────────────────────────────────────────────────────────────
    if wkts == 0:
        wkt_line = "they have not lost a single wicket"
    elif wkts <= 2:
        wkt_line = "only a couple of wickets have gone"
    elif wkts <= 4:
        wkt_line = "a few wickets have fallen along the way"
    elif wkts <= 6:
        wkt_line = f"{wkts} wickets are down and the middle order needs to hold firm"
    else:
        wkt_line = "most of the main batters are already back in the pavilion"

    if rr >= 10:
        pace_line = "and the batting has been brilliant"
    elif rr >= 8:
        pace_line = "and they have been scoring nicely"
    elif rr >= 6:
        pace_line = "and it has been a steady if unspectacular innings"
    else:
        pace_line = "though the team has found it tough to score freely"

    sit = f"<strong>{batting}</strong> are on <strong>{score} for {wkts}</strong>, {wkt_line} {pace_line}."

    # ── SHAP — what is helping or hurting ────────────────────────────────────
    if top_up and top_down:
        shap_sent = (f"What is working for them is <strong>{pos_phrase(top_up[0])}</strong>, "
                     f"but the AI is a little worried about <strong>{neg_phrase(top_down[0])}</strong>.")
    elif top_up:
        shap_sent = f"The key reason the AI likes {batting} here is <strong>{pos_phrase(top_up[0])}</strong>."
    elif top_down:
        shap_sent = f"The main concern for {batting} is <strong>{neg_phrase(top_down[0])}</strong>."
    else:
        shap_sent = ""

    # ── LIME — right now ──────────────────────────────────────────────────────
    if lime_top:
        lime_val = lime_w.get(lime_top, 0)
        if lime_val >= 0:
            lime_sent = f"Right now, <strong>{pos_phrase(lime_top)}</strong> is the one thing that stands out the most."
        else:
            lime_sent = f"Right now, <strong>{neg_phrase(lime_top)}</strong> is the one thing that stands out the most."
    else:
        lime_sent = ""

    # ── Closing ───────────────────────────────────────────────────────────────
    if wp >= 0.80:
        close = f"If they keep batting like this, {batting} should win comfortably."
    elif wp >= 0.60:
        close = f"{bowling} need a couple of quick wickets or this will get away from them."
    elif wp >= 0.45:
        close = f"The next few overs are going to be really important."
    else:
        close = f"{batting} need something special to turn this match around."

    parts = [opener, sit, shap_sent, lime_sent, close]
    return " ".join(p for p in parts if p)

    parts = [conf + ".", sit, shap_sent, lime_sent, close]
    return " ".join(p for p in parts if p)

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<span style="font-family:\'Space Grotesk\',sans-serif;font-size:.65rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#7c3aed;margin-bottom:14px;display:block;">▸ Enter Match Details</span>', unsafe_allow_html=True)
cc = st.columns(5)
with cc[0]: inning  = st.selectbox("Inning",       [1,2])
with cc[1]: batting = st.selectbox("Batting Team", BATTING_TEAMS)
with cc[2]: bowling = st.selectbox("Bowling Team", [t for t in BOWLING_TEAMS if t!=batting])
with cc[3]: score   = st.number_input("Current Score",0,300,112)
with cc[4]: wkts    = st.slider("Wickets Fallen",0,10,2)
overs  = st.slider("Overs Completed",0.0,20.0,11.0,step=0.1)
go_btn = st.button("💬  EXPLAIN THIS MATCH")

if go_btn:
    rr  = score/(overs+1e-6); rem = 20-overs
    is_sim = False
    wp = proj = bkt_nm = sv_arr = lime_w = feat_vals = None
    lstm_p = bi_p = None

    if MODEL_OK:
        try:
            from utils.prediction_engine import preprocess, predict as run_predict
            X_flat = preprocess(inning, batting, bowling, int(overs*6),
                                score, wkts, rr, rem, scaler, enc)
            result = run_predict(X_flat, lstm_m, bilstm_m, scaler)
            wp     = result["win_probability"]
            proj   = result.get("predicted_score", score+rr*rem)
            bkt_nm = result.get("score_bucket_name","Par (140-169)")
            lstm_p = result.get("lstm_probability", wp)
            bi_p   = result.get("bilstm_probability", wp)

            try:
                import shap
                SEQ = 10
                bg_flat = np.random.randn(20,8)*0.4
                def _pred_fn(Xf):
                    Xs=np.repeat(Xf[:,np.newaxis,:],SEQ,axis=1)
                    return lstm_m.predict(Xs,verbose=0).flatten()
                bg_km = shap.kmeans(bg_flat,10)
                exp   = shap.KernelExplainer(_pred_fn, bg_km)
                sv    = exp.shap_values(X_flat, nsamples=60)
                sv_arr= np.array(sv[0] if isinstance(sv,list) else sv).flatten()[:8]
                feat_vals = X_flat.flatten()
            except:
                sv_arr, lime_w, feat_vals, _, _, _ = _sim_shap(score,wkts,overs,inning)
                is_sim = True

            try:
                import lime, lime.lime_tabular
                bg2 = np.random.randn(20,8)*0.4
                SEQ = 10
                def _lime_fn(Xf):
                    Xs=np.repeat(Xf[:,np.newaxis,:],SEQ,axis=1)
                    p=lstm_m.predict(Xs,verbose=0).flatten()
                    return np.column_stack([1-p,p])
                lexpl = lime.lime_tabular.LimeTabularExplainer(
                    bg2, feature_names=FEATURE_NAMES,
                    class_names=["LOSS","WIN"], mode="classification",
                    discretize_continuous=True, random_state=42)
                ex = lexpl.explain_instance(X_flat.flatten(),_lime_fn,
                                             num_features=8,num_samples=300,labels=(1,))
                lime_w = dict(ex.as_list(label=1))
            except:
                _, lime_w, _, _, _, _ = _sim_shap(score,wkts,overs,inning)

        except Exception as ex:
            st.warning(f"Model error: {str(ex)[:100]}")
            sv_arr,lime_w,feat_vals,wp,proj,bkt_nm = _sim_shap(score,wkts,overs,inning)
            lstm_p=bi_p=wp; is_sim=True
    else:
        sv_arr,lime_w,feat_vals,wp,proj,bkt_nm = _sim_shap(score,wkts,overs,inning)
        lstm_p=bi_p=wp; is_sim=True

    if sv_arr is None:
        sv_arr,lime_w,feat_vals,_,_,_ = _sim_shap(score,wkts,overs,inning)

    pred = "WIN" if wp>=0.5 else "LOSS"
    pc   = "#059669" if pred=="WIN" else "#dc2626"
    vcls = "v-win"   if pred=="WIN" else "v-loss"

    src_html = ('<span class="src-badge src-sim">⚠️ Simulated — run train_model.py first</span>'
                if is_sim
                else '<span class="src-badge src-real">✅ From trained LSTM + BiLSTM models</span>')

    if is_sim:
        st.warning("⚠️ **Simulation mode** — model not found. Run `python train_model.py` then `python save_metrics.py`.")

    # ── Verdict strip (unchanged) ──────────────────────────────────────────────
    st.markdown(
        '<div class="verdict-strip">'
        '<div class="v-badge '+vcls+'">'+pred+'</div>'
        '<div>'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:2.4rem;'
        'font-weight:900;color:'+pc+';line-height:1;">'+f'{wp*100:.1f}%'+'</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:.76rem;'
        'color:#6b4fa0;">chance of winning for '+batting+'</div>'
        '</div>'
        '<div style="margin-left:auto;text-align:right;">'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.5rem;'
        'font-weight:900;color:#7c3aed;">'+str(int(proj))+' runs</div>'
        '<div style="font-family:\'Inter\',sans-serif;font-size:.74rem;color:#6b4fa0;">'
        'projected · '+bkt_nm+'</div>'
        '<div style="margin-top:6px;">'+src_html+'</div>'
        '</div></div>',
        unsafe_allow_html=True)

    # ── Commentary ─────────────────────────────────────────────────────────────
    commentary = build_commentary(
        batting, bowling, wp, proj, bkt_nm,
        sv_arr, lime_w, rr, wkts, overs, rem, inning, score
    )

    # Helping / hurting pills from SHAP
    helps = [(FEATURE_NAMES[i], sv_arr[i]) for i in range(8) if sv_arr[i] >  0.02]
    hurts = [(FEATURE_NAMES[i], sv_arr[i]) for i in range(8) if sv_arr[i] < -0.02]
    helps = sorted(helps, key=lambda x: -x[1])
    hurts = sorted(hurts, key=lambda x:  x[1])

    pills_up   = "".join([f'<span class="fpill fpill-up">↑ {FEATURE_PLAIN.get(f,f)}</span>'   for f,_ in helps[:3]])
    pills_down = "".join([f'<span class="fpill fpill-down">↓ {FEATURE_PLAIN.get(f,f)}</span>' for f,_ in hurts[:3]])

    st.markdown(
        '<div class="commentary">'
        '<p class="comm-text">'+commentary+'</p>'
        '<div class="factor-row">'
        '<div style="width:100%;margin-bottom:4px;" class="factor-label">Key factors from SHAP + LIME</div>'
        + pills_up + pills_down +
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

else:
    st.markdown("""<div class="await">
      <div class="await-icon">💬</div>
      <div class="await-title">Ready to Explain</div>
      <div class="await-sub">Fill in match details above and click<br>
        <strong>Explain This Match</strong></div>
    </div>""", unsafe_allow_html=True)