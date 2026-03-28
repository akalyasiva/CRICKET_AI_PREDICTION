"""pages/5_Historical_Analytics.py — Light theme (warm orange/cream)"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nav import inject_navbar

st.set_page_config(page_title="Historical Analysis", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")
inject_navbar("Historical")

# ── LIGHT THEME — warm cream/orange (mirrors green theme structure exactly) ───
THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#fff8f0;--bg2:#ffffff;--bg3:#fff9f5;--surface:#fff0e0;
  --border:#f0d0b0;--border2:#d09060;
  --text:#1a0800;--text2:#3d1a00;--text3:#7a4010;--text4:#b07040;
  --gold:#b8860b;--gold2:#d97706;--orange:#c2410c;
  --inp-bg:#ffffff;--plot-paper:#ffffff;--plot-bg:#fff9f5;
}
*,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{display:none!important;}
section[data-testid="stSidebarContent"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
#MainMenu,header,footer{visibility:hidden;}
.main .block-container{padding-top:76px!important;max-width:1280px;}
label{color:var(--text2)!important;font-size:.82rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div,div[data-baseweb="select"]>div{
  background:var(--inp-bg)!important;border-color:var(--border)!important;
  color:var(--text)!important;border-radius:8px!important;}
.stMultiSelect>div{background:var(--inp-bg)!important;border-color:var(--border)!important;}
.stSlider>div>div>div{background:var(--orange)!important;}
.stButton>button{
  background:linear-gradient(90deg,#92400e,var(--orange))!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-family:Orbitron,monospace!important;font-weight:900!important;
  letter-spacing:3px!important;padding:14px 0!important;width:100%!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg2)!important;
  border:1px solid var(--border);border-radius:10px;}
.stTabs [data-baseweb="tab"]{color:var(--text4)!important;
  font-family:Orbitron,monospace!important;font-size:.6rem!important;
  letter-spacing:2px!important;padding:11px 15px!important;}
.stTabs [aria-selected="true"]{color:var(--orange)!important;
  background:var(--bg3)!important;border-radius:8px!important;}
.stat-mini{background:var(--bg2);border:1px solid var(--border);
  border-radius:10px;padding:16px;text-align:center;
  box-shadow:0 2px 5px rgba(0,0,0,.04);}
</style>
"""
st.markdown(THEME, unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#3d1a00 0%,#7c3a0c 50%,#1a0800 100%);
    border:1px solid #c2410c;border-radius:18px;padding:40px 48px;margin-bottom:28px;
    position:relative;overflow:hidden;'>
    <div style='position:absolute;font-size:220px;right:-20px;top:-50px;opacity:.03;'>📈</div>
    <div style='font-family:Orbitron,monospace;font-size:.6rem;letter-spacing:5px;
        color:#fb923c;text-transform:uppercase;margin-bottom:10px;'>Dataset Intelligence</div>
    <div style='font-family:Orbitron,monospace;font-size:2rem;font-weight:900;color:#fff;'>
        Historical <span style="background:linear-gradient(90deg,#f97316,#fbbf24);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Analysis</span>
    </div>
    <div style='color:rgba(255,255,255,.55);font-size:.9rem;margin-top:8px;'>
        Deep exploration of the CRICKET.csv dataset — ball-by-ball IPL statistics
    </div>
</div>
""", unsafe_allow_html=True)

# ── Plot helper ────────────────────────────────────────────────────────────────
def PL(h=400):
    return dict(
        paper_bgcolor="#ffffff",
        font=dict(color="#3d1a00", family="Inter"),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#c2410c", font_color="#1a0800"),
        margin=dict(t=50, b=40, l=20, r=20), height=h
    )
PBG  = "#fff9f5"
GCOL = "#f0d0b0"
AXIS = dict(color="#b07040", size=10)   # reusable title_font for all axes

TEAM_COLORS = {
    "Chennai Super Kings":          "#c8a400",
    "Deccan Chargers":              "#4682b4",
    "Delhi Capitals":               "#0078BC",
    "Gujarat Lions":                "#ff6600",
    "Gujarat Titans":               "#4a7fc1",
    "Kochi Tuskers Kerala":         "#800080",
    "Kolkata Knight Riders":        "#7b5ea7",
    "Lucknow Super Giants":         "#44a8d8",
    "Mumbai Indians":               "#004BA0",
    "Pune Warriors":                "#008b8b",
    "Punjab Kings":                 "#ED1B24",
    "Rajasthan Royals":             "#d63384",
    "Rising Pune Supergiants":      "#8b0000",
    "Royal Challengers Bengaluru":  "#EC1C24",
    "Sunrisers Hyderabad":          "#e8650a",
}

TEAM_NAME_MAP = {
    "Royal Challengers Bangalore":  "Royal Challengers Bengaluru",
    "Rising Pune Supergiant":       "Rising Pune Supergiants",
    "Delhi Daredevils":             "Delhi Capitals",
    "Kings XI Punjab":              "Punjab Kings",
}

# ── Load CSV ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = [
        os.path.join(base, "datasets", "CRICKET.csv"),
        "datasets/CRICKET.csv",
        "../datasets/CRICKET.csv",
        "CRICKET.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                return pd.read_csv(p), True, p
            except Exception:
                pass
    return None, False, None

df_raw, DATA_LOADED, data_path = load_data()

if DATA_LOADED:
    st.success(f"✅ Dataset loaded: **{data_path}** — {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
else:
    st.warning("⚠️ CRICKET.csv not found. Showing simulated demo data. "
               "Place your file at: `datasets/CRICKET.csv`")

# ── Feature engineering ────────────────────────────────────────────────────────
@st.cache_data
def engineer(df):
    df = df.copy()
    for col in ['batting_team', 'bowling_team']:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_NAME_MAP)
    df['extras_type']    = df['extras_type'].fillna('No_Extra')
    df['dismissal_kind'] = df['dismissal_kind'].fillna('No_Dismissal')
    df['ball_number']    = df['over'] * 6 + df['ball']
    df = df.sort_values(['match_id','inning','ball_number'])
    df['current_score']  = df.groupby(['match_id','inning'])['total_runs'].cumsum()
    df['wickets_fallen'] = df.groupby(['match_id','inning'])['is_wicket'].cumsum()
    df['overs_completed']= df['ball_number'] / 6.0
    df['run_rate']       = df['current_score'] / (df['overs_completed'] + 1e-6)
    df['remaining_overs']= 20 - df['overs_completed']
    final = df.groupby(['match_id','inning'])['total_runs'].sum().reset_index()
    inn1  = final[final['inning']==1][['match_id','total_runs']].rename(columns={'total_runs':'score_1'})
    inn2  = final[final['inning']==2][['match_id','total_runs']].rename(columns={'total_runs':'score_2'})
    mdf   = inn1.merge(inn2, on='match_id', how='inner')
    mdf['team2_win'] = (mdf['score_2'] > mdf['score_1']).astype(int)
    df    = df.merge(mdf[['match_id','team2_win','score_1','score_2']], on='match_id', how='left')
    df['win'] = np.where(df['inning']==2, df['team2_win'], 1 - df['team2_win'])
    return df, mdf

if DATA_LOADED:
    df, match_df = engineer(df_raw)
else:
    np.random.seed(42)
    teams = ["Chennai Super Kings","Mumbai Indians","Royal Challengers Bangalore",
             "Kolkata Knight Riders","Sunrisers Hyderabad","Delhi Capitals",
             "Rajasthan Royals","Punjab Kings","Deccan Chargers"]
    rows = []
    for mid in range(1, 201):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        for inn in [1,2]:
            for ov in range(20):
                for bl in range(1,7):
                    runs = int(np.random.choice([0,1,2,3,4,6], p=[0.35,0.28,0.12,0.05,0.12,0.08]))
                    wkt  = int(np.random.random() < 0.05)
                    rows.append({"match_id":mid,"inning":inn,"over":ov,"ball":bl,
                                 "batting_team":t1 if inn==1 else t2,
                                 "bowling_team":t2 if inn==1 else t1,
                                 "batsman_runs":runs,"total_runs":runs,
                                 "is_wicket":wkt,
                                 "dismissal_kind":"caught" if wkt else "No_Dismissal",
                                 "extras_type":"No_Extra"})
    df_raw = pd.DataFrame(rows)
    df, match_df = engineer(df_raw)

# ── Overview stats ─────────────────────────────────────────────────────────────
stat_colors = ["#b8860b","#2563eb","#059669","#dc2626","#7c3aed"]
for col, (ic, lbl, val, clr) in zip(st.columns(5), [
    ("🏏","Matches",    f"{df['match_id'].nunique():,}",      stat_colors[0]),
    ("⚾","Total Balls", f"{len(df):,}",                       stat_colors[1]),
    ("🏢","Teams",       f"{df['batting_team'].nunique()}",    stat_colors[2]),
    ("💀","Wickets",     f"{int(df['is_wicket'].sum()):,}",   stat_colors[3]),
    ("🔢","Columns",     f"{df.shape[1]}",                    stat_colors[4]),
]):
    with col:
        st.markdown(f"""<div class="stat-mini">
            <div style='font-size:1.6rem;'>{ic}</div>
            <div style='font-family:Orbitron,monospace;font-size:1.2rem;
                font-weight:700;color:{clr};'>{val}</div>
            <div style='font-size:.68rem;color:var(--text4);
                letter-spacing:1px;margin-top:3px;'>{lbl.upper()}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 Team Performance","🏏 Batting Analysis",
    "🎯 Bowling Analysis","📈 Win Probability Flow",
    "🔍 Over-by-Over","📋 Raw Dataset",
])

# ─── Tab 1: Team Performance ──────────────────────────────────────────────────
with t1:
    team_stats = []
    for team in df['batting_team'].unique():
        tm = df[df['batting_team']==team]
        matches = tm['match_id'].nunique()
        if matches < 2: continue
        wins      = tm.groupby('match_id')['win'].first().sum()
        avg_score = tm.groupby(['match_id','inning'])['total_runs'].sum().mean()
        avg_rr    = float(tm['run_rate'].mean())
        team_stats.append({"Team":team,"Matches":matches,"Wins":int(wins),
                           "Win Rate":float(wins/matches),
                           "Avg Score":float(avg_score),"Avg RR":avg_rr})
    ts_df = pd.DataFrame(team_stats).sort_values("Win Rate", ascending=False)

    ca, cb = st.columns(2)
    with ca:
        fig = go.Figure(go.Bar(
            y=ts_df["Team"], x=ts_df["Win Rate"]*100, orientation='h',
            marker=dict(color=[TEAM_COLORS.get(t,"#b8860b") for t in ts_df["Team"]],
                        opacity=0.85, line=dict(color=GCOL, width=.5)),
            text=[f"{v:.0f}%" for v in ts_df["Win Rate"]*100], textposition="outside",
            textfont=dict(color="#7a4010", size=10, family="Orbitron"),
            hovertemplate="<b>%{y}</b><br>Win Rate: %{x:.1f}%<extra></extra>"
        ))
        fig.update_layout(**PL(420), plot_bgcolor=PBG,
            title=dict(text="Team Win Rates", font=dict(color="#1a0800",size=14,family="Orbitron")),
            xaxis=dict(gridcolor=GCOL, range=[0,100],
                       title="Win Rate (%) — percentage of matches won",
                       title_font=AXIS),
            yaxis=dict(gridcolor=GCOL,
                       title="IPL Team",
                       title_font=AXIS))
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        fig2 = go.Figure(go.Bar(
            y=ts_df["Team"], x=ts_df["Avg Score"], orientation='h',
            marker=dict(color=[TEAM_COLORS.get(t,"#b8860b") for t in ts_df["Team"]],
                        opacity=0.75, line=dict(color=GCOL, width=.5)),
            text=[f"{v:.0f}" for v in ts_df["Avg Score"]], textposition="outside",
            textfont=dict(color="#7a4010", size=10, family="Orbitron"),
            hovertemplate="<b>%{y}</b><br>Avg Score: %{x:.1f}<extra></extra>"
        ))
        fig2.update_layout(**PL(420), plot_bgcolor=PBG,
            title=dict(text="Average Score Per Innings", font=dict(color="#1a0800",size=14,family="Orbitron")),
            xaxis=dict(gridcolor=GCOL,
                       title="Average Runs Scored per Innings",
                       title_font=AXIS),
            yaxis=dict(gridcolor=GCOL,
                       title="IPL Team",
                       title_font=AXIS))
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(ts_df.round(3).reset_index(drop=True), use_container_width=True, hide_index=True)

# ─── Tab 2: Batting ───────────────────────────────────────────────────────────
with t2:
    ca, cb = st.columns(2)
    with ca:
        inn_scores = df.groupby(['match_id','inning','batting_team'])['total_runs'].sum().reset_index()
        fig3 = go.Figure()
        for inn_num, col_v in [(1,"#2563eb"),(2,"#c2410c")]:
            s = inn_scores[inn_scores['inning']==inn_num]['total_runs']
            fig3.add_trace(go.Histogram(x=s, name=f"Inning {inn_num}", nbinsx=30,
                marker=dict(color=col_v, opacity=0.7, line=dict(color=GCOL, width=.5)),
                hovertemplate=f"Inning {inn_num}<br>Score: %{{x}}<br>Count: %{{y}}<extra></extra>"))
        fig3.update_layout(**PL(400), plot_bgcolor=PBG,
            title=dict(text="Score Distribution — Inning 1 vs 2", font=dict(color="#1a0800",size=14,family="Orbitron")),
            xaxis=dict(gridcolor=GCOL,
                       title="Total Runs Scored in the Innings",
                       title_font=AXIS),
            yaxis=dict(gridcolor=GCOL,
                       title="Number of Innings",
                       title_font=AXIS),
            barmode='overlay',
            legend=dict(bgcolor="#ffffff", bordercolor=GCOL, borderwidth=1))
        st.plotly_chart(fig3, use_container_width=True)

    with cb:
        ball_dist = df['batsman_runs'].value_counts().sort_index()
        bar_colors = ["#f0d0b0","#2563eb","#059669","#c2410c","#b8860b","#7c3aed","#dc2626"]
        fig4 = go.Figure(go.Bar(
            x=ball_dist.index, y=ball_dist.values,
            marker=dict(color=bar_colors[:len(ball_dist)], line=dict(color="#fff8f0", width=1)),
            text=ball_dist.values, textposition="outside",
            textfont=dict(color="#7a4010", size=10, family="Orbitron"),
            hovertemplate="<b>%{x} runs</b><br>Count: %{y:,}<extra></extra>"
        ))
        fig4.update_layout(**PL(400), plot_bgcolor=PBG,
            title=dict(text="Ball-by-Ball Run Distribution", font=dict(color="#1a0800",size=14,family="Orbitron")),
            xaxis=dict(gridcolor=GCOL,
                       title="Runs Scored off a Single Delivery",
                       title_font=AXIS,
                       tickmode="array",
                       tickvals=list(ball_dist.index),
                       ticktext=[str(v) for v in ball_dist.index]),
            yaxis=dict(gridcolor=GCOL,
                       title="Number of Deliveries",
                       title_font=AXIS))
        st.plotly_chart(fig4, use_container_width=True)

    extras = df[df['extras_type']!='No_Extra']['extras_type'].value_counts()
    if len(extras) > 0:
        st.markdown('<div style="font-family:Orbitron,monospace;font-size:.6rem;'
                    'letter-spacing:4px;color:var(--orange);margin:10px 0;">▸ EXTRAS TYPE DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        fig5 = go.Figure(go.Pie(
            labels=extras.index, values=extras.values, hole=0.6,
            marker=dict(colors=["#b8860b","#2563eb","#059669","#c2410c","#7c3aed"],
                        line=dict(color="#fff8f0", width=2)),
            textfont=dict(color="#1a0800", family="Orbitron", size=10),
            hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>"
        ))
        fig5.update_layout(**PL(300), plot_bgcolor=PBG,
            title=dict(text="Extras Breakdown — Type of Extra Deliveries Bowled",
                       font=dict(color="#1a0800",size=13,family="Orbitron")),
            legend=dict(bgcolor="#ffffff", bordercolor=GCOL, borderwidth=1))
        st.plotly_chart(fig5, use_container_width=True)

# ─── Tab 3: Bowling ───────────────────────────────────────────────────────────
with t3:
    ca, cb = st.columns(2)
    with ca:
        wkt_df = df.groupby('bowling_team')['is_wicket'].sum().sort_values(ascending=False).reset_index()
        wkt_df.columns = ['Team','Wickets']
        fig6 = go.Figure(go.Bar(
            x=wkt_df['Team'], y=wkt_df['Wickets'],
            marker=dict(color=[TEAM_COLORS.get(t,"#b8860b") for t in wkt_df['Team']],
                        opacity=0.85, line=dict(color=GCOL, width=.5)),
            text=wkt_df['Wickets'], textposition='outside',
            textfont=dict(color="#7a4010", size=10, family="Orbitron"),
            hovertemplate="<b>%{x}</b><br>Wickets: %{y}<extra></extra>"
        ))
        fig6.update_layout(**PL(420), plot_bgcolor=PBG,
            title=dict(text="Total Wickets Taken by Team", font=dict(color="#1a0800",size=14,family="Orbitron")),
            xaxis=dict(gridcolor=GCOL, tickangle=-35,
                       title="Bowling Team",
                       title_font=AXIS),
            yaxis=dict(gridcolor=GCOL,
                       title="Total Wickets Taken",
                       title_font=AXIS))
        st.plotly_chart(fig6, use_container_width=True)

    with cb:
        dis = df[df['dismissal_kind']!='No_Dismissal']['dismissal_kind'].value_counts().head(8)
        fig7 = go.Figure(go.Bar(
            y=dis.index, x=dis.values, orientation='h',
            marker=dict(color="#c2410c", opacity=0.85, line=dict(color=GCOL, width=.5)),
            text=dis.values, textposition='outside',
            textfont=dict(color="#7a4010", size=10, family="Orbitron"),
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>"
        ))
        fig7.update_layout(**PL(420), plot_bgcolor=PBG,
            title=dict(text="Dismissal Types", font=dict(color="#1a0800",size=14,family="Orbitron")),
            xaxis=dict(gridcolor=GCOL,
                       title="Number of Dismissals",
                       title_font=AXIS),
            yaxis=dict(gridcolor=GCOL,
                       title="Dismissal Method",
                       title_font=AXIS))
        st.plotly_chart(fig7, use_container_width=True)

# ─── Tab 4: Win Probability Flow ──────────────────────────────────────────────
with t4:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:.6rem;'
                'letter-spacing:4px;color:var(--orange);margin-bottom:12px;">▸ SELECT A MATCH</div>',
                unsafe_allow_html=True)
    match_ids = sorted(df['match_id'].unique())[:50]
    sel_match = st.selectbox("Match ID", match_ids)

    mdf_sel  = df[df['match_id']==sel_match].copy()
    teams_in = mdf_sel['batting_team'].unique()
    team_a   = teams_in[0] if len(teams_in)>0 else "Team A"
    team_b   = teams_in[1] if len(teams_in)>1 else "Team B"

    fig8 = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Run Rate Evolution (Runs per Over)","Cumulative Score Progression"],
        vertical_spacing=0.12, row_heights=[0.5,0.5])

    for inn, cv in [(1,"#2563eb"),(2,"#c2410c")]:
        idf = mdf_sel[mdf_sel['inning']==inn].sort_values('ball_number')
        if len(idf)==0: continue
        tn  = idf['batting_team'].iloc[0]
        c   = TEAM_COLORS.get(tn, cv)
        fig8.add_trace(go.Scatter(x=idf['overs_completed'], y=idf['run_rate'],
            mode='lines', name=f"Inn {inn} RR — {tn}", line=dict(color=c, width=2),
            hovertemplate=f"Inn {inn}<br>Over: %{{x:.1f}}<br>Run Rate: %{{y:.2f}} runs/over<extra></extra>"),
            row=1, col=1)
        fig8.add_trace(go.Scatter(x=idf['overs_completed'], y=idf['current_score'],
            mode='lines+markers', name=f"Inn {inn} Score — {tn}",
            line=dict(color=c, width=2.5), marker=dict(size=3, color=c),
            hovertemplate=f"Inn {inn}<br>Over: %{{x:.1f}}<br>Score: %{{y}} runs<extra></extra>"),
            row=2, col=1)
        wk = idf[idf['is_wicket']==1]
        fig8.add_trace(go.Scatter(x=wk['overs_completed'], y=wk['current_score'],
            mode='markers', name=f"Inn {inn} Wicket",
            marker=dict(symbol='x', size=10, color='#dc2626', line=dict(width=2)),
            hovertemplate="Wicket at over %{x:.1f}<extra></extra>"),
            row=2, col=1)

    fig8.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor=PBG,
        font=dict(color="#3d1a00", family="Inter"), height=520,
        title=dict(text=f"Match #{sel_match} — {team_a} vs {team_b}",
                   font=dict(color="#1a0800", size=14, family="Orbitron")),
        legend=dict(bgcolor="#ffffff", bordercolor=GCOL, borderwidth=1),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#c2410c"),
        margin=dict(t=60, b=40, l=20, r=20)
    )
    fig8.update_xaxes(gridcolor=GCOL, title_text="Overs Completed", title_font=AXIS, row=2, col=1)
    fig8.update_yaxes(gridcolor=GCOL, title_text="Run Rate (runs/over)", title_font=AXIS, row=1, col=1)
    fig8.update_yaxes(gridcolor=GCOL, title_text="Cumulative Runs Scored", title_font=AXIS, row=2, col=1)
    st.plotly_chart(fig8, use_container_width=True)

# ─── Tab 5: Over-by-over ──────────────────────────────────────────────────────
with t5:
    over_stats = df.groupby('over').agg(
        avg_runs=('total_runs','mean'),
        avg_wickets=('is_wicket','mean'),
    ).reset_index()

    fig9 = make_subplots(rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Average Runs per Over","Wicket Frequency per Over"],
        vertical_spacing=0.15, row_heights=[0.6,0.4])
    fig9.add_trace(go.Bar(x=over_stats['over'], y=over_stats['avg_runs'],
        marker=dict(color=over_stats['avg_runs'],
                    colorscale=[[0,"#f0d0b0"],[0.5,"#c2410c"],[1,"#b8860b"]],
                    showscale=False, line=dict(color="#fff8f0", width=.5)),
        hovertemplate="Over %{x}<br>Avg Runs: %{y:.2f} per delivery<extra></extra>"), row=1, col=1)
    fig9.add_trace(go.Bar(x=over_stats['over'], y=over_stats['avg_wickets']*6,
        marker=dict(color="#dc2626", opacity=0.8, line=dict(color="#fff8f0", width=.5)),
        hovertemplate="Over %{x}<br>Wickets per over: %{y:.3f}<extra></extra>"), row=2, col=1)

    fig9.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor=PBG,
        font=dict(color="#3d1a00", family="Inter"), height=500,
        showlegend=False, margin=dict(t=50, b=40, l=20, r=20)
    )
    fig9.update_xaxes(gridcolor=GCOL, title_text="Over Number (1–20)", title_font=AXIS, row=2, col=1)
    fig9.update_yaxes(gridcolor=GCOL, title_text="Avg Runs per Delivery", title_font=AXIS, row=1, col=1)
    fig9.update_yaxes(gridcolor=GCOL, title_text="Avg Wickets per Over", title_font=AXIS, row=2, col=1)
    st.plotly_chart(fig9, use_container_width=True)

    # Phase analysis
    pp    = float(df[df['over']<6]['batsman_runs'].mean())
    mid   = float(df[(df['over']>=6)&(df['over']<16)]['batsman_runs'].mean())
    death = float(df[df['over']>=16]['batsman_runs'].mean())

    st.markdown('<div style="font-family:Orbitron,monospace;font-size:.6rem;'
                'letter-spacing:4px;color:var(--orange);margin:20px 0 10px;">▸ PHASE ANALYSIS</div>',
                unsafe_allow_html=True)
    pcols = st.columns(3)
    for col, ph, val, clr in zip(pcols,
        ["⚡ Powerplay (Overs 1–6)","🏏 Middle (Overs 7–16)","💥 Death (Overs 17–20)"],
        [pp, mid, death],
        ["#b8860b","#2563eb","#dc2626"]):
        with col:
            st.markdown(f"""<div style='background:#ffffff;border:1px solid var(--border);
                border-left:3px solid {clr};border-radius:10px;padding:18px;text-align:center;
                box-shadow:0 2px 5px rgba(0,0,0,.04);'>
                <div style='font-size:.8rem;color:var(--text2);font-weight:600;'>{ph}</div>
                <div style='font-family:Orbitron,monospace;font-size:1.8rem;font-weight:900;
                    color:{clr};margin-top:8px;'>{val:.3f}</div>
                <div style='font-size:.68rem;color:var(--text4);margin-top:4px;'>avg runs per delivery</div>
            </div>""", unsafe_allow_html=True)

# ─── Tab 6: Raw Dataset ───────────────────────────────────────────────────────
with t6:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:.6rem;'
                'letter-spacing:4px;color:var(--orange);margin-bottom:12px;">▸ RAW DATASET PREVIEW</div>',
                unsafe_allow_html=True)
    src_df = df_raw if DATA_LOADED else df
    cf1, cf2 = st.columns([2,1])
    with cf1:
        sel_cols = st.multiselect("Select Columns", list(src_df.columns),
                                  default=list(src_df.columns[:8]))
    with cf2:
        n_rows = st.slider("Rows to show", 10, 200, 50)

    show_df = src_df[sel_cols].head(n_rows) if sel_cols else src_df.head(n_rows)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    if DATA_LOADED:
        st.markdown('<div style="font-family:Orbitron,monospace;font-size:.6rem;'
                    'letter-spacing:4px;color:var(--orange);margin:20px 0 10px;">▸ DATASET STATISTICS</div>',
                    unsafe_allow_html=True)
        st.dataframe(df_raw.describe().round(3), use_container_width=True)