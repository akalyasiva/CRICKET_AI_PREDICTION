"""utils/nav.py — Fixed Navigation Bar with correct Streamlit page URLs"""
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# HOW STREAMLIT GENERATES PAGE URLS (v1.x multi-page apps):
#
#   File name              →  URL path
#   pages/2_Predict.py     →  /Predict          (number+underscore prefix stripped)
#   pages/3_XAI_Commentary.py → /XAI_Commentary
#   pages/4_Model_Analytics.py → /Model_Analytics
#   pages/5_Historical_Analytics.py → /Historical_Analytics
#   Home.py (root)         →  /  (always the root)
#
# ─────────────────────────────────────────────────────────────────────────────

NAV_LINKS = [
    ("🏠", "Home",        "/"),
    ("⚡", "Predict",     "/Predict"),
    ("💬", "Commentary",  "/XAI_Commentary"),
    ("📊", "Analytics",   "/Model_Analytics"),
    ("📈", "Historical",  "/Historical_Analytics"),
]

_CSS = """<style>
#MainMenu, header, footer { visibility: hidden !important; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.main .block-container {
    padding-top: 72px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1280px;
    margin: 0 auto;
}
.topnav {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 99999;
    height: 58px;
    background: #050e1f;
    border-bottom: 2px solid #1e3a8a;
    display: flex;
    align-items: center;
    padding: 0 40px;
    gap: 4px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.6);
}
.topnav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-right: 36px;
    flex-shrink: 0;
    text-decoration: none;
}
.brand-icon { font-size: 1.3rem; line-height: 1; }
.brand-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: .95rem;
    font-weight: 800;
    letter-spacing: 1px;
    background: linear-gradient(90deg, #FFD700 0%, #FF8C00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-transform: uppercase;
    white-space: nowrap;
}
.nav-spacer { flex: 1; }
.nav-item {
    display: flex;
    align-items: center;
    gap: 6px;
    text-decoration: none !important;
    padding: 8px 18px;
    border-radius: 10px;
    border: 1.5px solid transparent;
    transition: all 0.2s ease;
    white-space: nowrap;
    cursor: pointer;
}
.nav-icon { font-size: .95rem; line-height: 1; }
.nav-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: .8rem;
    font-weight: 700;
    letter-spacing: .5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    transition: color 0.2s;
}
.nav-item:hover .nav-label { color: #ffffff; }
.nav-item:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(255,255,255,0.12);
}
.nav-active {
    background: rgba(37,99,235,0.2) !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.25);
}
.nav-active .nav-label {
    color: #ffffff !important;
    font-weight: 800 !important;
}
</style>"""


def inject_navbar(active: str = "Home"):
    """
    Inject fixed navbar using st.markdown.

    IMPORTANT — Streamlit URL rules:
      pages/2_Predict.py          → /Predict
      pages/3_XAI_Commentary.py   → /XAI_Commentary
      pages/4_Model_Analytics.py  → /Model_Analytics
      pages/5_Historical_Analytics.py → /Historical_Analytics

    Streamlit strips leading digits+underscore from the filename,
    then uses the rest (keeping internal underscores) as the URL slug.
    """

    # Step 1: CSS (plain string, no f-string, no brace conflicts)
    st.markdown(_CSS, unsafe_allow_html=True)

    # Step 2: Build nav items with string concatenation only
    items = ""
    for icon, label, href in NAV_LINKS:
        a = active.lower().replace(" ", "").replace("_", "")
        l = label.lower().replace(" ", "").replace("_", "")
        is_active = (a in l) or (l in a)
        cls = "nav-active" if is_active else ""
        items += (
            '<a href="' + href + '" target="_self" class="nav-item ' + cls + '">'
            + '<span class="nav-icon">' + icon + '</span>'
            + '<span class="nav-label">' + label + '</span>'
            + '</a>'
        )

    # Step 3: Inject navbar HTML (plain string concatenation)
    st.markdown(
        '<div class="topnav">'
        + '<a href="/" target="_self" class="topnav-brand">'
        + '<span class="brand-icon">🏏</span>'
        + '<span class="brand-text">Cricket AI</span>'
        + '</a>'
        + '<div class="nav-spacer"></div>'
        + items
        + '</div>',
        unsafe_allow_html=True
    )