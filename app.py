import base64
import hashlib
import os

import streamlit as st
from google import genai

from recommender import Recommender, rag_recommend

# ---------- Page setup ----------
st.set_page_config(page_title="prettyplease", page_icon="💋", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "static", "prettyplease_logo.png")

TILE_COLORS = ["#FFD8E6", "#FFC1D9", "#FFE1EC", "#F7A8C4", "#FBC7DA"]


import io
from PIL import Image

@st.cache_data
def get_logo_base64(path, pad_ratio=0.06):
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        left, top, right, bottom = bbox
        pad_x = int((right - left) * pad_ratio)
        pad_y = int((bottom - top) * pad_ratio)
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(img.width, right + pad_x)
        bottom = min(img.height, bottom + pad_y)
        img = img.crop((left, top, right, bottom))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@st.cache_resource
def get_recommender():
    return Recommender()


@st.cache_resource
def get_genai_client():
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key) if api_key else None


def stars(rating):
    if rating is None or rating != rating:
        return "☆☆☆☆☆"
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("½" if half else "") + "☆" * empty


def tile_color(seed_text):
    idx = int(hashlib.md5(str(seed_text).encode()).hexdigest(), 16) % len(TILE_COLORS)
    return TILE_COLORS[idx]


def render_cards(df, score_col="final_score"):
    cards = []
    for _, row in df.iterrows():
        brand = row.get("Product Brand", "") or ""
        name = row.get("Product Name", "") or ""
        price = row.get("Product Price", None)
        rating = row.get("Product Rating", None)
        ptype = row.get("product_type", "") or ""
        score = row.get(score_col, None)
        initial = (brand[:1] or name[:1] or "?").upper()
        match_pct = int(round(score * 100)) if score is not None and score == score else None

        price_html = f"₹{price:.0f}" if price is not None and price == price else "Price unavailable"
        badge_html = (
            f'<div class="pp-stamp">{match_pct}%<br><span>match</span></div>'
            if match_pct is not None else ""
        )

        card_html = (
            f'<div class="pp-card">{badge_html}'
            f'<div class="pp-tile" style="background:{tile_color(brand or name)}">{initial}</div>'
            f'<div class="pp-brand">{brand.upper()}</div>'
            f'<div class="pp-name">{name}</div>'
            f'<div class="pp-type">{ptype}</div>'
            f'<div class="pp-row">'
            f'<span class="pp-price">{price_html}</span>'
            f'<span class="pp-rating">{stars(rating)}</span>'
            f'</div></div>'
        )
        cards.append(card_html)

    return f'<div class="pp-grid">{"".join(cards)}</div>'


recommender = get_recommender()
client = get_genai_client()
logo_b64 = get_logo_base64(LOGO_PATH)

# ---------- Styles ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,600;0,700;1,600&family=Manrope:wght@400;500;600;700&family=Caveat:wght@600&display=swap');

:root {{
    --ink: #2B1420;
    --pink: #FF3D8F;
    --kiss: #D63857;
    --blush: #FFF7F8;
    --petal: #FFE1EC;
    --mauve: #8B6B78;
}}

.stApp {{
    background: var(--blush);
    font-family: 'Manrope', sans-serif;
    color: var(--ink);
}}

#MainMenu, footer, header {{visibility: hidden;}}

.block-container {{
    padding-top: 1rem;
    max-width: 1100px;
}}

.pp-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid var(--petal);
    margin-bottom: 1.5rem;
}}

.pp-header img {{ height: 68px; }}

.pp-nav-tag {{
    font-family: 'Manrope', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    color: var(--mauve);
    text-transform: uppercase;
}}

.pp-hero {{
    position: relative;
    background: linear-gradient(135deg, var(--petal) 0%, #FFF0F5 100%);
    border-radius: 24px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 1.5rem;
    overflow: hidden;
}}

.pp-hero-watermark {{
    position: absolute;
    right: -30px;
    top: -20px;
    width: 260px;
    opacity: 0.10;
    transform: rotate(-8deg);
}}

.pp-hero h1 {{
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    position: relative;
    z-index: 1;
}}

.pp-hero h1 em {{
    font-family: 'Caveat', cursive;
    font-style: normal;
    color: var(--pink);
    font-size: 1.15em;
}}

.pp-hero p {{
    color: var(--mauve);
    font-size: 1rem;
    margin: 0;
    position: relative;
    z-index: 1;
}}

div[data-testid="stTextInput"] > div,
div[data-testid="stTextInputRootElement"],
div[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
    border-radius: 999px !important;
    overflow: hidden !important;
}}

div[data-testid="stTextInput"] input {{
    border-radius: 999px !important;
    border: 1.5px solid var(--petal) !important;
    padding: 0.85rem 1.4rem !important;
    font-size: 1rem !important;
    font-family: 'Manrope', sans-serif;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    border-radius: 999px !important;
    border: 1.5px solid var(--petal) !important;
}}

div[data-testid="stButton"] button {{
    background: linear-gradient(135deg, var(--pink), var(--kiss));
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}

div[data-testid="stButton"] button:hover {{
    filter: brightness(1.05);
    color: white;
}}

.pp-editorial {{
    font-family: 'Fraunces', serif;
    font-style: italic;
    font-size: 1.15rem;
    background: white;
    border-left: 3px solid var(--pink);
    padding: 1rem 1.5rem;
    border-radius: 0 12px 12px 0;
    margin: 1rem 0 1.5rem 0;
}}

.pp-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 1rem;
}}

.pp-card {{
    position: relative;
    background: white;
    border-radius: 18px;
    padding: 1.1rem;
    box-shadow: 0 2px 10px rgba(43,20,32,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}

.pp-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(43,20,32,0.10);
}}

.pp-stamp {{
    position: absolute;
    top: -10px;
    right: -8px;
    background: var(--kiss);
    color: white;
    width: 52px;
    height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    font-size: 0.7rem;
    font-weight: 700;
    line-height: 1;
    transform: rotate(-10deg);
    border: 2px dashed rgba(255,255,255,0.6);
}}

.pp-stamp span {{ font-size: 0.5rem; font-weight: 500; }}

.pp-tile {{
    width: 100%;
    height: 90px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--kiss);
    margin-bottom: 0.7rem;
}}

.pp-brand {{
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--mauve);
    margin-bottom: 0.15rem;
}}

.pp-name {{
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1rem;
    line-height: 1.25;
    margin-bottom: 0.2rem;
}}

.pp-type {{
    font-size: 0.75rem;
    color: var(--mauve);
    margin-bottom: 0.6rem;
}}

.pp-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.pp-price {{
    font-weight: 700;
    color: var(--kiss);
}}

.pp-rating {{
    color: var(--pink);
    font-size: 0.85rem;
}}

.stApp, .stApp p, .stApp label, .stApp span, .stApp div {{
    color: var(--ink);
}}

div[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
    border: none !important;
    box-shadow: none !important;
    background: white !important;
}}

div[data-testid="stTextInput"] input {{
    background: white !important;
    color: var(--ink) !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    box-shadow: none !important;
}}

div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
    background: white !important;
    color: var(--ink) !important;
}}

ul[data-baseweb="menu"] {{
    background: white !important;
}}

ul[data-baseweb="menu"] li {{
    color: var(--ink) !important;
}}

ul[data-baseweb="menu"] li:hover {{
    background: var(--petal) !important;
}}

div[data-testid="stButton"] button,
div[data-testid="stButton"] button p {{
    color: white !important;
}}

div[data-testid="stRadio"] label span,
div[data-testid="stCheckbox"] label span {{
    color: var(--ink) !important;
}}

</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
logo_html = (
    f'<img src="data:image/png;base64,{logo_b64}" />' if logo_b64 else '<span style="font-family:Fraunces,serif;font-size:1.6rem;color:var(--pink);">prettyplease</span>'
)
st.markdown(f"""
<div class="pp-header">
    {logo_html}
    <span class="pp-nav-tag">Makeup · Skin · Hair · Fragrance</span>
</div>
""", unsafe_allow_html=True)

# ---------- Hero ----------
watermark_html = f'<img class="pp-hero-watermark" src="data:image/png;base64,{logo_b64}" />' if logo_b64 else ""
st.markdown(f"""
<div class="pp-hero">
    {watermark_html}
    <h1>Find your next <em>favourite</em></h1>
    <p>Tell us what you're after, we'll match it against 500+ real Nykaa products.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Controls ----------
mode = st.radio("Search by", ["What I'm looking for", "A product I already like"], horizontal=True, label_visibility="collapsed")

col1, col2 = st.columns(2)
categories = [None] + sorted(recommender.df["category_level_1"].dropna().unique().tolist())
category_filter = col1.selectbox("Category", categories, format_func=lambda x: x or "All categories")
price_filter = col2.selectbox("Price band", [None, "Budget", "Mid-range", "Premium", "Luxury"], format_func=lambda x: x or "Any price")

if mode == "What I'm looking for":
    query = st.text_input("", placeholder="e.g. hydrating face moisturiser for dry skin")
    use_rag = st.checkbox("Add a written recommendation", value=bool(client), disabled=client is None)

    if st.button("Find my picks") and query:
        with st.spinner("Matching products..."):
            if use_rag and client:
                result = rag_recommend(client, recommender, query, category_filter=category_filter, price_band_filter=price_filter)
                if result["retrieved_products"] is not None:
                    st.markdown(f'<div class="pp-editorial">{result["recommendation_text"]}</div>', unsafe_allow_html=True)
                results = result["retrieved_products"]
            else:
                results = recommender.recommend_from_preferences(query, category_filter=category_filter, price_band_filter=price_filter)

        if results is None:
            st.warning("No matches found. Try loosening your filters.")
        else:
            st.markdown(render_cards(results), unsafe_allow_html=True)
else:
    product_name = st.text_input("", placeholder="Enter a product name")
    if st.button("Find similar") and product_name:
        results = recommender.hybrid_recommend(product_name)
        if results is None:
            st.warning(f"Couldn't find a close match for '{product_name}' in the catalog.")
        else:
            st.markdown(render_cards(results), unsafe_allow_html=True)