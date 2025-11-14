# app.py — Updated: working product link buttons + embedding-based title matching
import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime
from pathlib import Path
import time
import math
import json
import numpy as np

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

# Optional sentence-transformers (embeddings)
EMBED_AVAILABLE = False
EMBED_MODEL = None
EMBED_DIM = None
EMBED_THRESHOLD = 0.72  # cosine similarity threshold to consider a match

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
    EMBED_AVAILABLE = True
    st_log = lambda *a, **k: None
except Exception as e:
    # If embedding lib not available, we'll fallback to token overlap
    EMBED_AVAILABLE = False
    EMBED_MODEL = None

# ---------------- CONFIG ----------------
DB_PATH = "price_compare.db"
CSV_PATH = "search_results.csv"
ALERTS_CSV = "alerts.csv"
MIN_LABELS_TO_TRAIN = 10
INDIAN_STORES = [
    "Amazon", "Amazon.in", "Flipkart", "Croma",
"Reliance Digital", "Tata Cliq", "Vijay Sales",
"Poorvika", "Samsung India", "Mi India",
"Boat Lifestyle", "OnePlus India", "Myntra",
"AJIO", "Snapdeal", "Paytm Mall", "Tata CLiQ",
"ShopClues", "eBay India", "IndiaMART",
"OLX", "Quikr", "SastaSundar", "Meesho",
"Pepperfry", "Urban Ladder", "BigBasket",
"Blinkit", "DMart", "Spencers", "Reliance Fresh",
"Nature’s Basket", "Natures Basket", "More Store",
"Easyday", "FreshToHome", "Milkbasket",
"Big Bazaar", "Lenskart", "Bose India",
"Audio House", "Sangeetha Mobiles", "UniverCell",
"The Mobile Store", "Ezone", "HomeTown",
"TataCliq Electronics", "Samsung Online Store",
"Mi Store", "OnePlus Store", "Zara India",
"H&M India", "Lifestyle", "Shoppers Stop",
"Pantaloons", "Max Fashion", "Brand Factory",
"Fabindia", "W for Woman", "Biba", "Wrogn",
"Arrow", "Raymond", "Allen Solly",
"Louis Philippe", "Jack & Jones India",
"Koovs", "Bewakoof", "The Souled Store",
"Nykaa Fashion", "Clovia", "Bata India",
"Woodland", "Metro Shoes", "Puma India",
"Nike India", "Adidas India", "Mochi",
"Hidesign", "Health & Glow", "Nykaa",
"Purplle", "MyGlamm", "The Body Shop India",
"Kama Ayurveda", "Forest Essentials",
"Sugar Cosmetics", "Colorbar", "Glamveda",
"PharmEasy", "1mg", "Netmeds", "Apollo Pharmacy",
"MedPlus", "CareOnGo", "Wellness Forever",
"Healthkart", "Durex India", "Manforce India",
"Moods Condoms", "Condom Junction",
"Kamasutra India", "Grocery Bigbasket",
"Reliance Smart", "Farmley", "Licious",
"Home Centre", "WoodenStreet", "FabFurnish",
"Durian Furniture", "@home", "Tanishq",
"CaratLane", "Kalyan Jewellers", "Malabar Gold",
"BlueStone", "FirstCry", "Hopscotch",
"BabyChakra", "Mothercare India", "Decathlon India",
"Sportking", "HRX", "Crossword", "SapnaOnline",
"Kitab Khana", "CarDekho", "BikeDekho",
"AutoZone India", "BharatBenz Store", "Nilgiris",
"Ratnadeep", "Metro Cash & Carry", "HyperCity",
"MD Computers", "PrimeABGB", "Vedant Computers",
"TheITDepot", "The Man Company", "The Minimalist",
"Chumbak", "FabAlley", "Zivame", "Plum Goodness",
"Mamaearth", "The Derma Co", "Cashify",
"ReGlobe", "ReStore", "Farmley India",
"Licious India", "KitchenAid India", "Hawkins Online",
"Prestige Appliances", "Bajaj Electronics",
"Staples India", "Amazon Business India",
"Heads Up For Tails", "PetsWorld", "Drools Store",
"Ferns N Petals", "FlowerAura", "Winni",
"JioMart", "BigBasket Local", "Smartprix",
"91Mobiles", "Gadget360", "Shop101", "Voonik",
"LimeRoad", "Craftsvilla", "Zansaar",
"Boat India", "boAt", "OnePlus", "Apple India",
"Realme India", "Vivo India", "Oppo India",
"Motorola India", "HP Store India", "Dell India",
"Asus India", "Acer India", "Lenovo India",
"MSI India", "Canon India", "Nikon India",
"DJI India", "GoPro India", "Syska India",
"Philips India", "Havells India", "Usha India",
"IFB Appliances", "Bosch India", "LG India",
"Whirlpool India", "Voltas India", "Godrej Appliances",
"Haier India", "TCL India", "Vu India",
"Ikea India", "JBL India", "Sennheiser India",
"Skullcandy India", "Harman India", "Marshall India",
"Big Fashion", "Aditya Birla Fashion",
"Nykaa Beauty", "Myntra Shoes", "Myntra Accessories",
"Amazon Fashion India", "Flipkart Fashion India",
"Croma Retail", "Reliance Digital Store",
"Spar Hypermarket", "Star Bazaar", "More Retail",
"Jiomart Grocery", "Armani Exchange India",
"Michael Kors India", "Charles & Keith India",
"Sephora India", "Bath & Body Works India",
"Miniso India", "Muji India", "Decathlon Sports",
"SportsJam", "Wildcraft India", "American Tourister India",
"Safari Bags India", "Skybags India",
"Campus Shoes India", "Skechers India", 
"Wrangler India", "Levis India", "Spykar", "U.S. Polo Assn",
"Peter England", "Van Heusen", "Tommy Hilfiger India",
"Calvin Klein India", "Superdry India",
"Jockey India", "Rupa", "Lux Cozi",
"Dollar Industries", "Max Innerwear", "Amante India",
"Enamor India", "PrettySecrets", "VIP Bags",
"Aristocrat Bags", "F Gear India", "Roadster (Myntra)",
"H&M Home", "Marks & Spencer India",
"Uniqlo India", "The Children's Place India",
"Forever 21 India", "Pantaloons Fashion",
"Ajio Luxe", "Nykaa Luxe", "TataCliq Luxury",
"Reliance Trends", "Central Mall Online",
"Trent Westside", "Westside Online",
"FoodHall India", "Spencer Retail Grocery",
"More Hypermarket", "Reliance Retail Online",
"Amazon Pantry India", "Flipkart Supermart",
"Zepto", "Urban Company Store",
"Wakefit", "Sleepyhead", "Duroflex India",
"Kurlon India", "Sleepwell India"
]
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"
# -----------------------------------------

st.set_page_config(page_title="India Price Compare (links+embeddings)", layout="wide")
st.title("🇮🇳 Best Deal Finder — Working Links + Embedding Matching")

# Load secrets
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# ---------------- DB init (embedding column added) ----------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS searches (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      query TEXT,
      created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      search_id INTEGER,
      title TEXT,
      store TEXT,
      price REAL,
      rating REAL,
      reviews INTEGER,
      link TEXT,
      image TEXT,
      score REAL,
      embedding TEXT,     -- JSON array (optional)
      created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS purchases (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      result_id INTEGER,
      search_id INTEGER,
      bought_price REAL,
      created_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      search_query TEXT,
      product_title TEXT,
      store TEXT,
      threshold_type TEXT,
      threshold_value REAL,
      triggered_price REAL,
      triggered_at TEXT
    )""")
    conn.commit()
    return conn

db_conn = init_db()

# ---------------- helpers ----------------
def get_usd_to_inr():
    try:
        r = requests.get(USD_TO_INR_API, timeout=8)
        data = r.json()
        return float(data.get("rates", {}).get("INR", 83.0))
    except:
        return 83.0

usd_inr_rate = get_usd_to_inr()

def clean_price(raw_price):
    if not raw_price:
        return None
    p = str(raw_price)
    p = p.replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    for sep in ["/", "per", "month", "mo", "EMI", " emi", "₹"]:
        p = p.split(sep)[0]
    if "$" in raw_price or "USD" in raw_price or "US$" in raw_price:
        nums = "".join(ch for ch in p if (ch.isdigit() or ch == "."))
        if not nums:
            return None
        return float(nums) * usd_inr_rate
    filtered = "".join(ch for ch in p if (ch.isdigit() or ch == "."))
    if filtered == "":
        return None
    price = float(filtered)
    if price < 50:
        return None
    return price

def is_indian_store(store_name):
    if not store_name:
        return False
    for s in INDIAN_STORES:
        if s.lower() in store_name.lower():
            return True
    if ".in" in store_name.lower() or "india" in store_name.lower():
        return True
    return False

# embedding helpers
def compute_embedding(text):
    if not EMBED_AVAILABLE or EMBED_MODEL is None:
        return None
    try:
        vec = EMBED_MODEL.encode(text, normalize_embeddings=True)
        return vec.tolist()
    except Exception:
        return None

def embedding_from_json(txt):
    if not txt:
        return None
    try:
        arr = json.loads(txt)
        return np.array(arr, dtype=float)
    except Exception:
        return None

def cosine_sim(a, b):
    if a is None or b is None:
        return -1.0
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------- DB logging ----------------
def log_search(q):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO searches(query,created_at) VALUES(?,?)", (q, t))
    db_conn.commit()
    return cur.lastrowid

def log_results(sid, rows):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    for r in rows:
        emb_json = None
        if r.get("embedding") is not None:
            emb_json = json.dumps(r["embedding"])
        cur.execute("""
            INSERT INTO results(search_id,title,store,price,rating,reviews,link,image,score,embedding,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """, (sid, r["title"], r["store"], r["price"], r["rating"], r["reviews"], r["link"], r["image"], r["score"], emb_json, t))
    db_conn.commit()

def append_to_csv(rows):
    df = pd.DataFrame(rows)
    file_exists = os.path.isfile(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)

# ---------------- SerpAPI fetch ----------------
def fetch_serpapi(query, selected_stores=None, max_results=30):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": max_results,
        "gl": "in",
        "hl": "en"
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    out = []
    for item in data.get("shopping_results", []):
        store = item.get("source") or item.get("store") or ""
        if selected_stores and len(selected_stores) > 0:
            if not any(sel.lower() in store.lower() for sel in selected_stores):
                continue
        else:
            if not is_indian_store(store):
                continue
        raw_price = item.get("price") or ""
        price = clean_price(raw_price)
        if not price:
            continue
        title = item.get("title") or ""
        emb = compute_embedding(title)  # may be None
        out.append({
            "title": title,
            "store": store,
            "price": float(price),
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "link": item.get("link") or "",
            "image": item.get("thumbnail") or "",
            "embedding": emb
        })
    return out

# ---------------- price history matching (embedding-aware) ----------------
def fetch_price_history_for_result(title, store, current_embedding=None):
    cur = db_conn.cursor()
    cur.execute("SELECT title, store, price, created_at, embedding FROM results WHERE store LIKE ?", (f"%{store}%",))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["price", "created_at"])
    cols = ["title", "store", "price", "created_at", "embedding"]
    df = pd.DataFrame(rows, columns=cols)
    # compute match score either by embedding cosine or token overlap
    matches = []
    for _, r in df.iterrows():
        matched = False
        if current_embedding is not None and r["embedding"]:
            past_emb = embedding_from_json(r["embedding"])
            if past_emb is not None:
                sim = cosine_sim(current_embedding, past_emb)
                if sim >= EMBED_THRESHOLD:
                    matched = True
        if not matched:
            # fallback token overlap
            try:
                if title_matches(r["title"], title):
                    matched = True
            except Exception:
                matched = False
        if matched:
            matches.append((r["price"], r["created_at"]))
    if not matches:
        # fallback: return recent prices for this store
        df2 = df.copy()
        df2["created_at"] = pd.to_datetime(df2["created_at"])
        df2 = df2.sort_values("created_at")
        return df2[["price", "created_at"]]
    # build dataframe
    dfm = pd.DataFrame(matches, columns=["price", "created_at"])
    dfm["created_at"] = pd.to_datetime(dfm["created_at"])
    dfm = dfm.sort_values("created_at")
    return dfm

# fallback token overlap matcher (used by previous code)
def title_matches(a, b):
    if not a or not b:
        return False
    a_tokens = set([t.lower() for t in a.split() if len(t) > 2])
    b_tokens = set([t.lower() for t in b.split() if len(t) > 2])
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens.intersection(b_tokens)
    return len(overlap) >= max(1, min(3, int(0.4 * min(len(a_tokens), len(b_tokens)))))

# ---------------- ranking ----------------
def heuristic_rank(df):
    if df.empty:
        return df
    d = df.copy()
    maxp = d["price"].max()
    d["rating_norm"] = d["rating"] / 5.0
    d["reviews_norm"] = d["reviews"] / (d["reviews"].max() + 1e-9)
    d["price_norm"] = (maxp - d["price"]) / (maxp + 1e-9)
    d["score"] = d["price_norm"] * 0.7 + d["rating_norm"] * 0.2 + d["reviews_norm"] * 0.1
    return d

def try_train_and_predict(df_features):
    try:
        q = """
            SELECT r.price, r.rating, r.reviews, 
                   CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END AS bought
            FROM results r LEFT JOIN purchases p ON p.result_id = r.id
        """
        df_train = pd.read_sql_query(q, db_conn)
    except Exception:
        return None
    if df_train.shape[0] < MIN_LABELS_TO_TRAIN or not LGB_AVAILABLE:
        return None
    df_train = df_train.fillna(0)
    feature_cols = ["price", "rating", "reviews"]
    X = df_train[feature_cols]
    y = df_train["bought"]
    lgb_train = lgb.Dataset(X, label=y)
    params = {"objective":"binary","metric":"binary_logloss","verbosity": -1}
    model = lgb.train(params, lgb_train, num_boost_round=100)
    X_new = df_features[feature_cols].fillna(0)
    preds = model.predict(X_new)
    return preds

# ---------------- UI ----------------
with st.sidebar:
    st.header("Settings")
    selected_stores = st.multiselect("Filter stores (leave empty = all Indian stores)", options=INDIAN_STORES)
    st.markdown("---")
    st.write("Embedding model: " + ("available" if EMBED_AVAILABLE else "not available (fallback to token-match)"))
    if EMBED_AVAILABLE:
        st.write(f"Embed dim: {EMBED_DIM}  — threshold: {EMBED_THRESHOLD}")

st.info("Product links are clickable buttons (Open in new tab). Historical matching uses embeddings if available; otherwise token overlap.")

query = st.text_input("Enter product name (India only)", placeholder="e.g. iPhone 15, Samsung QLED TV, JBL Flip 6")

if query:
    with st.spinner("Searching India stores via SerpAPI..."):
        results = fetch_serpapi(query, selected_stores)
    if not results:
        st.error("No valid Indian product listings found. Try a different query or broaden store filter.")
    else:
        df = pd.DataFrame(results)
        preds = try_train_and_predict(df[["price","rating","reviews"]]) if not df.empty else None
        if preds is not None:
            df["score"] = preds
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        else:
            df = heuristic_rank(df).reset_index(drop=True)

        # Prepare rows to log: include embedding JSON
        rows_to_log = []
        for _, r in df.iterrows():
            rows_to_log.append({
                "title": r["title"],
                "store": r["store"],
                "price": float(r["price"]),
                "rating": float(r["rating"]),
                "reviews": int(r["reviews"]),
                "link": r["link"],
                "image": r["image"],
                "score": float(r["score"]),
                "embedding": r.get("embedding")  # may be None or list
            })

        sid = log_search(query)
        log_results(sid, rows_to_log)
        append_to_csv(rows_to_log)

        # TOP 3 + working "Open Product Page" buttons (anchor-button)
        st.subheader("🏆 Top 3 Deals (India) — Click button to open product page")
        for i, row in df.head(3).iterrows():
            col1, col2, col3 = st.columns([1,4,1])
            with col1:
                if row.get("image"):
                    st.image(row["image"], width=110)
            with col2:
                st.markdown(f"### {row['title']}")
                st.write(f"**Store:** {row['store']}")
                st.write(f"**Price:** ₹{int(row['price']):,}")
                st.write(f"⭐ {row['rating']} ({row['reviews']} reviews)")
                # show link as small anchor too
                if row.get("link"):
                    st.markdown(f"[Open link in new tab]({row['link']})")
            with col3:
                # Create real clickable button that opens new tab using anchor wrapping a styled button
                link = row.get("link") or "#"
                safe_html = f"""
                <a href="{link}" target="_blank" rel="noopener">
                  <button style="background-color:#4CAF50;color:white;padding:8px 12px;border:none;border-radius:6px;cursor:pointer;">
                    Open Product Page
                  </button>
                </a>
                """
                st.markdown(safe_html, unsafe_allow_html=True)

                # price history chart button (uses embedding-aware matching)
                if st.button(f"Show Price History — {i}", key=f"hist_{i}"):
                    emb = row.get("embedding")
                    hist_df = fetch_price_history_for_result(row["title"], row["store"], current_embedding=emb)
                    if hist_df.empty:
                        st.info("No historical prices found for this product/store.")
                    else:
                        st.line_chart(data=hist_df.set_index("created_at")["price"])
                        st.write(f"Historical min: ₹{int(hist_df['price'].min())}, mean: ₹{int(hist_df['price'].mean())}")

        st.markdown("---")
        st.subheader("📦 All Results (ranked)")
        display_df = df[["title", "store", "price", "rating", "reviews", "score", "link"]].copy()
        st.dataframe(display_df)

# ---------------- Admin ----------------
st.markdown("---")
st.header("⚙️ Admin — DB & Embedding Info")
cur = db_conn.cursor()
cur.execute("SELECT COUNT(*) FROM searches")
search_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM results")
result_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM purchases")
purchase_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM alerts")
alert_count = cur.fetchone()[0]
st.write(f"Searches: **{search_count}** — Results: **{result_count}** — Purchases: **{purchase_count}** — Alerts: **{alert_count}**")
st.write(f"Embedding available: {EMBED_AVAILABLE}")

if st.button("Download DB (SQLite)"):
    try:
        with open(DB_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download DB file", data, file_name=Path(DB_PATH).name)
    except Exception as e:
        st.error(f"Cannot read DB: {e}")

if st.button("Download search CSV"):
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download search CSV", data, file_name=Path(CSV_PATH).name)
    else:
        st.info("No CSV yet.")
