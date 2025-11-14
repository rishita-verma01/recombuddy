# app.py
# India-only price compare — NO product links stored or shown anywhere.
# Features: SerpAPI (India), price cleaning, USD->INR conversion, embeddings (optional),
# heuristic/LightGBM ranking, SQLite logging (no link), CSV export (no link), alerts, history chart.
#
# Add SERPAPI_KEY to Streamlit Secrets before running.

import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import traceback

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
EMBED_THRESHOLD = 0.72  # cosine similarity threshold

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
    EMBED_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False
    EMBED_MODEL = None

# ---------------- CONFIG ----------------
DB_PATH = "price_compare.db"
CSV_PATH = "search_results.csv"      # saved results (no link)
ALERTS_CSV = "alerts.csv"
MIN_LABELS_TO_TRAIN = 10
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"

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
"Kurlon India", "Sleepwell India"
]
# -----------------------------------------

st.set_page_config(page_title="India Price Compare — No Links", layout="wide")
st.title("🇮🇳 Best Deal Finder — Links Removed (India-only)")

# Load SerpAPI Key
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# ---------------- SQLite helpers ----------------
def get_db_conn(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    return conn

db_conn = get_db_conn()

def table_has_column(conn, table_name, column_name):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = [r[1] for r in cur.fetchall()]
    return column_name in cols

def ensure_embedding_column(conn):
    if not table_has_column(conn, "results", "embedding"):
        try:
            cur = conn.cursor()
            cur.execute("ALTER TABLE results ADD COLUMN embedding TEXT")
            conn.commit()
            print("Added embedding column to results table.")
        except Exception as e:
            print("Could not add embedding column:", e)

# Initialize DB schema (no link column)
def init_db_schema(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        search_id INTEGER,
        title TEXT,
        store TEXT,
        price REAL,
        rating REAL,
        reviews INTEGER,
        image TEXT,
        score REAL,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id INTEGER,
        search_id INTEGER,
        bought_price REAL,
        created_at TEXT
    )
    """)
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
    )
    """)
    conn.commit()
    # ensure embedding column exists (migration)
    ensure_embedding_column(conn)

init_db_schema(db_conn)

# ---------------- utility functions ----------------
def safe_jsonify(obj):
    """Convert object to JSON-string safely (handles numpy arrays)."""
    if obj is None:
        return None
    try:
        if isinstance(obj, np.ndarray):
            return json.dumps(obj.tolist())
        if isinstance(obj, (list, tuple)):
            cleaned = []
            for v in obj:
                if isinstance(v, (np.generic,)):
                    cleaned.append(float(v))
                else:
                    cleaned.append(v)
            return json.dumps(cleaned)
        if isinstance(obj, (np.generic,)):
            return json.dumps(float(obj))
        # already JSON?
        if isinstance(obj, str):
            try:
                json.loads(obj)
                return obj
            except:
                return json.dumps(obj)
        return json.dumps(obj)
    except Exception:
        try:
            return json.dumps(str(obj))
        except:
            return None

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
    s_low = store_name.lower()
    for s in INDIAN_STORES:
        if s.lower() in s_low:
            return True
    if ".in" in s_low or "india" in s_low:
        return True
    return False

# Embedding helpers
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

# ---------------- DB logging (robust, no link) ----------------
def log_search(q):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO searches(query,created_at) VALUES(?,?)", (q, t))
    db_conn.commit()
    return cur.lastrowid

def log_results(search_id, rows):
    """
    Robust logging: detects presence of 'embedding' column and inserts accordingly.
    Embeddings are serialized. No 'link' field is inserted or stored.
    """
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    has_emb = table_has_column(db_conn, "results", "embedding")
    if has_emb:
        sql = """
        INSERT INTO results(search_id,title,store,price,rating,reviews,image,score,embedding,created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
    else:
        sql = """
        INSERT INTO results(search_id,title,store,price,rating,reviews,image,score,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """

    for r in rows:
        try:
            title = str(r.get("title") or "")[:2000]
            store = str(r.get("store") or "")[:500]
            try:
                price = float(r.get("price")) if r.get("price") is not None else None
            except:
                price = None
            try:
                rating = float(r.get("rating")) if r.get("rating") is not None else None
            except:
                rating = None
            try:
                reviews = int(r.get("reviews")) if r.get("reviews") not in (None, "") else None
            except:
                reviews = None
            image = str(r.get("image") or "")[:2000]
            try:
                score = float(r.get("score")) if r.get("score") is not None else None
            except:
                score = None

            if has_emb:
                emb = r.get("embedding", None)
                emb_json = safe_jsonify(emb)
                cur.execute(sql, (search_id, title, store, price, rating, reviews, image, score, emb_json, t))
            else:
                cur.execute(sql, (search_id, title, store, price, rating, reviews, image, score, t))
        except Exception:
            # log and continue
            print("=== log_results ERROR ===")
            print(traceback.format_exc())
            print("Row (sanitized):", {"title": title, "store": store, "price": price, "rating": rating, "reviews": reviews})
            continue
    db_conn.commit()

def log_purchase(result_row_id, search_id, bought_price):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO purchases(result_id,search_id,bought_price,created_at) VALUES(?,?,?,?)",
                (result_row_id, search_id, bought_price, t))
    db_conn.commit()

def append_to_csv(rows):
    # Remove any link fields from rows (defensive)
    safe_rows = []
    for r in rows:
        safe_rows.append({
            "title": r.get("title"),
            "store": r.get("store"),
            "price": r.get("price"),
            "rating": r.get("rating"),
            "reviews": r.get("reviews"),
            "image": r.get("image"),
            "score": r.get("score"),
            "embedding": r.get("embedding") if r.get("embedding") is not None else None
        })
    df = pd.DataFrame(safe_rows)
    file_exists = os.path.isfile(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)

def log_alert(search_query, title, store, ttype, tval, price):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    cur.execute("""
        INSERT INTO alerts(search_query,product_title,store,threshold_type,threshold_value,triggered_price,triggered_at)
        VALUES (?,?,?,?,?,?,?)
    """, (search_query, title, store, ttype, float(tval), float(price), t))
    db_conn.commit()
    df = pd.DataFrame([{
        "search_query": search_query,
        "product_title": title,
        "store": store,
        "threshold_type": ttype,
        "threshold_value": tval,
        "triggered_price": price,
        "triggered_at": t
    }])
    df.to_csv(ALERTS_CSV, mode="a", header=not os.path.exists(ALERTS_CSV), index=False)

# ---------------- SerpAPI fetch (no link) ----------------
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
    try:
        r = requests.get(url, params=params, timeout=20)
        data = r.json()
    except Exception:
        return []
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
        emb = compute_embedding(title) if EMBED_AVAILABLE else None
        out.append({
            "title": title,
            "store": store,
            "price": float(price),
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "image": item.get("thumbnail") or "",
            "embedding": emb
        })
    return out

# ---------------- matching & ranking ----------------
def title_matches(a, b):
    if not a or not b:
        return False
    a_tokens = set([t.lower() for t in a.split() if len(t) > 2])
    b_tokens = set([t.lower() for t in b.split() if len(t) > 2])
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens.intersection(b_tokens)
    return len(overlap) >= max(1, min(3, int(0.4 * min(len(a_tokens), len(b_tokens)))))

def fetch_price_history_for_result(title, store, current_embedding=None):
    cur = db_conn.cursor()
    cur.execute("SELECT title, store, price, created_at, embedding FROM results WHERE store LIKE ?", (f"%{store}%",))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["price", "created_at"])
    cols = ["title", "store", "price", "created_at", "embedding"]
    df = pd.DataFrame(rows, columns=cols)
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
            if title_matches(r["title"], title):
                matched = True
        if matched:
            matches.append((r["price"], r["created_at"]))
    if not matches:
        df2 = df.copy()
        df2["created_at"] = pd.to_datetime(df2["created_at"])
        df2 = df2.sort_values("created_at")
        return df2[["price", "created_at"]]
    dfm = pd.DataFrame(matches, columns=["price", "created_at"])
    dfm["created_at"] = pd.to_datetime(dfm["created_at"])
    dfm = dfm.sort_values("created_at")
    return dfm

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
    st.markdown("---")
    st.write(f"USD→INR rate: {usd_inr_rate:.2f} (live)")

st.info("Product links have been removed. Historical matching uses embeddings if available; otherwise token-overlap.")

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

        rows_to_log = []
        for _, r in df.iterrows():
            rows_to_log.append({
                "title": r["title"],
                "store": r["store"],
                "price": float(r["price"]),
                "rating": float(r["rating"]),
                "reviews": int(r["reviews"]),
                "image": r["image"],
                "score": float(r["score"]),
                "embedding": r.get("embedding")
            })

        sid = log_search(query)
        log_results(sid, rows_to_log)
        append_to_csv(rows_to_log)

        st.subheader("🏆 Top 3 Deals (India)")
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
            with col3:
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
        display_df = df[["title", "store", "price", "rating", "reviews", "score", "image"]].copy()
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

if st.button("Download alerts CSV"):
    if os.path.exists(ALERTS_CSV):
        with open(ALERTS_CSV, "rb") as f:
            data = f.read()
        st.download_button("Download alerts CSV", data, file_name=Path(ALERTS_CSV).name)
    else:
        st.info("No alerts triggered yet.")
