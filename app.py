import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from pathlib import Path
import html

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# ---------- CONFIG ----------
DB_PATH = "price_compare.db"
MIN_LABELS_TO_TRAIN = 10
st.set_page_config(page_title="Best Deal Finder (SerpAPI) — Links", layout="wide")
# ----------------------------

st.title("🛒 Best Deal Finder — Product Links Included (Button style)")

# Load SerpAPI key: prefer Streamlit secrets, then env var, otherwise placeholder
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "") or "YOUR_SERPAPI_KEY"  # placeholder key

if SERPAPI_KEY == "YOUR_SERPAPI_KEY":
    st.warning("Using placeholder SerpAPI key. Add your real key in Streamlit Secrets (SERPAPI_KEY) for production.")

# ---------- DB helpers ----------
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
    search_id TEXT,
    source TEXT,
    title TEXT,
    price REAL,
    mrp REAL,
    discount REAL,
    rating REAL,
    reviews INTEGER,
    link TEXT,
    image TEXT,
    score REAL,
    timestamp TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS purchases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id INTEGER,
        search_id INTEGER,
        bought_price REAL,
        created_at TEXT
    )""")
    conn.commit()
    return conn

db_conn = init_db(DB_PATH)

def log_search(query):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO searches (query, created_at) VALUES (?, ?)", (query, ts))
    db_conn.commit()
    return cur.lastrowid

def log_results(search_id, results):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    for r in results:
        cur.execute("""
            INSERT INTO results
            (search_id, title, store, price, mrp, discount, rating, reviews, link, image, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (search_id, r.get("title"), r.get("store"), r.get("price"), r.get("mrp"),
              r.get("discount"), r.get("rating") or 0, r.get("reviews") or 0,
              r.get("link"), r.get("image"), r.get("score") or 0, ts))
    db_conn.commit()

def log_purchase(result_row_id, search_id, bought_price):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO purchases (result_id, search_id, bought_price, created_at) VALUES (?, ?, ?, ?)",
                (result_row_id, search_id, bought_price, ts))
    db_conn.commit()

# ---------- SerpAPI fetch (extract link + possible MRP) ----------
def extract_price_value(raw):
    """Try to parse numeric price from various SerpAPI formats"""
    if raw is None:
        return None
    s = str(raw)
    # Remove common currency chars
    cleaned = s.replace("₹", "").replace("Rs.", "").replace(",", "").strip()
    # keep digits and dot
    num = ''.join(ch for ch in cleaned if (ch.isdigit() or ch == '.'))
    try:
        return float(num) if num else None
    except:
        return None

def fetch_serpapi(query, serpapi_key, max_results=30):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": serpapi_key,
        "num": max_results
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
    except Exception as e:
        st.error(f"SerpAPI request failed: {e}")
        return []

    out = []
    for item in data.get("shopping_results", []):
        # Common fields
        title = item.get("title") or item.get("product_title") or ""
        store = item.get("source") or item.get("store") or ""
        link = item.get("link") or item.get("product_link") or ""
        image = item.get("thumbnail") or item.get("image") or ""
        rating = 0
        reviews = 0
        try:
            rating = float(item.get("rating") or 0)
        except:
            rating = 0
        try:
            reviews = int(item.get("reviews") or 0)
        except:
            reviews = 0

        # Price and possible MRP/original price
        price_val = extract_price_value(item.get("price") or item.get("extracted_price") or item.get("price_with_currency"))
        # try some other possible keys
        mrp_val = None
        # SerpAPI sometimes returns 'original_price' or 'retail_price' or 'prices'
        for k in ("original_price", "retail_price", "mrp", "original_price_with_currency"):
            if k in item:
                mrp_val = extract_price_value(item.get(k))
                if mrp_val:
                    break
        # fallback: some items have 'savings' or 'discount' but not original price
        # compute discount if both found
        discount = None
        if price_val and mrp_val and mrp_val > 0:
            try:
                discount = round((mrp_val - price_val) / mrp_val * 100, 2)
            except:
                discount = None

        out.append({
            "title": title,
            "store": store,
            "price": price_val or 0.0,
            "mrp": mrp_val,
            "discount": discount,
            "rating": rating,
            "reviews": reviews,
            "link": link,
            "image": image
        })
    return out

# ---------- Ranking ----------
def heuristic_rank(df):
    if df.empty:
        return df
    df = df.copy()
    df["rating_norm"] = df["rating"].fillna(0) / 5.0
    df["reviews_norm"] = df["reviews"].fillna(0) / (df["reviews"].max() + 1e-9)
    max_price = df["price"].max() if df["price"].notnull().any() else 1.0
    df["rel_price"] = (max_price - df["price"]) / (max_price + 1e-9)
    df["score"] = df["rel_price"] * 0.65 + df["rating_norm"] * 0.25 + df["reviews_norm"] * 0.1
    return df

def train_lgb_and_score(feature_df, db_conn_local):
    q = """
    SELECT r.price, r.rating, r.reviews, CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END as bought
    FROM results r
    LEFT JOIN purchases p ON p.result_id = r.id
    """
    df_train = pd.read_sql_query(q, db_conn_local)
    if df_train.shape[0] < MIN_LABELS_TO_TRAIN or not LGB_AVAILABLE:
        return None
    df_train = df_train.fillna(0)
    feature_cols = ["price", "rating", "reviews"]
    X = df_train[feature_cols]
    y = df_train["bought"]
    lgb_train = lgb.Dataset(X, label=y)
    params = {"objective":"binary","metric":"binary_logloss","verbosity": -1}
    model = lgb.train(params, lgb_train, num_boost_round=100)
    X_new = feature_df[feature_cols].fillna(0)
    preds = model.predict(X_new)
    return preds, model

def get_ranked_results(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    features = df[["price", "rating", "reviews"]].fillna(0)
    ml = train_lgb_and_score(features, db_conn)
    if ml:
        preds, model = ml
        df["score"] = preds
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        return df
    df = heuristic_rank(df)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return df

# ---------- HTML helper for button-style links ----------
def html_open_button(url, label="Open", width_px=150):
    # safe-escape URL and label
    url_esc = html.escape(url, quote=True)
    label_esc = html.escape(label)
    html_button = f"""
    <button style="
        background-color:#0f62fe;
        color:white;
        border-radius:6px;
        padding:8px 12px;
        border:none;
        cursor:pointer;
        font-weight:600;
        ">
      <a href="{url_esc}" target="_blank" style="color:inherit; text-decoration:none;">{label_esc}</a>
    </button>
    """
    return html_button

# ---------- UI ----------
st.markdown("Type a product name and press Enter. Each result shows a button to open the product page (opens in a new tab). Mark 'Bought' to label purchases for future ML training.")

query = st.text_input("Enter product name (e.g. iPhone 13, Bluetooth speaker)")

if query:
    with st.spinner("Fetching prices from SerpAPI..."):
        serp_results = fetch_serpapi(query, SERPAPI_KEY)

    # dedupe by title+store lightly
    seen = set()
    unique = []
    for r in serp_results:
        key = (str(r.get("title",""))[:140].strip().lower(), str(r.get("store","")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    ranked_df = get_ranked_results(unique)

    # Log search + results
    search_id = log_search(query)
    logged = []
    for _, row in ranked_df.iterrows():
        rec = {
            "title": row.get("title"),
            "store": row.get("store"),
            "price": float(row.get("price") or 0),
            "mrp": float(row.get("mrp")) if row.get("mrp") else None,
            "discount": float(row.get("discount")) if row.get("discount") else None,
            "rating": float(row.get("rating") or 0),
            "reviews": int(row.get("reviews") or 0),
            "link": row.get("link") or "",
            "image": row.get("image") or "",
            "score": float(row.get("score") or 0)
        }
        logged.append(rec)
    log_results(search_id, logged)

    st.subheader("🏆 Top 3 Deals")
    top3 = ranked_df.head(3)
    for idx, row in top3.reset_index(drop=True).iterrows():
        cols = st.columns([1,4,1])
        with cols[0]:
            if row.get("image"):
                st.image(row.get("image"), width=120)
        with cols[1]:
            st.markdown(f"**{row.get('title')}**")
            price_text = f"₹{int(row.get('price')):,}" if row.get('price') else "N/A"
            st.write(f"Store: **{row.get('store') or 'Unknown'}** — Price: **{price_text}**")
            if row.get("mrp"):
                st.write(f"MRP: ₹{int(row.get('mrp')):,} — Discount: {row.get('discount')}%")
            if row.get("rating"):
                st.write(f"Rating: {row.get('rating')} ⭐ ({row.get('reviews')} reviews)")
        with cols[2]:
            link = row.get("link") or ""
            if link:
                html_btn = html_open_button(link, label=f"Open on {row.get('store') or 'Store'}")
                st.markdown(html_btn, unsafe_allow_html=True)
            # allow marking bought
            if st.button(f"Bought — #{idx+1}", key=f"buy_top_{idx}_{time.time()}"):
                cur = db_conn.cursor()
                cur.execute("SELECT id FROM results WHERE search_id = ? AND title = ? AND price = ? ORDER BY id DESC LIMIT 1",
                            (search_id, row.get("title"), float(row.get("price") or 0)))
                res = cur.fetchone()
                if res:
                    log_purchase(res[0], search_id, float(row.get("price") or 0))
                    st.success("Purchase recorded — will be used for future ML training.")
                else:
                    st.error("Could not record purchase (result not found).")

    st.markdown("---")
    st.subheader("📦 All Results (ranked)")
    display_df = ranked_df[["title","store","price","mrp","discount","rating","reviews","score","link"]].reset_index(drop=True)
    st.dataframe(display_df)

    st.markdown("---")
    st.info("Use the 'Open' buttons next to each top result, or mark any row below as 'Bought' to create a label for training.")

    # show rows with buttons
    for i, row in display_df.iterrows():
        cols = st.columns([4,1,1,1,2])
        with cols[0]:
            st.write(f"**{row['title'][:180]}** — {row['store']} — ₹{int(row['price']) if row['price'] else 'N/A'}")
            if row['mrp']:
                st.write(f"MRP ₹{int(row['mrp'])} — {row['discount']}% off")
        with cols[1]:
            st.write(f"{row['rating']} ⭐")
        with cols[2]:
            st.write(f"{int(row['reviews'])} rev")
        with cols[3]:
            if st.button(f"Buy ✓ (row {i})", key=f"buy_row_{i}_{time.time()}"):
                cur = db_conn.cursor()
                cur.execute("SELECT id FROM results WHERE search_id = ? AND title = ? AND price = ? ORDER BY id DESC LIMIT 1",
                            (search_id, row['title'], float(row['price'] or 0)))
                res = cur.fetchone()
                if res:
                    log_purchase(res[0], search_id, float(row['price'] or 0))
                    st.success("Purchase recorded.")
                else:
                    st.error("Could not find matching result to record purchase.")
        with cols[4]:
            if row['link']:
                html_btn = html_open_button(row['link'], label="Open Product Page")
                st.markdown(html_btn, unsafe_allow_html=True)

# ---------- Admin / Train ----------
st.markdown("---")
st.header("⚙️ Admin & Model")

cur = db_conn.cursor()
cur.execute("SELECT COUNT(*) FROM searches")
search_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM results")
result_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM purchases")
purchase_count = cur.fetchone()[0]
st.write(f"Searches: **{search_count}** — Results stored: **{result_count}** — Purchases labeled: **{purchase_count}**")

if st.button("Force-train LightGBM (if enough labels)"):
    if not LGB_AVAILABLE:
        st.error("LightGBM not installed. Install via requirements to enable training.")
    else:
        q = """
        SELECT r.price, r.rating, r.reviews,
               CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END as bought
        FROM results r
        LEFT JOIN purchases p ON p.result_id = r.id
        """
        df_train = pd.read_sql_query(q, db_conn)
        if df_train.shape[0] < MIN_LABELS_TO_TRAIN:
            st.error(f"Not enough labeled rows to train. Need at least {MIN_LABELS_TO_TRAIN}.")
        else:
            df_train = df_train.fillna(0)
            feature_cols = ["price","rating","reviews"]
            X = df_train[feature_cols]
            y = df_train["bought"]
            lgb_train = lgb.Dataset(X, label=y)
            params = {"objective":"binary","metric":"binary_logloss","verbosity": -1}
            model = lgb.train(params, lgb_train, num_boost_round=150)
            model.save_model("lgb_model.txt")
            st.success("LightGBM model trained and saved as lgb_model.txt")

if st.button("Download SQLite DB"):
    try:
        with open(DB_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download DB file", data, file_name=Path(DB_PATH).name)
    except Exception as e:
        st.error(f"Cannot read DB: {e}")

st.markdown("Notes: Buttons open product pages in a new tab. The app will use a heuristic ranking until you gather enough labeled purchases to train a LightGBM model.")
