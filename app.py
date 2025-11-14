import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
import time
import re
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
CSV_PATH = "results_log.csv"
MIN_LABELS_TO_TRAIN = 10
MAX_SERP_RESULTS = 40
st.set_page_config(page_title="Best Deal Finder — Improved", layout="wide")
# ----------------------------

st.title("🛒 Best Deal Finder — Improved (Links + CSV + Better Filters)")

# Load secrets (Streamlit Cloud)
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY not found. Add it in Streamlit Secrets or set SERPAPI_KEY as an env var.")
    st.stop()

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
        search_id INTEGER,
        title TEXT,
        store TEXT,
        price REAL,
        rating REAL,
        reviews INTEGER,
        link TEXT,
        image TEXT,
        score REAL,
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
    conn.commit()
    return conn

db_conn = init_db(DB_PATH)

def log_search(query):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO searches (query, created_at) VALUES (?, ?)", (query, ts))
    db_conn.commit()
    return cur.lastrowid

def log_results_db(search_id, results):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    for r in results:
        cur.execute("""
            INSERT INTO results
            (search_id, title, store, price, rating, reviews, link, image, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (search_id, r.get("title"), r.get("store"), r.get("price"), r.get("rating") or 0,
              r.get("reviews") or 0, r.get("link"), r.get("image"), r.get("score") or 0, ts))
    db_conn.commit()

def log_purchase(result_row_id, search_id, bought_price):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO purchases (result_id, search_id, bought_price, created_at) VALUES (?, ?, ?, ?)",
                (result_row_id, search_id, bought_price, ts))
    db_conn.commit()

# ---------- CSV logging ----------
def append_results_csv(search_query, results, csv_path=CSV_PATH):
    """
    Append results to CSV. Columns: timestamp, query, title, store, price, rating, reviews, link, image, score
    """
    rows = []
    ts = datetime.utcnow().isoformat()
    for r in results:
        rows.append({
            "logged_at": ts,
            "query": search_query,
            "title": r.get("title"),
            "store": r.get("store"),
            "price": r.get("price"),
            "rating": r.get("rating"),
            "reviews": r.get("reviews"),
            "link": r.get("link"),
            "image": r.get("image"),
            "score": r.get("score")
        })
    df = pd.DataFrame(rows)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=header, encoding="utf-8")

# ---------- Utilities & robust price parsing ----------
def parse_price_to_float(price_raw):
    """
    Robust price parser: remove currency symbols, commas, and extract first reasonable number.
    Returns None if parsing fails.
    """
    if price_raw is None:
        return None
    s = str(price_raw)
    # replace common separators and currency words
    s = s.replace(",", "").replace("\u20b9", "").replace("Rs.", "").replace("₹", "").strip()
    # extract first number-looking chunk
    match = re.search(r"(\d+[\.]?\d*)", s)
    if not match:
        return None
    try:
        value = float(match.group(1))
        if value <= 0:
            return None
        return value
    except:
        return None

def tokenize(text):
    if not text:
        return []
    # lowercase, remove punctuation, split
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    tokens = [tok for tok in t.split() if len(tok) >= 3]
    return tokens

# ---------- Smart filtering ----------
# Keyword-based minimum price thresholds for categories likely to be expensive devices.
# These are conservative defaults and can be tweaked.
KEYWORD_MIN_PRICE = {
    "phone": 2000,
    "iphone": 5000,
    "samsung": 2000,
    "laptop": 5000,
    "macbook": 20000,
    "tv": 5000,
    "camera": 5000,
    "dslr": 10000,
    "playstation": 5000,
    "xbox": 5000,
    "airpod": 1000,
    "refrigerator": 8000
}

def estimate_min_price_from_query(query):
    q = query.lower()
    for k, v in KEYWORD_MIN_PRICE.items():
        if k in q:
            return v
    return 50  # default minimum

def filter_results_for_query(results, query):
    """
    Filters out results that are likely unrelated / accessory / erroneous low-priced items.
    Heuristics:
      - price must parse
      - title must have token overlap with query (at least 1 token of length >=3)
      - price must be >= estimated_min_price_by_query OR >= median_price*0.4
      - price must be > absolute_floor (like Rs 10)
    """
    parsed = []
    # parse prices and tokens
    for r in results:
        price = parse_price_to_float(r.get("price") or r.get("raw_price") or "")
        if price is None:
            continue
        r["price"] = price
        r["tokens_title"] = tokenize(r.get("title") or "")
        parsed.append(r)
    if not parsed:
        return []

    prices = [r["price"] for r in parsed]
    median_price = float(pd.Series(prices).median())
    estimated_min = estimate_min_price_from_query(query)
    absolute_floor = 10.0

    filtered = []
    q_tokens = set(tokenize(query))
    for r in parsed:
        # token overlap: require at least 1 token from query present in title OR store contains brand
        overlap = len(q_tokens.intersection(set(r["tokens_title"])))
        passes_overlap = overlap >= 1
        # price thresholds
        p = r["price"]
        passes_min_by_keyword = p >= estimated_min
        passes_relative = p >= (median_price * 0.4)
        passes_floor = p >= absolute_floor
        # Accept if either token overlap AND (passes_min_by_keyword or passes_relative)
        if passes_overlap and passes_floor and (passes_min_by_keyword or passes_relative):
            filtered.append(r)
        else:
            # If title strongly matches (contains full query phrase) allow
            title_lower = (r.get("title") or "").lower()
            if query.lower() in title_lower and passes_floor and p >= (median_price * 0.25):
                filtered.append(r)
            else:
                # skip
                continue
    # dedupe by (normalized title + store)
    seen = set()
    final = []
    for r in filtered:
        key = (str(r.get("title","")).strip().lower()[:140], str(r.get("store","")).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        final.append(r)
    return final

# ---------- SerpAPI fetch ----------
def fetch_serpapi(query, serpapi_key, max_results=MAX_SERP_RESULTS):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": serpapi_key,
        "num": max_results
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    out = []
    for item in data.get("shopping_results", []):
        # serpapi fields: price, title, link, source, thumbnail, rating, reviews
        raw_price = item.get("price") or item.get("raw_price") or ""
        title = item.get("title") or item.get("product_title") or ""
        link = item.get("link") or item.get("product_link") or ""
        store = item.get("source") or item.get("store") or item.get("merchant") or ""
        rating = float(item.get("rating") or 0)
        reviews = int(item.get("reviews") or 0)
        thumb = item.get("thumbnail") or ""
        out.append({
            "title": title,
            "store": store,
            "raw_price": raw_price,
            "link": link,
            "rating": rating,
            "reviews": reviews,
            "image": thumb
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
    params = {"objective":"binary", "metric":"binary_logloss", "verbosity": -1}
    model = lgb.train(params, lgb_train, num_boost_round=100)
    X_new = feature_df[feature_cols].fillna(0)
    preds = model.predict(X_new)
    return preds, model

def get_ranked_results(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    features = df[["price","rating","reviews"]].fillna(0)
    ml = train_lgb_and_score(features, db_conn)
    if ml:
        preds, model = ml
        df["score"] = preds
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        return df
    df = heuristic_rank(df)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return df

# ---------- UI ----------
st.markdown("Type a product name and press Enter. The app uses SerpAPI (Google Shopping). We now filter out obvious accessory results and save everything to CSV. Use the blue button to open product pages.")

query = st.text_input("Enter product name (e.g., iPhone 13, JBL earphones, laptop)")

if query:
    with st.spinner("Fetching results from SerpAPI..."):
        raw_results = fetch_serpapi(query, SERPAPI_KEY, MAX_SERP_RESULTS)

    # apply filtering
    filtered = filter_results_for_query(raw_results, query)
    if not filtered:
        st.warning("No relevant results found after filtering. Try adding brand or model number (e.g., 'iPhone 14 128GB').")
    else:
        # rank
        ranked_df = get_ranked_results(filtered)

        # Prepare results for logging and CSV
        search_id = log_search(query)
        logged = []
        for _, row in ranked_df.iterrows():
            rec = {
                "title": row.get("title"),
                "store": row.get("store"),
                "price": float(row.get("price") or 0),
                "rating": float(row.get("rating") or 0),
                "reviews": int(row.get("reviews") or 0),
                "link": row.get("link") or "",
                "image": row.get("image") or "",
                "score": float(row.get("score") or 0)
            }
            logged.append(rec)

        # DB + CSV logging
        log_results_db(search_id, logged)
        append_results_csv(query, logged, CSV_PATH)

        st.subheader("🏆 Top 3 Deals")
        top3 = ranked_df.head(3)
        for idx, row in top3.reset_index(drop=True).iterrows():
            cols = st.columns([1,5,1])
            with cols[0]:
                if row.get("image"):
                    st.image(row.get("image"), width=120)
            with cols[1]:
                st.markdown(f"**{row.get('title')}**")
                st.write(f"Store: **{row.get('store')}** — Price: **₹{row.get('price'):,}**")
                if row.get("rating"):
                    st.write(f"Rating: {row.get('rating')} ⭐ ({row.get('reviews')} reviews)")
            with cols[2]:
                # Button that opens link in new tab using an HTML anchor styled as a button
                link = row.get("link") or "#"
                safe_link = html.escape(str(link), quote=True)
                button_html = f"""
                <a href="{safe_link}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
                    <div style="
                        display:inline-block;
                        background-color:#1f77b4;
                        color:white;
                        padding:8px 12px;
                        border-radius:6px;
                        font-weight:600;
                        text-align:center;
                        ">
                        Open Product
                    </div>
                </a>
                """
                st.components.v1.html(button_html, height=50)

        st.markdown("---")
        st.subheader("📦 All Results (ranked)")
        display_df = ranked_df[["title","store","price","rating","reviews","score","link"]].reset_index(drop=True)
        st.dataframe(display_df)

        st.markdown("---")
        st.info("Use the 'Open Product' buttons beside Top 3 to open the store page. The CSV is appended with every search and can be downloaded below.")

        # Show table with small open buttons per row
        st.markdown("### Open / Mark Bought")
        for i, row in display_df.iterrows():
            cols = st.columns([6,1,1,1])
            with cols[0]:
                st.write(f"**{row['title'][:200]}** — {row['store']} — ₹{row['price']:,}")
            with cols[1]:
                # small open button
                link = row['link'] or "#"
                safe_link = html.escape(str(link), quote=True)
                small_btn = f"""
                <a href="{safe_link}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
                    <div style="
                        display:inline-block;
                        background-color:#2b8a3e;
                        color:white;
                        padding:6px 10px;
                        border-radius:6px;
                        font-weight:600;
                        text-align:center;
                        font-size:13px;
                        ">
                        Open
                    </div>
                </a>
                """
                st.components.v1.html(small_btn, height=38)
            with cols[2]:
                if st.button(f"Bought ✓ (row {i})", key=f"buy_row_{i}_{time.time()}"):
                    # find result_id in DB (recent)
                    cur = db_conn.cursor()
                    cur.execute("SELECT id FROM results WHERE search_id = ? AND title = ? AND price = ? ORDER BY id DESC LIMIT 1",
                                (search_id, row['title'], float(row['price'] or 0)))
                    res = cur.fetchone()
                    if res:
                        log_purchase(res[0], search_id, float(row['price'] or 0))
                        st.success("Purchase recorded (for model training).")
                    else:
                        st.error("Could not find matching result to record purchase.")
            with cols[3]:
                if row['link']:
                    st.markdown(f"[Open link]({row['link']})")

# ---------- Admin / Model ----------
st.markdown("---")
st.header("⚙️ Admin & Data")
cur = db_conn.cursor()
cur.execute("SELECT COUNT(*) FROM searches")
search_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM results")
result_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM purchases")
purchase_count = cur.fetchone()[0]
st.write(f"Searches: **{search_count}** — Results stored: **{result_count}** — Purchases labeled: **{purchase_count}**")

if st.button("Download CSV results"):
    try:
        with open(CSV_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download results_log.csv", data, file_name=Path(CSV_PATH).name)
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")

if st.button("Download SQLite DB"):
    try:
        with open(DB_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download DB file", data, file_name=Path(DB_PATH).name)
    except Exception as e:
        st.error(f"Cannot read DB: {e}")

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

st.markdown("Notes: Filtering heuristics try to remove accessory/irrelevant low-price results. If you see a good product filtered out, tweak KEYWORD_MIN_PRICE or the relative threshold (median*0.4) in the code.")
