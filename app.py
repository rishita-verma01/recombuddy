# app.py
# Streamlit Price Compare (SerpAPI) + SQLite + optional LightGBM ranking
# Put this file in a GitHub repo and deploy on share.streamlit.io
#
# Requirements: see requirements.txt below

import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from pathlib import Path

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# ---------- CONFIG ----------
DB_PATH = "price_compare.db"
MIN_LABELS_TO_TRAIN = 10
st.set_page_config(page_title="Best Deal Finder (SerpAPI)", layout="wide")
# ----------------------------

st.title("🛒 Best Deal Finder — SerpAPI + Streamlit (Free)")

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

def log_results(search_id, results):
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

# ---------- SerpAPI fetch ----------
def fetch_serpapi(query, serpapi_key, max_results=30):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": serpapi_key,
        "num": max_results
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    out = []
    for item in data.get("shopping_results", []):
        price = item.get("price", "")
        try:
            price_val = float(str(price).replace("₹", "").replace(",", "").replace("Rs.", "").strip())
        except Exception:
            # try other currencies or skip
            try:
                price_val = float(''.join(ch for ch in str(price) if (ch.isdigit() or ch=='.')))
            except Exception:
                continue
        out.append({
            "title": item.get("title"),
            "store": item.get("source") or item.get("store") or "",
            "price": price_val,
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "link": item.get("link") or "",
            "image": item.get("thumbnail") or ""
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
    # Build training data from DB: results joined with purchases (simple approach)
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
    # Predict on feature_df
    X_new = feature_df[feature_cols].fillna(0)
    preds = model.predict(X_new)
    return preds, model

def get_ranked_results(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    # Try ML
    features = df[["price", "rating", "reviews"]].fillna(0)
    ml = train_lgb_and_score(features, db_conn)
    if ml:
        preds, model = ml
        df["score"] = preds
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        return df
    # fallback heuristic
    df = heuristic_rank(df)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return df

# ---------- UI ----------
st.markdown("Type a product name and press Enter. Results come from SerpAPI (Google Shopping). Mark 'Bought' to label purchases for future training.")

query = st.text_input("Enter product name (e.g. iPhone 13, Bluetooth speaker)")

if query:
    with st.spinner("Fetching prices from SerpAPI..."):
        serp_results = fetch_serpapi(query, SERPAPI_KEY)

    # dedupe by title+store (light)
    combined = serp_results
    seen = set()
    unique = []
    for r in combined:
        key = (str(r.get("title",""))[:120].strip().lower(), str(r.get("store","")).strip().lower())
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
            st.write(f"Store: **{row.get('store')}** — Price: **₹{row.get('price'):,}**")
            if row.get("rating"):
                st.write(f"Rating: {row.get('rating')} ⭐ ({row.get('reviews')} reviews)")
            if row.get("link"):
                st.markdown(f"[View deal]({row.get('link')})")
        with cols[2]:
            if st.button(f"Bought — #{idx+1}", key=f"buy_top_{idx}_{time.time()}"):
                # find result_id
                cur = db_conn.cursor()
                cur.execute("SELECT id FROM results WHERE search_id = ? AND title = ? AND price = ? ORDER BY id DESC LIMIT 1",
                            (search_id, row.get("title"), float(row.get("price") or 0)))
                res = cur.fetchone()
                if res:
                    log_purchase(res[0], search_id, float(row.get("price") or 0))
                    st.success("Purchase recorded — used for future ML training.")
                else:
                    st.error("Could not record purchase (not found).")

    st.markdown("---")
    st.subheader("📦 All Results (ranked)")
    display_df = ranked_df[["title","store","price","rating","reviews","score","link"]].reset_index(drop=True)
    st.dataframe(display_df)

    st.markdown("---")
    st.info("Click 'Buy' next to any row below to mark it as purchased and create labels for training.")

    for i, row in display_df.iterrows():
        cols = st.columns([4,1,1,1,1])
        with cols[0]:
            st.write(f"**{row['title'][:200]}** — {row['store']} — ₹{row['price']:,}")
        with cols[1]:
            st.write(f"{row['rating']} ⭐")
        with cols[2]:
            st.write(f"{row['reviews']} rev")
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
                st.markdown(f"[Open]({row['link']})")

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

st.markdown("Notes: The app uses a heuristic score until you have enough purchase labels to train LightGBM. LightGBM is optional — heuristic works well for price-based ranking.")
