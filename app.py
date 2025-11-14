# app.py
# Streamlit Price Compare (SerpAPI) + robust price parsing + INR conversion + CSV save + SQLite + optional LightGBM
# Deploy on share.streamlit.io (Streamlit Cloud)
#
# Requirements: streamlit, pandas, requests, lightgbm (optional)
# Add your SERPAPI_KEY to Streamlit Secrets as before.

import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from pathlib import Path
import re
import math

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# ---------- CONFIG ----------
DB_PATH = "price_compare.db"
MIN_LABELS_TO_TRAIN = 10
CSV_FOLDER = "csv_results"
os.makedirs(CSV_FOLDER, exist_ok=True)
st.set_page_config(page_title="Best Deal Finder (SerpAPI) — INR-aware", layout="wide")
# ----------------------------

st.title("🛒 Best Deal Finder — SerpAPI + INR Conversion + CSV + Buttons")

# Secrets
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY not set. Add it to Streamlit Secrets or as an env var.")
    st.stop()

# ---------- DB ----------
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
        price_original TEXT,
        currency TEXT,
        price_in_inr REAL,
        rating REAL,
        reviews INTEGER,
        link TEXT,
        image TEXT,
        score REAL,
        flags TEXT,
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
            (search_id, title, store, price_original, currency, price_in_inr, rating, reviews, link, image, score, flags, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (search_id, r.get("title"), r.get("store"), r.get("price_raw"), r.get("currency"),
              r.get("price_in_inr"), r.get("rating") or 0, r.get("reviews") or 0,
              r.get("link"), r.get("image"), r.get("score") or 0, r.get("flags") or "", ts))
    db_conn.commit()

def log_purchase(result_row_id, search_id, bought_price):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO purchases (result_id, search_id, bought_price, created_at) VALUES (?, ?, ?, ?)",
                (result_row_id, search_id, bought_price, ts))
    db_conn.commit()

# ---------- Utility: fetch live USD->INR rate (free) ----------
def fetch_usd_to_inr():
    """
    Uses exchangerate.host free API to get latest USD->INR rate.
    Falls back to a conservative constant if API fails.
    """
    try:
        r = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=INR", timeout=6)
        j = r.json()
        rate = float(j.get("rates", {}).get("INR"))
        if rate and rate > 0:
            return rate
    except Exception:
        pass
    # fallback safe default (will be updated if you refresh)
    return 88.7

# ---------- SerpAPI fetch & robust price parsing ----------
CURRENCY_SYMBOLS = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "rs": "INR",
    "inr": "INR", "usd": "USD", "$": "USD", "us$": "USD", "€": "EUR",
    "aud$": "AUD", "sgd": "SGD"
}

def detect_currency_and_amount(price_str):
    """
    Try to detect currency and numeric amount from various SerpAPI price strings.
    Returns (amount_float, currency_code, raw_string)
    """
    if price_str is None:
        return None, None, ""
    s = str(price_str).strip()
    # normalize spaces
    s_norm = re.sub(r"\s+", " ", s.lower())
    # detect common currency symbols first
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in s_norm:
            # remove non numeric except . and ,
            num = re.sub(r"[^\d.,]", "", s_norm)
            # replace comma thousands, keep decimal dot
            num = num.replace(",", "")
            try:
                val = float(num)
                return val, code, s
            except Exception:
                pass
    # try to find standalone $ or ₹
    m = re.search(r"([\$\₹])\s*([0-9\.,]+)", s)
    if m:
        sym = m.group(1)
        amt = m.group(2).replace(",", "")
        code = "USD" if sym == "$" else "INR"
        try:
            return float(amt), code, s
        except:
            pass
    # fallback: extract digits
    m2 = re.search(r"([0-9\.,]{2,})", s)
    if m2:
        num = m2.group(1).replace(",", "")
        try:
            return float(num), None, s
        except:
            pass
    return None, None, s

def normalize_price_to_inr(price_raw):
    """
    Input: raw price string or numeric
    Output: (price_in_inr: float, currency_code: str, price_original_str)
    """
    # If already numeric and we assume INR (common)
    if price_raw is None:
        return None, None, ""
    # if price_raw is numeric
    if isinstance(price_raw, (int, float)):
        return float(price_raw), "INR", str(price_raw)
    amt, code, raw = detect_currency_and_amount(price_raw)
    if amt is None:
        return None, None, raw
    # If USD convert
    if code == "USD":
        rate = fetch_usd_to_inr()
        return round(amt * rate, 2), "USD", raw
    # If INR or unknown treat as INR
    if code == "INR" or code is None:
        return round(amt, 2), "INR", raw
    # other currencies: convert via exchangerate.host
    try:
        r = requests.get(f"https://api.exchangerate.host/convert?from={code}&to=INR&amount={amt}", timeout=6)
        jr = r.json()
        res = float(jr.get("result"))
        return round(res, 2), code, raw
    except Exception:
        # fallback: return numeric as-is
        return round(amt, 2), code, raw

# ---------- Fetch from SerpAPI ----------
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
        # SerpAPI sometimes has price in different keys; prefer 'price' but keep raw
        price_raw = item.get("price") or item.get("extracted_price") or item.get("product_price") or item.get("offers", {}).get("price", "")
        price_in_inr, currency_code, price_orig = normalize_price_to_inr(price_raw)
        out.append({
            "title": item.get("title"),
            "store": item.get("source") or item.get("store") or "",
            "price_raw": price_orig,
            "currency": currency_code,
            "price_in_inr": price_in_inr,
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
    # if price_in_inr missing, drop those rows for ranking
    df = df[df["price_in_inr"].notnull()].copy()
    if df.empty:
        return df
    df["rating_norm"] = df["rating"].fillna(0) / 5.0
    df["reviews_norm"] = df["reviews"].fillna(0) / (df["reviews"].max() + 1e-9)
    max_price = df["price_in_inr"].max() if df["price_in_inr"].notnull().any() else 1.0
    df["rel_price"] = (max_price - df["price_in_inr"]) / (max_price + 1e-9)
    df["score"] = df["rel_price"] * 0.65 + df["rating_norm"] * 0.25 + df["reviews_norm"] * 0.1
    return df

def train_lgb_and_score(feature_df, db_conn_local):
    q = """
    SELECT r.price_in_inr as price, r.rating, r.reviews, CASE WHEN p.id IS NOT NULL THEN 1 ELSE 0 END as bought
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
    X_new = feature_df[["price_in_inr", "rating", "reviews"]].rename(columns={"price_in_inr":"price"})
    X_new = X_new.fillna(0)
    preds = model.predict(X_new)
    return preds, model

def get_ranked_results(results):
    df = pd.DataFrame(results)
    if df.empty:
        return df
    # remove rows with no price converted
    df = df[df["price_in_inr"].notnull()].copy()
    if df.empty:
        return df
    ml = train_lgb_and_score(df, db_conn)
    if ml:
        preds, model = ml
        df["score"] = preds
        df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
        return df
    df = heuristic_rank(df)
    df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    return df

# ---------- Heuristic flag for improbable prices ----------
PHONE_KEYWORDS = ["iphone", "galaxy", "pixel", "oneplus", "redmi", "moto", "microsoft surface", "ipad", "ipad mini", "ipad pro", "phone", "smartphone"]

def flag_improbable(row):
    flags = []
    title = (row.get("title") or "").lower()
    price = row.get("price_in_inr")
    if price is None or math.isnan(price):
        flags.append("price-missing")
        return ";".join(flags)
    # if title contains phone brand and price less than threshold -> likely accessory
    for k in PHONE_KEYWORDS:
        if k in title:
            if price < 5000:  # phone priced under 5k likely accessory or wrong
                flags.append("likely-accessory-or-wrong-price")
            break
    # if price very small (<50 INR) -> suspicious
    if price < 50:
        flags.append("suspiciously-low-price")
    return ";".join(flags)

# ---------- CSV writer ----------
def save_results_to_csv(search_id, query, ranked_df):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{CSV_FOLDER}/results_{ts}.csv"
    out_rows = []
    for _, r in ranked_df.iterrows():
        out_rows.append({
            "search_id": search_id,
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "title": r.get("title"),
            "store": r.get("store"),
            "price_original": r.get("price_raw"),
            "currency": r.get("currency"),
            "price_in_inr": r.get("price_in_inr"),
            "rating": r.get("rating"),
            "reviews": r.get("reviews"),
            "link": r.get("link"),
            "image": r.get("image"),
            "score": r.get("score"),
            "flags": r.get("flags")
        })
    df_out = pd.DataFrame(out_rows)
    # append if file exists else write
    df_out.to_csv(filename, index=False)
    return filename

# ---------- UI ----------
st.markdown("Type a product name and press Enter. Results come from SerpAPI (Google Shopping). Prices in other currencies will be converted to INR automatically. Product links are 'Open' buttons.")

query = st.text_input("Enter product name (e.g. iPhone 13, Bluetooth speaker)")

# Show current USD->INR rate fetched live (helpful for transparency)
with st.spinner("Fetching live USD → INR rate..."):
    usd2inr = fetch_usd_to_inr()
st.caption(f"Live USD → INR used for conversion: 1 USD = {usd2inr:.4f} INR. (Live source: exchangerate.host; example market rates: Wise/XE show ~88.7 INR).")

if query:
    with st.spinner("Fetching prices from SerpAPI..."):
        serp_results = fetch_serpapi(query, SERPAPI_KEY)

    # dedupe by title+store lightly
    seen = set()
    unique = []
    for r in serp_results:
        key = (str(r.get("title",""))[:120].strip().lower(), str(r.get("store","")).strip().lower(), str(r.get("price_raw","")).strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    # Add flags
    for r in unique:
        r["flags"] = flag_improbable(r)

    ranked_df = get_ranked_results(unique)

    # Fill missing scores with 0 if needed
    if "score" not in ranked_df.columns:
        ranked_df["score"] = 0.0

    # Log search + results
    search_id = log_search(query)
    log_results(search_id, ranked_df.to_dict('records'))

    # Save CSV and offer download
    csv_file = save_results_to_csv(search_id, query, ranked_df)
    with open(csv_file, "rb") as f:
        st.download_button("📥 Download CSV for this search", f, file_name=os.path.basename(csv_file))

    st.subheader("🏆 Top 3 Deals")
    top3 = ranked_df.head(3)
    for idx, row in top3.reset_index(drop=True).iterrows():
        cols = st.columns([1,5,2])
        with cols[0]:
            if row.get("image"):
                st.image(row.get("image"), width=120)
        with cols[1]:
            st.markdown(f"### {row.get('title')}")
            st.write(f"**Store:** {row.get('store')}  •  **Price (INR):** ₹{row.get('price_in_inr'):,}" if row.get("price_in_inr") is not None else "Price: N/A")
            st.write(f"**Original:** {row.get('price_raw')} ({row.get('currency') or 'unknown'})")
            if row.get("rating"):
                st.write(f"⭐ {row.get('rating')} ({row.get('reviews')} reviews)")
            if row.get("flags"):
                st.warning(f"Flag: {row.get('flags')}")
        with cols[2]:
            # Option A: button-style open link (HTML button)
            link = row.get("link") or "#"
            if link and link != "#":
                st.markdown(f"""<a href="{link}" target="_blank" rel="noopener noreferrer"><button style="padding:10px 14px;border-radius:6px;background-color:#1f77b4;color:white;border:none;">Open</button></a>""", unsafe_allow_html=True)
            # a buy button to label purchases for ML
            if st.button(f"Bought — #{idx+1}", key=f"buy_top_{idx}_{time.time()}"):
                cur = db_conn.cursor()
                cur.execute("SELECT id FROM results WHERE search_id = ? AND title = ? AND price_in_inr = ? ORDER BY id DESC LIMIT 1",
                            (search_id, row.get("title"), float(row.get("price_in_inr") or 0)))
                res = cur.fetchone()
                if res:
                    log_purchase(res[0], search_id, float(row.get("price_in_inr") or 0))
                    st.success("Purchase recorded — used for future ML training.")
                else:
                    st.error("Could not record purchase (not found).")

    st.markdown("---")
    st.subheader("📦 All Results (ranked)")
    display_df = ranked_df[["title","store","price_raw","currency","price_in_inr","rating","reviews","score","flags","link"]].reset_index(drop=True)
    st.dataframe(display_df)

    st.markdown("---")
    st.info("Use the 'Open' button to visit the product page. Mark Bought to create labels for model training. Low or $ prices are converted to INR automatically.")

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
        SELECT r.price_in_inr as price, r.rating, r.reviews,
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

if st.button("Download full SQLite DB"):
    try:
        with open(DB_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download DB file", data, file_name=Path(DB_PATH).name)
    except Exception as e:
        st.error(f"Cannot read DB: {e}")

st.markdown("Notes: The app converts USD or other currencies into INR via exchangerate.host. It flags improbable prices (e.g., phones < ₹5,000) so you can inspect results. CSV files are saved per-search in the app folder — download them immediately to keep history (Streamlit Cloud filesystem is ephemeral).")
