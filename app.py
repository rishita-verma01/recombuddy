# app.py
# Streamlit Price Compare (SerpAPI) + INR conversion + CSV logging + defensive DB writes
# Deploy on Streamlit Cloud. Add SERPAPI_KEY to Streamlit Secrets.
#
# Requirements: streamlit, pandas, requests, lightgbm (optional)
# Save this file as app.py

import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
import time
from datetime import datetime
from pathlib import Path
import re
import traceback

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

# ---------- CONFIG ----------
DB_PATH = "price_compare.db"
CSV_PATH = "results_log.csv"
BAD_ROWS_LOG = "bad_rows.log"
MIN_LABELS_TO_TRAIN = 10
MAX_SERPAPI_RESULTS = 30
PLAUSIBILITY_MULTIPLIER = 0.3   # results with INR price < median*multiplier will be flagged/removed
EXCHANGE_API = "https://api.exchangerate.host/latest?base=USD&symbols=INR"
st.set_page_config(page_title="Best Deal Finder (SerpAPI) — INR", layout="wide")
# ----------------------------

st.title("🛒 Best Deal Finder — SerpAPI + INR conversion + CSV logging (defensive DB writes)")

# Load secrets (Streamlit Cloud)
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY not found. Add it in Streamlit Secrets or set SERPAPI_KEY as an env var.")
    st.stop()

# ---------- Utilities ----------
def fetch_usd_inr_rate():
    """Fetch live USD -> INR exchange rate from exchangerate.host (no key)."""
    try:
        r = requests.get(EXCHANGE_API, timeout=8)
        data = r.json()
        rate = data.get("rates", {}).get("INR")
        if rate:
            return float(rate)
    except Exception:
        pass
    # fallback hard-coded reasonable default (keep in case API fails)
    return 88.7

def clean_price_string(p):
    """Take raw price string and return (value, currency) or (None, None)."""
    if p is None:
        return None, None
    s = str(p).strip()
    # Remove non-printables
    s = s.replace('\u200b','').replace('\xa0',' ')
    # If it contains '₹' or 'Rs' or 'INR'
    if "₹" in s or "rs" in s.lower() or "inr" in s.lower():
        num = re.sub(r"[^\d.,]", "", s)
        num = num.replace(",", "")
        try:
            return float(num), "INR"
        except:
            return None, None
    # USD/dollar
    if "$" in s or "usd" in s.lower():
        num = re.sub(r"[^\d.,]", "", s)
        num = num.replace(",", "")
        try:
            return float(num), "USD"
        except:
            return None, None
    # If it's numeric without currency, try to heuristically decide:
    num = re.sub(r"[^\d.,]", "", s).replace(",", "")
    if num == "":
        return None, None
    try:
        val = float(num)
        currency_guess = "INR" if val > 1000 else "UNKNOWN"
        return val, currency_guess
    except:
        return None, None

def to_inr(value, currency, usd_inr_rate):
    if value is None:
        return None
    if currency == "INR":
        return float(value)
    if currency == "USD":
        return float(value) * float(usd_inr_rate)
    # UNKNOWN: assume INR (best-effort)
    return float(value)

# ---------- DB & CSV helpers ----------
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
        raw_price TEXT,
        parsed_currency TEXT,
        price_in_inr REAL,
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

def append_csv(rows, csv_path=CSV_PATH):
    df = pd.DataFrame(rows)
    is_new = not Path(csv_path).exists()
    df.to_csv(csv_path, mode="a", index=False, header=is_new)

db_conn = init_db(DB_PATH)

def log_search(query):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO searches (query, created_at) VALUES (?, ?)", (query, ts))
    db_conn.commit()
    return cur.lastrowid

# Defensive log_results implementation
def log_results(search_id, results):
    """
    Robustly insert rows into results table.
    Coerces/normalizes values, skips bad rows and logs them to bad_rows.log and Streamlit.
    """
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    bad_log_path = BAD_ROWS_LOG

    for r in results:
        try:
            # Normalize/clean each field (force types)
            title = str(r.get("title") or "")[:1000]
            store = str(r.get("store") or "")[:200]
            raw_price = str(r.get("raw_price") or "")[:200]
            parsed_currency = str(r.get("parsed_currency") or "")[:50]

            # price_in_inr: allow NULL if not parseable
            price_in_inr = r.get("price_in_inr")
            if price_in_inr in (None, "", "None"):
                price_in_inr = None
            else:
                try:
                    price_in_inr = float(price_in_inr)
                except Exception:
                    price_in_inr = None

            # rating -> float
            rating = r.get("rating") if r.get("rating") not in (None, "") else 0.0
            try:
                rating = float(rating)
            except Exception:
                rating = 0.0

            # reviews -> int
            reviews = r.get("reviews") if r.get("reviews") not in (None, "") else 0
            try:
                reviews = int(reviews)
            except Exception:
                try:
                    reviews = int(float(reviews))
                except Exception:
                    reviews = 0

            link = str(r.get("link") or "")[:2000]
            image = str(r.get("image") or "")[:2000]
            score = r.get("score") if r.get("score") not in (None, "") else 0.0
            try:
                score = float(score)
            except Exception:
                score = 0.0

            # Execute INSERT using parameterized query
            cur.execute("""
                INSERT INTO results
                (search_id, title, store, raw_price, parsed_currency, price_in_inr, rating, reviews, link, image, score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (search_id, title, store, raw_price, parsed_currency, price_in_inr, rating, reviews, link, image, score, ts))

        except Exception as exc:
            # Log the error (append to a simple local log) and continue
            try:
                with open(bad_log_path, "a", encoding="utf8") as f:
                    f.write(f"{datetime.utcnow().isoformat()} | ERROR: {exc} | ROW: {repr(r)}\n")
            except Exception:
                pass
            # Surface a small non-blocking message to Streamlit logs
            try:
                st.warning("Skipped one result row due to logging error. Check bad_rows.log for details.")
            except Exception:
                print(f"Skipped one result row due to error: {exc}")
            continue

    # commit once after loop
    db_conn.commit()

def log_purchase(result_row_id, search_id, bought_price):
    cur = db_conn.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO purchases (result_id, search_id, bought_price, created_at) VALUES (?, ?, ?, ?)",
                (result_row_id, search_id, bought_price, ts))
    db_conn.commit()

# ---------- SerpAPI fetch + robust parsing ----------
def fetch_serpapi(query, serpapi_key, max_results=MAX_SERPAPI_RESULTS, usd_inr_rate=88.7):
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
    except Exception as exc:
        st.error(f"SerpAPI request failed: {exc}")
        return []
    out = []
    for item in data.get("shopping_results", []):
        raw_price = item.get("price") or item.get("inventoried_price") or ""
        raw_price_str = str(raw_price)
        value, currency = clean_price_string(raw_price_str)
        price_in_inr = to_inr(value, currency, usd_inr_rate) if value is not None else None

        if price_in_inr is None:
            # Skip items we cannot parse price for
            continue

        out.append({
            "title": item.get("title") or item.get("product_title") or "",
            "store": item.get("source") or item.get("store") or "",
            "raw_price": raw_price_str,
            "parsed_currency": currency,
            "price_in_inr": float(price_in_inr),
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "link": item.get("link") or "",
            "image": item.get("thumbnail") or ""
        })
    return out

# ---------- Ranking & plausibility ----------
def heuristic_rank_df(df):
    if df.empty:
        return df
    df = df.copy()
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
    try:
        df_train = pd.read_sql_query(q, db_conn_local)
    except Exception:
        return None
    if df_train.shape[0] < MIN_LABELS_TO_TRAIN or not LGB_AVAILABLE:
        return None
    df_train = df_train.fillna(0)
    feature_cols = ["price", "rating", "reviews"]
    X = df_train[feature_cols]
    y = df_train["bought"]
    lgb_train = lgb.Dataset(X, label=y)
    params = {"objective":"binary", "metric":"binary_logloss", "verbosity": -1}
    model = lgb.train(params, lgb_train, num_boost_round=100)
    X_new = feature_df[["price_in_inr", "rating", "reviews"]].fillna(0)
    X_new = X_new.rename(columns={"price_in_inr":"price"})
    preds = model.predict(X_new[["price","rating","reviews"]])
    return preds, model

def get_ranked_results(results, usd_inr_rate):
    df = pd.DataFrame(results)
    if df.empty:
        return df, []
    median_price = df["price_in_inr"].median()
    plausible_threshold = (median_price * PLAUSIBILITY_MULTIPLIER) if median_price and median_price>0 else 0
    plausible = df[df["price_in_inr"] >= plausible_threshold].copy()
    suspect = df[df["price_in_inr"] < plausible_threshold].copy()

    ml = None
    try:
        ml = train_lgb_and_score(plausible, db_conn)
    except Exception:
        ml = None

    if ml:
        preds, model = ml
        plausible["score"] = preds
        plausible = plausible.sort_values(by="score", ascending=False).reset_index(drop=True)
    else:
        plausible = heuristic_rank_df(plausible)
        plausible = plausible.sort_values(by="score", ascending=False).reset_index(drop=True)

    if not suspect.empty:
        suspect = suspect.copy()
        suspect["score"] = -1.0
        suspect = suspect.reset_index(drop=True)
    combined = pd.concat([plausible, suspect], ignore_index=True)
    return combined, suspect.to_dict('records')

# ---------- Helper: show recent bad rows ----------
def show_recent_bad_rows(n=20, bad_log_path=BAD_ROWS_LOG):
    if Path(bad_log_path).exists():
        with open(bad_log_path, "r", encoding="utf8") as f:
            lines = f.readlines()[-n:]
        return "".join(lines)
    return "No bad_rows.log present."

# ---------- UI ----------
st.markdown("Type product name. We fetch Google Shopping results (SerpAPI), normalize currencies to INR, filter suspiciously cheap results, and save everything to CSV + SQLite. Buttons open product pages in a new tab.")

query = st.text_input("Enter product name (e.g. iPhone 16, JBL earphones)")

if query:
    with st.spinner("Fetching live USD→INR rate..."):
        usd_inr_rate = fetch_usd_inr_rate()
    st.write(f"Using USD → INR rate: **{usd_inr_rate:.4f}** (fetched live).")
    with st.spinner("Fetching prices from SerpAPI..."):
        serp_results = fetch_serpapi(query, SERPAPI_KEY, max_results=MAX_SERPAPI_RESULTS, usd_inr_rate=usd_inr_rate)

    seen = set()
    unique = []
    for r in serp_results:
        key = (str(r.get("title",""))[:120].strip().lower(), str(r.get("store","")).strip().lower(), int(r.get("price_in_inr") or 0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)

    ranked_df, suspect_list = get_ranked_results(unique, usd_inr_rate)

    # Log search + results & append CSV (robust)
    search_id = log_search(query)
    logged_rows = []
    for _, row in ranked_df.iterrows():
        rec = {
            "timestamp": datetime.utcnow().isoformat(),
            "search_id": search_id,
            "title": row.get("title"),
            "store": row.get("store"),
            "raw_price": row.get("raw_price"),
            "parsed_currency": row.get("parsed_currency"),
            "price_in_inr": float(row.get("price_in_inr") or 0),
            "rating": float(row.get("rating") or 0),
            "reviews": int(row.get("reviews") or 0),
            "link": row.get("link") or "",
            "image": row.get("image") or "",
            "score": float(row.get("score") or 0)
        }
        logged_rows.append(rec)

    if logged_rows:
        try:
            append_csv(logged_rows, CSV_PATH)
        except Exception as e:
            st.warning(f"Failed appending CSV: {e}")
    # Use defensive DB logger
    log_results(search_id, logged_rows)

    st.subheader("🏆 Top 3 (plausible) Deals")
    top3 = ranked_df[ranked_df["score"]>=0].head(3)
    for idx, row in top3.reset_index(drop=True).iterrows():
        cols = st.columns([1,4,1])
        with cols[0]:
            if row.get("image"):
                try:
                    st.image(row.get("image"), width=120)
                except Exception:
                    pass
        with cols[1]:
            st.markdown(f"**{row.get('title')}**")
            st.write(f"Store: **{row.get('store')}** — Price: **₹{int(row.get('price_in_inr')):,}**")
            st.write(f"Raw price: `{row.get('raw_price')}` — Parsed currency: `{row.get('parsed_currency')}`")
            if row.get("rating"):
                st.write(f"Rating: {row.get('rating')} ⭐ ({row.get('reviews')} reviews)")
        with cols[2]:
            url = row.get("link") or "#"
            btn_html = f"""<a href="{url}" target="_blank" rel="noopener noreferrer"><button style="padding:8px 12px;border-radius:6px;">Open Product</button></a>"""
            st.markdown(btn_html, unsafe_allow_html=True)
            if st.button(f"Bought — #{idx+1}", key=f"buy_top_{idx}_{time.time()}"):
                try:
                    cur = db_conn.cursor()
                    cur.execute("SELECT id FROM results WHERE search_id = ? AND title = ? AND price_in_inr = ? ORDER BY id DESC LIMIT 1",
                                (search_id, row.get("title"), float(row.get("price_in_inr") or 0)))
                    res = cur.fetchone()
                    if res:
                        log_purchase(res[0], search_id, float(row.get("price_in_inr") or 0))
                        st.success("Purchase recorded — used for future training.")
                    else:
                        st.error("Could not record purchase (not found).")
                except Exception as e:
                    st.error(f"Error recording purchase: {e}")

    st.markdown("---")
    st.subheader("📦 All Results (ranked; suspect items shown at bottom)")
    display_df = ranked_df[["title","store","price_in_inr","raw_price","parsed_currency","rating","reviews","score","link"]].reset_index(drop=True)
    display_df = display_df.rename(columns={"price_in_inr":"price_in_inr (₹)"})
    st.dataframe(display_df)

    st.markdown("---")
    if suspect_list:
        st.warning(f"{len(suspect_list)} suspiciously low-price item(s) were detected and pushed to the bottom. You can still view them in 'All Results'.")

# ---------- Admin / Train & Debug ----------
st.markdown("---")
st.header("⚙️ Admin & Data")

cur = db_conn.cursor()
try:
    cur.execute("SELECT COUNT(*) FROM searches")
    search_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM results")
    result_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM purchases")
    purchase_count = cur.fetchone()[0]
except Exception:
    search_count = result_count = purchase_count = 0

st.write(f"Searches: **{search_count}** — Results stored: **{result_count}** — Purchases labeled: **{purchase_count}**")

if st.button("Force-train LightGBM (if enough labels)"):
    if not LGB_AVAILABLE:
        st.error("LightGBM not installed. Install via requirements to enable training.")
    else:
        try:
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
        except Exception as e:
            st.error(f"Training failed: {e}")

# Download DB or CSV
if st.button("Download SQLite DB"):
    try:
        with open(DB_PATH, "rb") as f:
            data = f.read()
        st.download_button("Download DB file", data, file_name=Path(DB_PATH).name)
    except Exception as e:
        st.error(f"Cannot read DB: {e}")

if Path(CSV_PATH).exists():
    try:
        with open(CSV_PATH, "rb") as f:
            csv_data = f.read()
        st.download_button("Download CSV Log", csv_data, file_name=Path(CSV_PATH).name)
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")

# Show / download bad_rows.log
st.markdown("---")
st.subheader("Debug: bad_rows.log (rows skipped during DB write)")
bad_preview = show_recent_bad_rows(n=50, bad_log_path=BAD_ROWS_LOG)
st.text_area("Recent bad_rows.log entries", bad_preview, height=200)
if Path(BAD_ROWS_LOG).exists():
    try:
        with open(BAD_ROWS_LOG, "rb") as f:
            bad_bytes = f.read()
        st.download_button("Download bad_rows.log", bad_bytes, file_name=Path(BAD_ROWS_LOG).name)
    except Exception as e:
        st.error(f"Cannot read bad_rows.log: {e}")

st.markdown("Notes: This version uses defensive DB writes. If SerpAPI returns odd/malformed fields, those rows are skipped and logged to bad_rows.log (visible here). Adjust PLAUSIBILITY_MULTIPLIER near the top to be stricter/looser about suspiciously cheap results.")
