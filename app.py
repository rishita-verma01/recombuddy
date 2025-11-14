import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime
from pathlib import Path

# Try LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

# ---------------- CONFIG ----------------
DB_PATH = "price_compare.db"
CSV_PATH = "search_results.csv"
MIN_LABELS_TO_TRAIN = 10
INDIAN_STORES = [
    "Amazon", "Amazon.in", "Flipkart", "Croma",
    "Reliance Digital", "Tata Cliq", "Vijay Sales",
    "Poorvika", "Samsung India", "Mi India",
    "Boat Lifestyle", "OnePlus India"
]
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"
# -----------------------------------------

st.set_page_config(page_title="India Price Compare", layout="wide")
st.title("🇮🇳 Best Deal Finder — India-Only Price Comparison (SerpAPI)")

# Load SerpAPI Key
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    st.error("Add SERPAPI_KEY in Streamlit Secrets.")
    st.stop()

# -----------------------------------------
# DATABASE INIT
# -----------------------------------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS searches(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT, created_at TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS results(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        search_id INTEGER, title TEXT, store TEXT,
        price REAL, rating REAL, reviews INTEGER,
        link TEXT, image TEXT, score REAL, created_at TEXT
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS purchases(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id INTEGER, search_id INTEGER,
        bought_price REAL, created_at TEXT
    )""")
    conn.commit()
    return conn

db_conn = init_db()

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
        cur.execute("""INSERT INTO results(search_id,title,store,price,rating,reviews,link,image,score,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (sid, r["title"], r["store"], r["price"], r["rating"],
                     r["reviews"], r["link"], r["image"], r["score"], t))
    db_conn.commit()

def append_to_csv(rows):
    df = pd.DataFrame(rows)
    file_exists = os.path.isfile(CSV_PATH)
    df.to_csv(CSV_PATH, mode="a", header=not file_exists, index=False)

# -----------------------------------------
# FETCH USD → INR
# -----------------------------------------
def get_usd_to_inr():
    try:
        r = requests.get(USD_TO_INR_API, timeout=10)
        data = r.json()
        return float(data["rates"]["INR"])
    except:
        return 83.0   # fallback

usd_inr_rate = get_usd_to_inr()

# -----------------------------------------
# CLEAN PRICE
# -----------------------------------------
def clean_price(p):
    if not p:
        return None

    p = p.replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    p = p.split("/")[0]   # remove EMI/m per month
    p = p.split("month")[0]
    p = p.split("EMI")[0]

    # Remove random characters
    filtered = "".join(ch for ch in p if (ch.isdigit() or ch == "."))

    if filtered == "":
        return None

    price = float(filtered)

    # If price suspiciously low (< INR 500) for premium products → accessory
    if price < 500:
        return None

    # If price likely USD (e.g. 799) convert to INR
    if price < 2000:  
        # maybe real accessory, or cheap product
        # but check keyword: if it is iPhone, laptop, TV etc convert
        pass

    if "$" in p or "USD" in p:
        price = price * usd_inr_rate

    return price

# -----------------------------------------
# SerpAPI FETCH (India-only)
# -----------------------------------------
def fetch_serpapi(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": 20,
        "hl": "en",
        "gl": "in"  # force India results
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    results = []

    for item in data.get("shopping_results", []):
        store = item.get("source", "")

        # filter non-Indian stores
        if not any(s.lower() in store.lower() for s in INDIAN_STORES):
            continue

        raw_price = item.get("price")
        final_price = clean_price(raw_price)
        if not final_price:
            continue

        results.append({
            "title": item.get("title", ""),
            "store": store,
            "price": final_price,
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "link": item.get("link", ""),
            "image": item.get("thumbnail", "")
        })

    return results

# -----------------------------------------
# RANKING
# -----------------------------------------
def heuristic(df):
    maxp = df["price"].max()
    df["rating_norm"] = df["rating"] / 5
    df["reviews_norm"] = df["reviews"] / (df["reviews"].max() + 1)
    df["price_norm"] = (maxp - df["price"]) / maxp
    df["score"] = df["price_norm"] * 0.7 + df["rating_norm"] * 0.2 + df["reviews_norm"] * 0.1
    return df

# -----------------------------------------
# UI
# -----------------------------------------
query = st.text_input("Search any product in India", placeholder="iPhone 15, Samsung AC, JBL Speaker")

if query:
    with st.spinner("Fetching real-time India prices…"):
        raw = fetch_serpapi(query)

    if not raw:
        st.error("No valid India price results found. Try different query.")
        st.stop()

    df = pd.DataFrame(raw)
    df = heuristic(df)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Logging
    search_id = log_search(query)
    log_results(search_id, df.to_dict("records"))
    append_to_csv(df.to_dict("records"))

    # ------------ TOP 3 ------------
    st.subheader("🚀 Top 3 Best Deals in India (Filtered, Cleaned, Sorted)")
    for i, row in df.head(3).iterrows():
        col1, col2, col3 = st.columns([1,4,1])
        with col1:
            if row["image"]:
                st.image(row["image"], width=110)
        with col2:
            st.markdown(f"### {row['title']}")
            st.write(f"**Store:** {row['store']}")
            st.write(f"**Price:** ₹{int(row['price']):,}")
            st.write(f"⭐ {row['rating']} ({row['reviews']} reviews)")
        with col3:
            st.button("Open Product Page", key=f"open_{i}", on_click=lambda link=row['link']: st.markdown(f"<script>window.open('{link}');</script>", unsafe_allow_html=True))

    st.markdown("---")

    # ---------- ALL RESULTS ----------
    st.subheader("📦 All Results (India Stores Only)")
    st.dataframe(df[["title","store","price","rating","reviews","link"]])

# Finish
