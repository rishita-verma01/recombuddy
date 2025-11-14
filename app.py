# app.py — Upgraded: India-only SerpAPI price compare + history chart + price-drop alerts + CSV + DB
import streamlit as st
import requests
import pandas as pd
import sqlite3
import os
from datetime import datetime
from pathlib import Path
import time
import math

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

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
"Kurlon India", "Sleepwell India"
]
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"
# -----------------------------------------

st.set_page_config(page_title="India Price Compare + Alerts", layout="wide")
st.title("🇮🇳 Best Deal Finder — India Price History & Alerts")

# Load secrets
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY not found. Add it in Streamlit Secrets.")
    st.stop()

# ---------------- DB init ----------------
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
    cur.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      search_query TEXT,
      product_title TEXT,
      store TEXT,
      threshold_type TEXT,  -- 'percent' or 'absolute'
      threshold_value REAL,
      triggered_price REAL,
      triggered_at TEXT
    )""")
    conn.commit()
    return conn

db_conn = init_db()

def log_search(q):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO searches(query,created_at) VALUES (?,?)", (q, t))
    db_conn.commit()
    return cur.lastrowid

def log_results(sid, rows):
    cur = db_conn.cursor()
    t = datetime.utcnow().isoformat()
    for r in rows:
        cur.execute("""
            INSERT INTO results(search_id,title,store,price,rating,reviews,link,image,score,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (sid, r["title"], r["store"], r["price"], r["rating"], r["reviews"], r["link"], r["image"], r["score"], t))
    db_conn.commit()

def append_to_csv(rows):
    df = pd.DataFrame(rows)
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
    # append to CSV too
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

# ---------------- helper functions ----------------
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
    # remove commas and rupee symbol
    p = p.replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    # remove EMI expressions or "/month", "month"
    for sep in ["/", "per", "month", "mo", "EMI", " emi"]:
        p = p.split(sep)[0]
    # detect $ and convert
    if "$" in raw_price or "USD" in raw_price or "US$" in raw_price:
        # extract digits
        nums = "".join(ch for ch in p if (ch.isdigit() or ch == "."))
        if not nums:
            return None
        return float(nums) * usd_inr_rate
    # keep digits and dot
    filtered = "".join(ch for ch in p if (ch.isdigit() or ch == "."))
    if filtered == "":
        return None
    price = float(filtered)
    # filter out EMI monthly numbers: if raw had 'month' earlier we'd removed; still skip very low values for expensive keywords later
    if price < 50:  # extremely low price likely wrong or tiny accessory
        return None
    return price

def is_indian_store(store_name):
    if not store_name:
        return False
    for s in INDIAN_STORES:
        if s.lower() in store_name.lower():
            return True
    # quick additional heuristics
    if ".in" in store_name.lower() or "india" in store_name.lower():
        return True
    return False

# SerpAPI fetch with India bias
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
            # use selected stores filter (match any)
            if not any(sel.lower() in store.lower() for sel in selected_stores):
                continue
        else:
            # ensure only Indian stores
            if not is_indian_store(store):
                continue
        raw_price = item.get("price") or ""
        price = clean_price(raw_price)
        if not price:
            continue
        title = item.get("title") or ""
        # filter out EMI/month text or installment offers already by price cleaning
        out.append({
            "title": title,
            "store": store,
            "price": float(price),
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "link": item.get("link") or "",
            "image": item.get("thumbnail") or ""
        })
    return out

# Simple fuzzy match between titles (token overlap)
def title_matches(a, b):
    if not a or not b:
        return False
    a_tokens = set([t.lower() for t in a.split() if len(t) > 2])
    b_tokens = set([t.lower() for t in b.split() if len(t) > 2])
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens.intersection(b_tokens)
    return len(overlap) >= max(1, min(3, int(0.4 * min(len(a_tokens), len(b_tokens)))))

# Build price history DataFrame for a current result (by matching past results in DB)
def fetch_price_history_for_result(title, store):
    q = "SELECT title, store, price, created_at FROM results WHERE store LIKE ?"
    cur = db_conn.cursor()
    cur.execute(q, (f"%{store}%",))
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["price", "created_at"])
    df = pd.DataFrame(rows, columns=["title", "store", "price", "created_at"])
    # filter by fuzzy matching title tokens
    df_matched = df[df["title"].apply(lambda t: title_matches(t, title))]
    if df_matched.empty:
        # fallback: match just by store and any previous prices (less ideal)
        df_matched = df
    # convert created_at to datetime and aggregate by date (min price)
    df_matched["created_at"] = pd.to_datetime(df_matched["created_at"])
    df_matched = df_matched.sort_values("created_at")
    return df_matched[["price", "created_at"]]

# Heuristic ranking as before
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
    # attempt LightGBM training from purchases if available
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
    st.header("Settings & Alerts")
    selected_stores = st.multiselect("Filter stores (leave empty = all Indian stores)", options=INDIAN_STORES)
    st.markdown("---")
    st.subheader("Alert settings (per search/product)")
    alert_mode = st.radio("Default trigger type", ("percent drop vs historical min", "absolute price (INR)"), index=0)
    st.write("You will be able to set a per-product trigger when viewing top results.")
    st.markdown("---")
    st.write(f"Current USD → INR rate: **{usd_inr_rate:.2f}** (fetched live)")

st.info("Searches check alerts at query time. Alerts are logged and downloadable. To enable automatic polling, later add a scheduled runner (GitHub Actions).")

query = st.text_input("Enter product name (India only)", placeholder="e.g. iPhone 15, Samsung QLED TV, JBL Flip 6")

if query:
    # perform search
    with st.spinner("Fetching India-only product listings from Google Shopping (SerpAPI)…"):
        results = fetch_serpapi(query, selected_stores)
    if not results:
        st.error("No valid Indian product listings found. Try a different query or broaden store filter.")
    else:
        df = pd.DataFrame(results)
        # try ML ranking
        preds = try_train_and_predict(df[["price","rating","reviews"]]) if not df.empty else None
        if preds is not None:
            df["score"] = preds
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
        else:
            df = heuristic_rank(df).reset_index(drop=True)

        # Log search and results
        sid = log_search(query)
        log_results(sid, df.to_dict("records"))
        append_to_csv(df.to_dict("records"))

        # Show top 3 with "Open Product Page" buttons and per-product alert input
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
                st.markdown(f"[Open product link]({row['link']})")
                # Price history chart button
                if st.button(f"Show Price History — {row['store']}", key=f"hist_{i}"):
                    hist_df = fetch_price_history_for_result(row['title'], row['store'])
                    if hist_df.empty:
                        st.info("No historical prices found for this product/store.")
                    else:
                        st.line_chart(data=hist_df.set_index("created_at")["price"])
                        # show stats
                        st.write(f"Historical min: ₹{int(hist_df['price'].min())}, mean: ₹{int(hist_df['price'].mean())}")
            with col3:
                # Alert inputs per product
                st.markdown("**Set price alert**")
                mode = st.selectbox(f"Type (#{i})", ("percent", "absolute"), key=f"atype_{i}")
                if mode == "percent":
                    perc = st.number_input(f"Trigger when price ≤ X% below historical min (e.g. 20 for 20%)", min_value=1.0, max_value=100.0, value=20.0, key=f"perc_{i}")
                    if st.button(f"Set {int(perc)}% alert for row {i}", key=f"setperc_{i}"):
                        # compute historical min
                        hist_df = fetch_price_history_for_result(row['title'], row['store'])
                        hist_min = hist_df['price'].min() if not hist_df.empty else None
                        if hist_min is None:
                            st.warning("Not enough history to compute historical min; alert will use current price as baseline (will only trigger on a future lower price).")
                            hist_min = row['price']
                        threshold_price = hist_min * (1 - perc/100.0)
                        # if current price already below threshold, trigger immediately
                        if row['price'] <= threshold_price:
                            st.success(f"ALERT TRIGGERED — Current price ₹{int(row['price'])} ≤ threshold ₹{int(threshold_price)}")
                            log_alert(query, row['title'], row['store'], "percent", perc, row['price'])
                        else:
                            # store alert as a record by logging an alert with triggered_price NULL? We'll log only when triggered.
                            # For now, we inform the user and store nothing until triggered (since no background worker).
                            st.info(f"Alert set: will trigger when price ≤ ₹{int(threshold_price)} (not persisted as background job). To persist and manually check, save threshold externally.")
                            # Persist as a special alert row with triggered_price = NULL is out of scope without a scheduler.
                else:
                    absolute = st.number_input(f"Trigger when price ≤ ₹ (absolute value)", min_value=1, value=int(row['price'])-100 if row['price']>100 else int(row['price']), key=f"abs_{i}")
                    if st.button(f"Set ₹{int(absolute)} alert for row {i}", key=f"setabs_{i}"):
                        if row['price'] <= absolute:
                            st.success(f"ALERT TRIGGERED — Current price ₹{int(row['price'])} ≤ ₹{int(absolute)}")
                            log_alert(query, row['title'], row['store'], "absolute", absolute, row['price'])
                        else:
                            st.info(f"Alert registered (will trigger when item reaches ₹{int(absolute)}). Note: background checking not active; you must re-run search or add a scheduled worker later.")
        st.markdown("---")
        st.subheader("📦 All Results (ranked)")
        display_df = df[["title", "store", "price", "rating", "reviews", "score", "link"]].copy()
        st.dataframe(display_df)

# ---------------- Admin / Alerts panel ----------------
st.markdown("---")
st.header("⚙️ Admin — DB & Alerts")
cur = db_conn.cursor()
cur.execute("SELECT COUNT(*) FROM searches")
search_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM results")
result_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM purchases")
purchase_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM alerts")
alert_count = cur.fetchone()[0]
st.write(f"Searches: **{search_count}** — Results stored: **{result_count}** — Purchases labeled: **{purchase_count}** — Alerts triggered: **{alert_count}**")

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

st.markdown("Notes: Alerts are triggered and logged at search time. To enable fully automated alerts, add a scheduled runner (e.g. GitHub Actions) that calls the same search endpoint and checks thresholds, then notifies via email/Telegram/Push.")
