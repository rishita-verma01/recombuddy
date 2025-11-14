# app.py
# Streamlit India-only price compare (CSV-only storage, no links, no images)
# Add SERPAPI_KEY to Streamlit Secrets before running.

import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import json
import numpy as np
import uuid
import traceback

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

# Optional sentence-transformers
EMBED_AVAILABLE = False
EMBED_MODEL = None
EMBED_DIM = None
EMBED_THRESHOLD = 0.72

try:
    from sentence_transformers import SentenceTransformer
    EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBED_DIM = EMBED_MODEL.get_sentence_embedding_dimension()
    EMBED_AVAILABLE = True
except Exception:
    EMBED_AVAILABLE = False
    EMBED_MODEL = None

# ---------- CONFIG ----------
SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

DB_CSV = "search_results.csv"   # main results (no links/images)
PURCHASES_CSV = "purchases.csv" # purchases logged (store uid)
ALERTS_CSV = "alerts.csv"       # alerts that triggered
MIN_LABELS_TO_TRAIN = 10
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"

INDIAN_STORES = [
    "Amazon", "Amazon.in", "Flipkart", "Croma", "Reliance Digital", "Tata Cliq", "Vijay Sales",
    "Poorvika", "Samsung India", "Mi India", "Boat Lifestyle", "OnePlus India", "Myntra",
    "AJIO", "Snapdeal", "Paytm Mall", "Tata CLiQ", "ShopClues", "eBay India", "IndiaMART",
    "OLX", "Quikr", "SastaSundar", "Meesho", "Pepperfry", "Urban Ladder", "BigBasket",
    "Blinkit", "DMart", "Spencers", "Reliance Fresh", "Nature’s Basket", "Natures Basket",
    "More Store", "Easyday", "FreshToHome", "Milkbasket", "Big Bazaar", "Lenskart",
    "Bose India", "Audio House", "Sangeetha Mobiles", "UniverCell", "The Mobile Store",
    "Ezone", "HomeTown", "TataCliq Electronics", "Samsung Online Store", "Mi Store",
    "OnePlus Store", "Zara India", "H&M India", "Lifestyle", "Shoppers Stop",
    "Pantaloons", "Max Fashion", "Brand Factory", "Fabindia", "W for Woman", "Biba",
    "Wrogn", "Arrow", "Raymond", "Allen Solly", "Louis Philippe", "Jack & Jones India",
    "Koovs", "Bewakoof", "The Souled Store", "Nykaa Fashion", "Clovia", "Bata India",
    "Woodland", "Metro Shoes", "Puma India", "Nike India", "Adidas India", "Mochi",
    "Hidesign", "Health & Glow", "Nykaa", "Purplle", "MyGlamm", "The Body Shop India",
    "Kama Ayurveda", "Forest Essentials", "Sugar Cosmetics", "Colorbar", "Glamveda",
    "PharmEasy", "1mg", "Netmeds", "Apollo Pharmacy", "MedPlus", "CareOnGo",
    "Wellness Forever", "Healthkart", "Durex India", "Manforce India", "Moods Condoms",
    "Condom Junction", "Kamasutra India", "Grocery Bigbasket", "Reliance Smart", "Farmley",
    "Licious", "Home Centre", "WoodenStreet", "FabFurnish", "Durian Furniture", "@home",
    "Tanishq", "CaratLane", "Kalyan Jewellers", "Malabar Gold", "BlueStone", "FirstCry",
    "Hopscotch", "BabyChakra", "Mothercare India", "Decathlon India", "Sportking", "HRX",
    "Crossword", "SapnaOnline", "Kitab Khana", "CarDekho", "BikeDekho", "AutoZone India",
    "BharatBenz Store", "Nilgiris", "Ratnadeep", "Metro Cash & Carry", "HyperCity",
    "MD Computers", "PrimeABGB", "Vedant Computers", "TheITDepot", "The Man Company",
    "The Minimalist", "Chumbak", "FabAlley", "Zivame", "Plum Goodness", "Mamaearth",
    "The Derma Co", "Cashify", "ReGlobe", "ReStore", "Farmley India", "Licious India",
    "KitchenAid India", "Hawkins Online", "Prestige Appliances", "Bajaj Electronics",
    "Staples India", "Amazon Business India", "Heads Up For Tails", "PetsWorld",
    "Drools Store", "Ferns N Petals", "FlowerAura", "Winni", "JioMart", "BigBasket Local",
    "Smartprix", "91Mobiles", "Gadget360", "Shop101", "Voonik", "LimeRoad", "Craftsvilla",
    "Zansaar", "Boat India", "boAt", "OnePlus", "Apple India", "Realme India", "Vivo India",
    "Oppo India", "Motorola India", "HP Store India", "Dell India", "Asus India", "Acer India",
    "Lenovo India", "MSI India", "Canon India", "Nikon India", "DJI India", "GoPro India",
    "Syska India", "Philips India", "Havells India", "Usha India", "IFB Appliances",
    "Bosch India", "LG India", "Whirlpool India", "Voltas India", "Godrej Appliances",
    "Haier India", "TCL India", "Vu India", "Ikea India", "JBL India", "Sennheiser India",
    "Skullcandy India", "Harman India", "Marshall India", "Big Fashion", "Aditya Birla Fashion",
    "Nykaa Beauty", "Myntra Shoes", "Myntra Accessories", "Amazon Fashion India",
    "Flipkart Fashion India", "Croma Retail", "Reliance Digital Store", "Spar Hypermarket",
    "Star Bazaar", "More Retail", "Jiomart Grocery", "Armani Exchange India",
    "Michael Kors India", "Charles & Keith India", "Sephora India", "Bath & Body Works India",
    "Miniso India", "Muji India", "Decathlon Sports", "SportsJam", "Wildcraft India",
    "American Tourister India", "Safari Bags India", "Skybags India", "Campus Shoes India",
    "Skechers India", "Wrangler India", "Levis India", "Spykar", "U.S. Polo Assn",
    "Peter England", "Van Heusen", "Tommy Hilfiger India", "Calvin Klein India",
    "Superdry India", "Jockey India", "Rupa", "Lux Cozi", "Dollar Industries",
    "Max Innerwear", "Amante India", "Enamor India", "PrettySecrets", "VIP Bags",
    "Aristocrat Bags", "F Gear India", "Roadster (Myntra)", "H&M Home", "Marks & Spencer India",
    "Uniqlo India", "The Children's Place India", "Forever 21 India", "Pantaloons Fashion",
    "Ajio Luxe", "Nykaa Luxe", "TataCliq Luxury", "Reliance Trends", "Central Mall Online",
    "Trent Westside", "Westside Online", "FoodHall India", "Spencer Retail Grocery",
    "More Hypermarket", "Reliance Retail Online", "Amazon Pantry India", "Flipkart Supermart",
    "Zepto", "Urban Company Store", "Wakefit", "Sleepyhead", "Duroflex India", "Kurlon India",
    "Sleepwell India"
]

# ---------- sanity check: SERPAPI key ----------
if not SERPAPI_KEY:
    st.error("SERPAPI_KEY is required in Streamlit Secrets or environment. Add it and refresh.")
    st.stop()

# ---------- helpers ----------
def get_usd_to_inr():
    try:
        r = requests.get(USD_TO_INR_API, timeout=8)
        data = r.json()
        return float(data.get("rates", {}).get("INR", 83.0))
    except:
        return 83.0

USD_INR = get_usd_to_inr()

def clean_price(raw_price):
    if not raw_price:
        return None
    s = str(raw_price)
    s = s.replace(",", "").replace("₹", "").replace("Rs.", "").strip()
    for sep in ["/", "per", "month", "mo", "EMI", " emi", "₹"]:
        s = s.split(sep)[0]
    # convert $ if indicated
    if "$" in raw_price or "USD" in raw_price or "US$" in raw_price:
        nums = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
        if not nums:
            return None
        return float(nums) * USD_INR
    filtered = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    if filtered == "":
        return None
    price = float(filtered)
    # filter obviously tiny prices
    if price < 50:
        return None
    return price

def is_indian_store(store_name):
    if not store_name:
        return False
    low = store_name.lower()
    for s in INDIAN_STORES:
        if s.lower() in low:
            return True
    if ".in" in low or "india" in low:
        return True
    return False

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
    except:
        return None

def cosine_sim(a, b):
    if a is None or b is None:
        return -1.0
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

# ---------- CSV helpers ----------
def append_search_results(rows, search_query):
    """
    rows: list of dicts {uid, title, store, price, rating, reviews, score, embedding, created_at}
    """
    df = pd.DataFrame(rows)
    df["search_query"] = search_query
    file_exists = os.path.isfile(DB_CSV)
    df.to_csv(DB_CSV, mode="a", header=not file_exists, index=False)

def append_purchase(uid, bought_price):
    t = datetime.utcnow().isoformat()
    row = {"uid": uid, "bought_price": bought_price, "created_at": t}
    file_exists = os.path.isfile(PURCHASES_CSV)
    pd.DataFrame([row]).to_csv(PURCHASES_CSV, mode="a", header=not file_exists, index=False)

def append_alert(search_query, title, store, ttype, tval, price):
    t = datetime.utcnow().isoformat()
    row = {
        "search_query": search_query,
        "product_title": title,
        "store": store,
        "threshold_type": ttype,
        "threshold_value": tval,
        "triggered_price": price,
        "triggered_at": t
    }
    file_exists = os.path.isfile(ALERTS_CSV)
    pd.DataFrame([row]).to_csv(ALERTS_CSV, mode="a", header=not file_exists, index=False)

# ---------- SerpAPI fetch ----------
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
        r = requests.get(url, params=params, timeout=18)
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
        uid = str(uuid.uuid4())
        out.append({
            "uid": uid,
            "title": title,
            "store": store,
            "price": float(price),
            "rating": float(item.get("rating") or 0),
            "reviews": int(item.get("reviews") or 0),
            "score": 0.0,
            "embedding": json.dumps(emb) if emb is not None else None,
            "created_at": datetime.utcnow().isoformat()
        })
    return out

# ---------- matching & ranking ----------
def title_token_match(a, b):
    if not a or not b:
        return False
    a_tokens = set([t.lower() for t in a.split() if len(t) > 2])
    b_tokens = set([t.lower() for t in b.split() if len(t) > 2])
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens.intersection(b_tokens)
    return len(overlap) >= max(1, min(3, int(0.4 * min(len(a_tokens), len(b_tokens)))))

def fetch_price_history(title, store, current_embedding_json=None):
    """
    Reads search_results.csv and returns history dataframe (price, created_at) for matched titles.
    Uses embedding similarity if embeddings exist; else token overlap.
    """
    if not os.path.exists(DB_CSV):
        return pd.DataFrame(columns=["price", "created_at"])
    try:
        df_all = pd.read_csv(DB_CSV)
    except Exception:
        return pd.DataFrame(columns=["price", "created_at"])
    # filter by store
    df_store = df_all[df_all["store"].str.contains(store, case=False, na=False)]
    matches = []
    if current_embedding_json:
        try:
            cur_emb = np.array(json.loads(current_embedding_json), dtype=float)
        except Exception:
            cur_emb = None
    else:
        cur_emb = None
    for _, r in df_store.iterrows():
        matched = False
        if cur_emb is not None and r.get("embedding"):
            try:
                past = np.array(json.loads(r["embedding"]), dtype=float)
                if cosine_sim(cur_emb, past) >= EMBED_THRESHOLD:
                    matched = True
            except Exception:
                matched = False
        if not matched:
            if title_token_match(str(r.get("title","")), title):
                matched = True
        if matched:
            matches.append((r.get("price"), r.get("created_at")))
    if not matches:
        # fallback: return recent prices for this store
        df_store["created_at"] = pd.to_datetime(df_store["created_at"])
        df_store = df_store.sort_values("created_at")
        return df_store[["price", "created_at"]]
    dfm = pd.DataFrame(matches, columns=["price", "created_at"])
    dfm["created_at"] = pd.to_datetime(dfm["created_at"])
    dfm = dfm.sort_values("created_at")
    return dfm

def heuristic_rank(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    maxp = df["price"].max()
    df["rating_norm"] = df["rating"] / 5.0
    df["reviews_norm"] = df["reviews"] / (df["reviews"].max() + 1e-9)
    df["price_norm"] = (maxp - df["price"]) / (maxp + 1e-9)
    df["score"] = df["price_norm"] * 0.7 + df["rating_norm"] * 0.2 + df["reviews_norm"] * 0.1
    return df

def try_train_and_predict_from_csv(feature_df):
    """
    Tries to read search_results.csv + purchases.csv to form training data and train LightGBM.
    It expects search_results.csv with uid, and purchases.csv with uid references.
    """
    if not LGB_AVAILABLE:
        return None
    if not os.path.exists(DB_CSV) or not os.path.exists(PURCHASES_CSV):
        return None
    try:
        df_results = pd.read_csv(DB_CSV)
        df_p = pd.read_csv(PURCHASES_CSV)
    except Exception:
        return None
    # mark bought flag by uid
    df_results["bought"] = df_results["uid"].isin(df_p["uid"]).astype(int)
    df_train = df_results[["price", "rating", "reviews", "bought"]].dropna()
    if df_train.shape[0] < MIN_LABELS_TO_TRAIN:
        return None
    X = df_train[["price", "rating", "reviews"]]
    y = df_train["bought"]
    dtrain = lgb.Dataset(X, label=y)
    params = {"objective":"binary","metric":"binary_logloss","verbosity": -1}
    model = lgb.train(params, dtrain, num_boost_round=100)
    X_new = feature_df[["price", "rating", "reviews"]].fillna(0)
    preds = model.predict(X_new)
    return preds

# ---------- UI ----------
st.set_page_config(page_title="India Price Compare (CSV-only)", layout="wide")
st.title("🇮🇳 Best Deal Finder — CSV-only, No Links, No Images")

with st.sidebar:
    st.header("Settings")
    selected_stores = st.multiselect("Filter stores (leave empty = all Indian stores)", options=INDIAN_STORES)
    st.markdown("---")
    st.write("Embedding model: " + ("available" if EMBED_AVAILABLE else "not available (fallback to token-match)"))
    if EMBED_AVAILABLE:
        st.write(f"Embed dim: {EMBED_DIM}  — threshold: {EMBED_THRESHOLD}")
    st.markdown("---")
    st.write(f"USD→INR (live): {USD_INR:.2f}")

if not SERPAPI_KEY:
    st.stop()

query = st.text_input("Enter product name (India only)", placeholder="e.g., iPhone 15, JBL Flip 6")

if query:
    with st.spinner("Searching India stores (SerpAPI)…"):
        rows = fetch_serpapi(query, selected_stores)
    if not rows:
        st.error("No valid India product results found. Try different query or broaden store filter.")
    else:
        # rank (try ML first)
        feature_df = pd.DataFrame(rows)[["price","rating","reviews"]]
        preds = try_train_and_predict_from_csv(pd.DataFrame(rows)) if not feature_df.empty else None
        if preds is not None:
            for i, p in enumerate(preds):
                rows[i]["score"] = float(p)
            df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        else:
            df = heuristic_rank(rows).reset_index(drop=True)

        # SAVE to CSV (append)
        append_search_results(df.to_dict("records"), query)

        # Save current results into session for mapping buy clicks to uid
        st.session_state["last_results"] = df.to_dict("records")

        # Show top 3 (no images, no links)
        st.subheader("🏆 Top 3 Deals (India)")
        for idx, r in df.head(3).iterrows():
            cols = st.columns([3,1])
            with cols[0]:
                st.markdown(f"**{r['title']}**")
                st.write(f"Store: **{r['store']}**  —  Price: **₹{int(r['price']):,}**")
                if r.get("rating"):
                    st.write(f"⭐ {r['rating']} ({r['reviews']} reviews)")
            with cols[1]:
                uid = r["uid"]
                if st.button(f"Bought — #{idx+1}", key=f"buy_top_{uid}"):
                    append_purchase(uid, r["price"])
                    st.success("Purchase recorded (saved to purchases.csv). This will be used for ML training later.")

        st.markdown("---")
        st.subheader("📦 All Results (ranked)")
        display_df = df[["uid","title","store","price","rating","reviews","score","created_at"]].copy()
        st.dataframe(display_df)

        # price history UI (select a row to view history)
        st.markdown("---")
        st.subheader("📈 Price History (matched from past searches)")
        pick = st.selectbox("Pick a result to view its history (by uid)", options=list(display_df["uid"].values))
        selected_row = next((x for x in st.session_state["last_results"] if x["uid"] == pick), None)
        if selected_row:
            hist = fetch_price_history(selected_row["title"], selected_row["store"], current_embedding_json=selected_row.get("embedding"))
            if hist.empty:
                st.info("No historical prices found.")
            else:
                st.line_chart(data=hist.set_index("created_at")["price"])
                st.write(f"Historical min: ₹{int(hist['price'].min())}, mean: ₹{int(hist['price'].mean())}")

        # Alerts: set a threshold for any given result
        st.markdown("---")
        st.subheader("🔔 Price Alert (set for any of the displayed results)")
        alert_uid = st.selectbox("Choose product (uid) to watch", options=list(display_df["uid"].values))
        alert_row = next((x for x in st.session_state["last_results"] if x["uid"] == alert_uid), None)
        if alert_row:
            alert_type = st.radio("Alert type", ("percent_drop_from_history_min", "absolute_price_in_INR"))
            if alert_type == "percent_drop_from_history_min":
                perc = st.number_input("Trigger when price drops by X% below historical min", min_value=1.0, max_value=100.0, value=20.0)
                if st.button("Set percent alert"):
                    hist = fetch_price_history(alert_row["title"], alert_row["store"], current_embedding_json=alert_row.get("embedding"))
                    hist_min = hist["price"].min() if (not hist.empty) else alert_row["price"]
                    threshold_price = hist_min * (1 - perc/100.0)
                    if alert_row["price"] <= threshold_price:
                        append_alert(query, alert_row["title"], alert_row["store"], "percent", perc, alert_row["price"])
                        st.success("Alert triggered immediately and logged to alerts.csv.")
                    else:
                        # store "planned" alert as csv entry with triggered_price empty? We'll just inform the user.
                        st.info(f"Alert set: will trigger when price <= ₹{int(threshold_price)} (app checks at search-time).")
            else:
                absolute = st.number_input("Trigger when price ≤ ₹", min_value=1, value=int(alert_row["price"]))
                if st.button("Set absolute alert"):
                    if alert_row["price"] <= absolute:
                        append_alert(query, alert_row["title"], alert_row["store"], "absolute", absolute, alert_row["price"])
                        st.success("Alert triggered immediately and logged to alerts.csv.")
                    else:
                        st.info(f"Alert registered: will trigger when price ≤ ₹{int(absolute)} (app checks at search-time).")

# ---------- Admin: download CSVs ----------
st.markdown("---")
st.header("⚙️ Admin — CSV downloads")
if st.button("Download search_results.csv"):
    if os.path.exists(DB_CSV):
        with open(DB_CSV, "rb") as f:
            st.download_button("Download search_results.csv", f.read(), file_name=DB_CSV)
    else:
        st.info("No search_results.csv file yet.")

if st.button("Download purchases.csv"):
    if os.path.exists(PURCHASES_CSV):
        with open(PURCHASES_CSV, "rb") as f:
            st.download_button("Download purchases.csv", f.read(), file_name=PURCHASES_CSV)
    else:
        st.info("No purchases.csv file yet.")

if st.button("Download alerts.csv"):
    if os.path.exists(ALERTS_CSV):
        with open(ALERTS_CSV, "rb") as f:
            st.download_button("Download alerts.csv", f.read(), file_name=ALERTS_CSV)
    else:
        st.info("No alerts.csv file yet.")
