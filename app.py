import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
import json
import numpy as np
import uuid
import base64
import traceback

try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

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

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

SERPAPI_KEY = ""
if hasattr(st, "secrets"):
    SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
if not SERPAPI_KEY:
    SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

CURRENT_CSV = "current_search.csv"   
ALL_RESULTS_CSV = "all_results.csv"  
PURCHASES_CSV = "purchases.csv"      
MIN_LABELS_TO_TRAIN = 10
USD_TO_INR_API = "https://api.exchangerate-api.com/v4/latest/USD"

CSV_PASSWORD = "021202"

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
    if "$" in raw_price or "USD" in raw_price or "US$" in raw_price:
        nums = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
        if not nums:
            return None
        return float(nums) * USD_INR
    filtered = "".join(ch for ch in s if (ch.isdigit() or ch == "."))
    if filtered == "":
        return None
    price = float(filtered)
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

def title_token_match(a, b):
    if not a or not b:
        return False
    a_tokens = set([t.lower() for t in a.split() if len(t) > 2])
    b_tokens = set([t.lower() for t in b.split() if len(t) > 2])
    if not a_tokens or not b_tokens:
        return False
    overlap = a_tokens.intersection(b_tokens)
    return len(overlap) >= max(1, min(3, int(0.4 * min(len(a_tokens), len(b_tokens)))))


def derive_key(password: str, salt: bytes, iterations: int = 200_000):
    if not CRYPTO_AVAILABLE:
        return None
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
        backend=default_backend()
    )
    return kdf.derive(password.encode("utf-8"))

def encrypt_bytes(password: str, plaintext_bytes: bytes) -> bytes:
    """
    Returns: header (b'CSVENC1') + salt(16) + iv(16) + ciphertext
    """
    if not CRYPTO_AVAILABLE:
        return None
    salt = os.urandom(16)
    key = derive_key(password, salt)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    pad_len = 16 - (len(plaintext_bytes) % 16)
    padded = plaintext_bytes + bytes([pad_len]) * pad_len
    ct = encryptor.update(padded) + encryptor.finalize()
    header = b"CSVENC1"
    return header + salt + iv + ct
def append_all_results(rows, search_query):
    """
    rows: list of dicts with fields:
      uid,title,store,price,rating,reviews,score,embedding,created_at,search_query
    Appends to ALL_RESULTS_CSV (creates if missing).
    """
    df = pd.DataFrame(rows)
    df["search_query"] = search_query
    file_exists = os.path.exists(ALL_RESULTS_CSV)
    df.to_csv(ALL_RESULTS_CSV, mode="a", header=not file_exists, index=False)

def save_current_search_csv(rows):
    """
    rows: list of dicts — write current_search CSV (overwrite).
    """
    df = pd.DataFrame(rows)
    df.to_csv(CURRENT_CSV, index=False)

def load_all_results_df():
    if not os.path.exists(ALL_RESULTS_CSV):
        return pd.DataFrame(columns=["uid","title","store","price","rating","reviews","score","embedding","created_at","search_query"])
    try:
        return pd.read_csv(ALL_RESULTS_CSV)
    except Exception:
        return pd.DataFrame(columns=["uid","title","store","price","rating","reviews","score","embedding","created_at","search_query"])
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

def fetch_price_history_for(uid, title, store, current_embedding_json=None):
    df_all = load_all_results_df()
    if df_all.empty:
        return pd.DataFrame(columns=["price","created_at"])
    df_store = df_all[df_all["store"].str.contains(store, case=False, na=False)]
    matches = []
    cur_emb = None
    if current_embedding_json:
        try:
            cur_emb = np.array(json.loads(current_embedding_json), dtype=float)
        except Exception:
            cur_emb = None
    for _, r in df_store.iterrows():
        matched = False
        if cur_emb is not None and r.get("embedding"):
            try:
                past = np.array(json.loads(r["embedding"]), dtype=float)
                if np.linalg.norm(past) != 0:
                    sim = float(np.dot(cur_emb, past) / (np.linalg.norm(cur_emb) * np.linalg.norm(past)))
                else:
                    sim = -1.0
                if sim >= EMBED_THRESHOLD:
                    matched = True
            except Exception:
                matched = False
        if not matched:
            if title_token_match(str(r.get("title","")), title):
                matched = True
        if matched:
            matches.append((r.get("price"), r.get("created_at")))
    if not matches:
        # fallback return recent prices for store
        try:
            df_store["created_at"] = pd.to_datetime(df_store["created_at"])
            df_store = df_store.sort_values("created_at")
            return df_store[["price","created_at"]]
        except Exception:
            return pd.DataFrame(columns=["price","created_at"])
    dfm = pd.DataFrame(matches, columns=["price","created_at"])
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

def try_train_and_predict_from_csv(rows_feature_df):
    if not LGB_AVAILABLE:
        return None
    if not os.path.exists(ALL_RESULTS_CSV) or not os.path.exists(PURCHASES_CSV):
        return None
    try:
        df_all = pd.read_csv(ALL_RESULTS_CSV)
        df_p = pd.read_csv(PURCHASES_CSV)
    except Exception:
        return None
    df_all["bought"] = df_all["uid"].isin(df_p["uid"]).astype(int)
    df_train = df_all[["price","rating","reviews","bought"]].dropna()
    if df_train.shape[0] < MIN_LABELS_TO_TRAIN:
        return None
    X = df_train[["price","rating","reviews"]]
    y = df_train["bought"]
    dtrain = lgb.Dataset(X, label=y)
    params = {"objective":"binary","metric":"binary_logloss","verbosity": -1}
    model = lgb.train(params, dtrain, num_boost_round=100)
    X_new = rows_feature_df[["price","rating","reviews"]].fillna(0)
    preds = model.predict(X_new)
    return preds

st.set_page_config(page_title="RecomBuddy", layout="wide")
st.title("RecomBuddy 🇮🇳")

if not SERPAPI_KEY:
    st.error("SERPAPI_KEY required. Add to Streamlit Secrets or set SERPAPI_KEY env var.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    selected_stores = st.multiselect("Filter stores (leave empty = all Indian stores)", options=INDIAN_STORES)
    st.markdown("---")
    st.write("Embeddings: " + ("available" if EMBED_AVAILABLE else "not available (token-match fallback)"))
    if EMBED_AVAILABLE:
        st.write(f"Embedding dim: {EMBED_DIM}, threshold: {EMBED_THRESHOLD}")
    st.markdown("---")
    st.write(f"USD→INR: {USD_INR:.2f} (live)")

query = st.text_input("Enter product name (India only)", placeholder="Jo Dhundhoge woh milega")

if query:
    with st.spinner("Fetching India results via SerpAPI..."):
        rows = fetch_serpapi(query, selected_stores)
    if not rows:
        st.error("No valid India product results found.")
    else:
        feature_df = pd.DataFrame(rows)[["price","rating","reviews"]]
        preds = try_train_and_predict_from_csv(feature_df) if not feature_df.empty else None
        if preds is not None:
            for i, p in enumerate(preds):
                rows[i]["score"] = float(p)
            df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        else:
            df = heuristic_rank(rows).reset_index(drop=True)
        save_current_search_csv(df.to_dict("records"))
        append_all_results(df.to_dict("records"), query)
        st.session_state["last_results"] = df.to_dict("records")

        st.subheader("🏆 Top 3 Deals (India)")
        for idx, row in df.head(3).iterrows():
            cols = st.columns([3,1])
            with cols[0]:
                st.markdown(f"**{row['title']}**")
                st.write(f"Store: **{row['store']}** — Price: **₹{int(row['price']):,}**")
                if row.get("rating"):
                    st.write(f"⭐ {row['rating']} ({row['reviews']} reviews)")
            with cols[1]:
                uid = row["uid"]
                if st.button(f"Bought — #{idx+1}", key=f"buy_top_{uid}"):
                    if not os.path.exists(PURCHASES_CSV):
                        pd.DataFrame([{"uid": uid, "bought_price": row["price"], "created_at": datetime.utcnow().isoformat()}]) \
                          .to_csv(PURCHASES_CSV, mode="a", header=True, index=False)
                    else:
                        pd.DataFrame([{"uid": uid, "bought_price": row["price"], "created_at": datetime.utcnow().isoformat()}]) \
                          .to_csv(PURCHASES_CSV, mode="a", header=False, index=False)
                    st.success("Purchase recorded to purchases.csv")

        st.markdown("---")
        st.subheader("📦 All Results (ranked)")
        display_df = df[["uid","title","store","price","rating","reviews","score","created_at"]].copy()
        st.dataframe(display_df)
        st.markdown("---")
        st.subheader("📈 Price History (from cumulative dataset)")
        pick = st.selectbox("Select a result (uid) to view its history", options=list(display_df["uid"].values))
        selected_row = next((x for x in st.session_state["last_results"] if x["uid"] == pick), None)
        if selected_row:
            hist_df = fetch_price_history_for(pick, selected_row["title"], selected_row["store"], current_embedding_json=selected_row.get("embedding"))
            if hist_df.empty:
                st.info("No historical rows matched for this product.")
            else:
                st.line_chart(data=hist_df.set_index("created_at")["price"])
                st.write(f"Historical min: ₹{int(hist_df['price'].min())}, mean: ₹{int(hist_df['price'].mean())}")
        st.markdown("---")
        st.subheader("⬇️ Download CSVs (password protected)")

        try:
            with open(CURRENT_CSV, "rb") as f:
                current_bytes = f.read()
        except Exception:
            current_bytes = None

        if current_bytes:
            if CRYPTO_AVAILABLE:
                enc_current = encrypt_bytes(CSV_PASSWORD, current_bytes)
                st.download_button("Download current_search.csv (encrypted)", enc_current, file_name="current_search.csv.enc")
            else:
                st.warning("cryptography not installed — downloading current_search.csv unencrypted.")
                st.download_button("Download current_search.csv (unencrypted)", current_bytes, file_name="current_search.csv")

        try:
            with open(ALL_RESULTS_CSV, "rb") as f:
                all_bytes = f.read()
        except Exception:
            all_bytes = None

        if all_bytes:
            if CRYPTO_AVAILABLE:
                enc_all = encrypt_bytes(CSV_PASSWORD, all_bytes)
                st.download_button("Download all_results.csv (encrypted)", enc_all, file_name="all_results.csv.enc")
            else:
                st.warning("cryptography not installed — downloading all_results.csv unencrypted.")
                st.download_button("Download all_results.csv (unencrypted)", all_bytes, file_name="all_results.csv")
