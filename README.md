# 🇮🇳 RecomBuddy 

🔗 **Live App:** https://recombuddy.streamlit.app/

RecomBuddy is a **zero-cost**, India-only **price comparison system** that fetches real-time prices from Indian ecommerce sites using **SerpAPI (Google Shopping)**.  
It ranks products, stores price history, encrypts downloads, and works perfectly on Streamlit Cloud.

✔ No database  
✔ Two password-protected CSVs  
✔ India-only store filtering  
✔ Price-history tracking  
✔ Optional ML ranking  
✔ Optional embeddings  
✔ Live deployed app  

---

## 🚀 Features

### 🇮🇳 India-only price fetching
- Filters strictly to Indian stores
- Removes EMI / per-month prices
- Converts USD → INR automatically
- Filters out unrealistic cheap accessories

### 🏆 Deal Ranking
- Heuristic scoring (price + rating + reviews)
- Optional ML model (LightGBM) when purchases exist

### 🧠 Intelligent Matching
- Embedding-based (sentence-transformers) if installed  
- Token-overlap fallback  
- Works even when product titles differ

### 📈 Price History Tracking
- Every search appends to **all_results.csv**
- Future searches detect and match same product
- Line graph shows how price changed over time

### 🔐 CSV Encryption (Password Protected)
| File | Description |
|------|-------------|
| `current_search.csv.enc` | Latest search results |
| `all_results.csv.enc` | Full historical dataset |
