🌟 RecomBuddy — India’s Smartest Price Comparison & Deal Finder
Get the best prices across Indian e-commerce instantly — ranked, clean, and powered by ML.

RecomBuddy is a zero-cost, India-focused price comparison engine built with Streamlit + SerpAPI + lightweight ML.
Just enter a product name (like iPhone 15, JBL Speaker, Smart TV), and RecomBuddy will:

🔍 Fetch real-time prices from all major Indian stores

⚖️ Rank them using a built-in LightGBM-powered scoring system

📈 Generate historical price graphs from past searches

📁 Export encrypted CSVs for analysis

🧠 Use embeddings (optional) for smarter product matching

🛒 Track purchases and improve recommendations over time

🚀 Live Demo

👉 (Add your Streamlit Cloud link here)
Example: https://recombuddy.streamlit.app/

🧠 Why RecomBuddy?

Because India doesn’t have a good transparent, cross-platform price comparison tool—most apps:

❌ Don’t show real prices
❌ Don’t track price history
❌ Don’t support all Indian stores
❌ Don’t let you export real data
❌ Don’t use ML to improve ranking

RecomBuddy fixes all of that.

✨ Features
🔍 1. India-Only Price Search (Accurate & Clean)

Fetches from Amazon, Flipkart, Croma, Reliance Digital, TataCliq, Myntra, Ajio and 200+ Indian stores.

Filters out:

accessories

EMI prices

fake low prices

foreign stores

Converts USD → INR when needed.

🧠 2. Smart Ranking (ML + Heuristics)

RecomBuddy ranks deals using:

Price score

Rating score

Review-weighting

ML preference learning (auto-trains using your purchase CSV)

If ML is not ready, it falls back to a powerful heuristic model.

📈 3. Historical Price Graphs

Every search you run gets saved (encrypted) into a master CSV.
If someone searches the same item later — RecomBuddy matches it using:

Sentence-transformer embeddings (if installed)

or Token-based matching fallback

👉 Shows complete price history trends!
Perfect for price tracking or making deal predictions.

🗂️ 4. Two Encrypted CSV Exports

RecomBuddy exports:

current_search.csv.enc

Only the latest search results

Fully ranked

AES-encrypted

Password: 021202

all_results.csv.enc

Full historical dataset

All searches since day one

AES-encrypted

Password: 021202

Encryption uses PBKDF2 + AES-CBC for maximum safety.

🛒 5. Purchase Logging (Optional)

Click Bought on any deal →
purchases.csv stores that UID + price.

Over time, RecomBuddy learns your preferences 🔥
(Perfect for building a personalized recommender later.)

🏗️ Tech Stack
Layer	Tech
UI	Streamlit
Data Fetch	SerpAPI + Google Shopping
ML	LightGBM, Embeddings (optional)
Storage	Two CSV files (encrypted)
Deployment	Streamlit Cloud
Encryption	cryptography (AES-CBC)
