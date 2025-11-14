import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Best Deal Finder", layout="wide")

st.title("🛒 Best Deal Finder (Free Price Comparison Tool)")

SERPAPI_KEY = st.secrets["b8facb5f7c78f0f5ef30abd7a0a09c45a0ea4877"]  # Add in Streamlit Secrets

def fetch_prices(query):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "b8facb5f7c78f0f5ef30abd7a0a09c45a0ea4877": SERPAPI_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    results = []

    if "shopping_results" not in data:
        return []

    for item in data["shopping_results"]:
        store = item.get("source", "")
        price = item.get("price", "")
        link = item.get("link", "")
        title = item.get("title", "")
        rating = item.get("rating", 0)
        reviews = item.get("reviews", 0)
        thumbnail = item.get("thumbnail", "")

        try:
            price_value = float(price.replace("₹", "").replace(",", "").strip())
        except:
            continue

        results.append({
            "title": title,
            "store": store,
            "price": price_value,
            "rating": rating,
            "reviews": reviews,
            "link": link,
            "image": thumbnail
        })

    return results


def rank_results(results):
    if not results:
        return []

    df = pd.DataFrame(results)

    df["score"] = (
        (df["price"].max() - df["price"]) * 0.6 +    # lower price = better
        (df["rating"].fillna(0)) * 0.3 +             # rating helpful
        (df["reviews"].fillna(0)) * 0.1              # more reviews = trust
    )

    df = df.sort_values(by="score", ascending=False)
    return df


query = st.text_input("Enter product name:", placeholder="Example: iPhone 13, earphones, laptop stand")

if query:
    with st.spinner("Fetching prices... please wait ⏳"):
        data = fetch_prices(query)

    if not data:
        st.error("No data found. Try another product.")
    else:
        ranked = rank_results(data)

        st.subheader("🏆 Top 3 Best Deals")
        top3 = ranked.head(3)

        for _, row in top3.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                if row["image"]:
                    st.image(row["image"], width=120)
            with col2:
                st.markdown(f"### {row['title']}")
                st.markdown(f"**Store:** {row['store']}")
                st.markdown(f"**Price:** ₹{row['price']:,}")
                st.markdown(f"[🔗 View Deal]({row['link']})")

            st.markdown("---")

        st.subheader("📦 All Results")
        st.dataframe(ranked[["title", "store", "price", "rating", "reviews", "link"]])
