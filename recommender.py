import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

als_model = pickle.load(open("als_model.pkl", "rb"))
user_item_matrix = pickle.load(open("user_item_matrix.pkl", "rb"))
user_to_index = pickle.load(open("user_to_index.pkl", "rb"))
index_to_product = pickle.load(open("index_to_product.pkl", "rb"))

user_item_matrix = user_item_matrix.tocsr()

product_metadata = pd.read_csv("product_metadata.csv")
product_popularity = pd.read_csv("product_popularity.csv")

product_metadata["price"] = pd.to_numeric(product_metadata["price"], errors="coerce")
product_metadata["price"] = product_metadata["price"].fillna(0)

product_metadata = product_metadata.set_index("product_id")


product_popularity["popularity"] = (
    product_popularity["purchase_count"] /
    product_popularity["purchase_count"].max()
)

popularity_scores = product_popularity



def recommend_als(user_id, n=10):

    if user_id not in user_to_index:
        return []

    user_index = user_to_index[user_id]

    ids, scores = als_model.recommend(
        userid=user_index,
        user_items=user_item_matrix[user_index],
        N=n,
        filter_already_liked_items=True
    )

    results = []

    for item_index, score in zip(ids, scores):

        product_id = index_to_product.get(item_index)

        if product_id is None:
            continue

        results.append({
            "product_id": product_id,
            "score": float(score)
        })

    return results


def recommend_popular(n=10):

    top_items = popularity_scores.sort_values(
        "popularity",
        ascending=False
    ).head(n)

    results = []

    for _, row in top_items.iterrows():
        results.append({
            "product_id": row["product_id"],
            "score": float(row["popularity"])
        })

    return results


def recommend_hybrid(user_id, n=10, alpha=0.7):

    als_recs = recommend_als(user_id, n*3)
    popular_recs = recommend_popular(n*3)

    combined_scores = {}

    for item in als_recs:
        combined_scores[item["product_id"]] = alpha * item["score"]

    for item in popular_recs:
        if item["product_id"] in combined_scores:
            combined_scores[item["product_id"]] += (1 - alpha) * item["score"]
        else:
            combined_scores[item["product_id"]] = (1 - alpha) * item["score"]

    sorted_items = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    final_results = []

    for product_id, score in sorted_items:

        try:
            product_id = int(product_id)
        except:
            continue

        brand = f"Product {product_id}"
        category = "Unknown"
        price = 0

        if product_id in product_metadata.index:
            meta = product_metadata.loc[product_id]

            if pd.notna(meta.get("brand")):
                brand = meta["brand"]

            if pd.notna(meta.get("cat_0")):
                category = meta["cat_0"]

            if pd.notna(meta.get("price")):
                price = meta["price"]

        final_results.append({
            "product_id": product_id,
            "brand": brand,
            "category": category,
            "price": float(price),
            "score": float(score)
        })

    return final_results



def get_total_users():
    return len(user_to_index)


def get_average_price():
    return float(product_metadata["price"].mean())


def get_most_popular_product():
    return popularity_scores.sort_values(
        "popularity",
        ascending=False
    ).iloc[0]["product_id"]


def get_most_popular_category():
    return product_metadata["cat_0"].mode()[0]

