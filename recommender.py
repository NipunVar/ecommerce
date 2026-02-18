import pickle
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

als_model = pickle.load(open("als_model.pkl", "rb"))
user_item_matrix = pickle.load(open("user_item_matrix.pkl", "rb"))
user_to_index = pickle.load(open("user_to_index.pkl", "rb"))
index_to_product = pickle.load(open("index_to_product.pkl", "rb"))

user_item_matrix = user_item_matrix.tocsr()

train_df = pd.read_parquet("train.parquet")
train_df["price"] = pd.to_numeric(train_df["price"], errors="coerce")

product_metadata = (
    train_df
    .groupby("product_id")
    .agg({
        "brand": "first",
        "price": "mean",
        "cat_0": "first"
    })
    .reset_index()
)

product_metadata["price"] = product_metadata["price"].fillna(0)
product_metadata = product_metadata.set_index("product_id")

popularity_scores = (
    train_df
    .groupby("product_id")
    .size()
    .reset_index(name="popularity")
)

popularity_scores["popularity"] = (
    popularity_scores["popularity"] /
    popularity_scores["popularity"].max()
)

popularity_dict = dict(
    zip(popularity_scores["product_id"],
        popularity_scores["popularity"])
)


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

        brand = "Unknown"
        category = "Unknown"
        price = 0

        if product_id in product_metadata.index:
            meta = product_metadata.loc[product_id]
            brand = meta["brand"]
            category = meta["cat_0"]
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
    return train_df["cat_0"].value_counts().idxmax()
