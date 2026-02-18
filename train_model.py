import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

print("Loading data...")
df = pd.read_parquet("train.parquet")


MAX_USERS = 50000
users_sample = df["user_id"].unique()[:MAX_USERS]
df = df[df["user_id"].isin(users_sample)]

print("Users after sampling:", df["user_id"].nunique())
print("Items after sampling:", df["product_id"].nunique())


unique_products = df["product_id"].unique()
product_names = {pid: f"Product {pid}" for pid in unique_products}


user_ids = df["user_id"].unique()
item_ids = df["product_id"].unique()

user_to_index = {u: i for i, u in enumerate(user_ids)}
index_to_user = {i: u for u, i in user_to_index.items()}

item_to_index = {p: i for i, p in enumerate(item_ids)}
index_to_item = {i: p for p, i in item_to_index.items()}

rows = df["user_id"].map(user_to_index)
cols = df["product_id"].map(item_to_index)
data = np.ones(len(df))

user_item_matrix = csr_matrix(
    (data, (rows, cols)),
    shape=(len(user_ids), len(item_ids))
)

model = AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=10
)

model.fit(user_item_matrix)

print("Model trained successfully.")


popularity = df["product_id"].value_counts()
popular_products = list(popularity.index[:100])


pickle.dump(model, open("als_model.pkl", "wb"))
pickle.dump(user_item_matrix, open("user_item_matrix.pkl", "wb"))
pickle.dump(user_to_index, open("user_to_index.pkl", "wb"))
pickle.dump(index_to_item, open("index_to_item.pkl", "wb"))
pickle.dump(popular_products, open("popular_products.pkl", "wb"))
pickle.dump(product_names, open("product_names.pkl", "wb"))

print("All files saved successfully.")
