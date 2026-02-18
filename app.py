import streamlit as st
import pandas as pd
import plotly.express as px
from recommender import recommend_hybrid, user_to_index

st.set_page_config(
    page_title="E-Commerce Recommendation System",
    layout="wide",
    page_icon="🛍️"
)


st.markdown("""
<style>
.big-title {
    font-size:32px !important;
    font-weight:700;
}
.metric-card {
    background-color:#111827;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)


train_df = pd.read_parquet("train.parquet")
train_df["price"] = pd.to_numeric(train_df["price"], errors="coerce")

product_info = (
    train_df
    .groupby("product_id")
    .agg({
        "brand": "first",
        "cat_0": "first",
        "price": "mean"
    })
    .reset_index()
)

product_info = product_info.set_index("product_id")

total_users = train_df["user_id"].nunique()
avg_price = train_df["price"].mean()

most_bought_product_id = train_df["product_id"].value_counts().idxmax()
most_bought_product_name = product_info.loc[most_bought_product_id]["brand"]

most_bought_category = train_df["cat_0"].value_counts().idxmax()


st.sidebar.title(" Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Overview", "Recommendations", "Analytics"]
)



if page == "Overview":

    st.markdown('<p class="big-title">🛍️ E-Commerce Recommendation System</p>', unsafe_allow_html=True)
    st.header("Project Overview")

    st.markdown("""
This project implements a scalable hybrid recommendation system designed for modern e-commerce platforms. The model leverages collaborative filtering using Alternating Least Squares (ALS) combined with a popularity-based ranking strategy to deliver stable and personalized recommendations. The recommendation engine constructs a sparse user-item interaction matrix from implicit behavioral signals such as product views, cart interactions, and purchases. Matrix factorization decomposes this matrix into latent user and product embeddings, allowing the system to identify hidden behavioral patterns even in the absence of explicit ratings. To enhance robustness and mitigate cold-start challenges, a popularity baseline is incorporated into the hybrid scoring mechanism. This ensures high-engagement products maintain visibility while preserving personalization quality. The dataset contains thousands of users and products with engineered temporal and categorical features. The final hybrid system balances personalization and global engagement signals, improving recall, stability, and diversity in recommendations.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", f"{total_users:,}")
    col2.metric("Average Product Price", f"₹{avg_price:,.2f}")
    col3.metric("Most Bought Category", most_bought_category)

    st.subheader(" Most Bought Product")

   
    st.markdown(
        f"[{most_bought_product_name}](https://www.google.com/search?q={most_bought_product_name})",
        unsafe_allow_html=True
    )


elif page == "Recommendations":

    st.markdown('<p class="big-title">🎯 Hybrid Model Recommendations</p>', unsafe_allow_html=True)

    user_ids = list(user_to_index.keys())
    selected_user = st.selectbox("Select User ID", user_ids)

    category_filter = st.selectbox(
        "Filter by Category (Optional)",
        ["All"] + sorted(train_df["cat_0"].dropna().unique().tolist())
    )

    if st.button("Generate Recommendations"):

        recommendations = recommend_hybrid(selected_user, 10)

        if recommendations:

            df_recs = pd.DataFrame(recommendations)

            df_recs["price"] = pd.to_numeric(df_recs["price"], errors="coerce")
            df_recs["score"] = pd.to_numeric(df_recs["score"], errors="coerce")

            if category_filter != "All":
                df_recs = df_recs[df_recs["category"] == category_filter]

            st.download_button(
                "Download Recommendations CSV",
                df_recs.to_csv(index=False),
                file_name="recommendations.csv"
            )

            st.subheader("Top Recommendations")

            cols = st.columns(2)

            for i, row in df_recs.iterrows():

                col = cols[i % 2]

                with col:

                    rank = i + 1
                    confidence = min(max(row["score"] * 100, 0), 100)

                    st.markdown(f"### #{rank} {row.get('brand', 'Product')}")

                    
                    st.markdown(
                        f"[View Product](https://www.google.com/search?q={row.get('brand')})"
                    )

                    st.write(f"Category: {row.get('category', 'N/A')}")
                    st.write(f"Price: ₹{row.get('price', 0):,.2f}")
                    st.write(f"Score: {row.get('score', 0):.4f}")

                    st.progress(confidence / 100)

        else:
            st.warning("No recommendations found.")

elif page == "Analytics":

    st.markdown('<p class="big-title">📊 Analytics Dashboard</p>', unsafe_allow_html=True)

    st.subheader("Top 10 Most Bought Products")

    top_products = (
        train_df["product_id"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_products.columns = ["product_id", "count"]

    top_products["product_name"] = top_products["product_id"].map(
        product_info["brand"]
    )


    fig1 = px.pie(
        top_products,
        names="product_name",
        values="count",
        hole=0.5,  
        color_discrete_sequence=px.colors.sequential.Blues_r,
        title="Top 10 Most Bought Products"
    )

    fig1.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Purchases: %{value}<br>Percentage: %{percent}"
    )

    fig1.update_layout(
        showlegend=True,
        legend_title="Products",
        title_x=0.3
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Category Distribution")

    category_dist = train_df["cat_0"].value_counts().head(10).reset_index()
    category_dist.columns = ["category", "count"]

    fig2 = px.bar(
        category_dist,
        x="category",
        y="count",
        color="count",
        color_continuous_scale="blues"
    )

    fig2.update_layout(
        xaxis_title="Category",
        yaxis_title="Purchase Count"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Price Distribution")

    fig3 = px.histogram(
        train_df,
        x="price",
        nbins=40,
        color_discrete_sequence=["#3b82f6"]
    )

    fig3.update_layout(
        xaxis_title="Price",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig3, use_container_width=True)
