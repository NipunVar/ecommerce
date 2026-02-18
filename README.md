# 🛍️ E-Commerce Hybrid Recommendation System

A scalable end-to-end Hybrid Recommendation System built using **Implicit Alternating Least Squares (ALS)** and a **Popularity-Based Baseline**, deployed with an interactive **Streamlit Web Application**.

This project demonstrates:

- Implicit feedback modeling
- Sparse matrix optimization
- Hybrid recommendation logic
- Offline evaluation (Recall@K & Precision@K)
- Production-ready dashboard

---

## 🚀 Project Overview

Modern e-commerce platforms require recommendation systems that balance:

- 🎯 Personalization (Collaborative Filtering)
- 🔥 Global Popularity Trends
- ⚡ Scalability for Large Data
- 🧠 Cold-Start Handling

This system combines ALS collaborative filtering with a popularity baseline to deliver stable and personalized product recommendations.

---

## 🧠 Recommendation Architecture

### 1️⃣ Implicit Feedback Modeling

User interactions are weighted:

| Event Type | Weight |
|------------|--------|
| Cart       | 1      |
| Purchase   | 5      |

Then log-scaled:

weight = log1p(interaction_weight)


This boosts purchase importance and stabilizes training.

---

### 2️⃣ Time-Based Train-Test Split

For each user:
- All interactions except the last → Train
- Last interaction → Test

Users with fewer than 2 interactions are removed.

This ensures realistic offline evaluation.

---

### 3️⃣ Sparse Matrix Construction

- CSR Matrix (SciPy)
- Float32 optimized
- Zero-interaction items removed

This enables scalability for large datasets.

---

### 4️⃣ ALS Model (Implicit Library)

Parameters:

- Factors: 32
- Regularization: 0.1
- Iterations: 10
- Alpha (confidence scaling): 40
- Random State: 42

Confidence matrix:

confidence_matrix = interaction_matrix × alpha


ALS factorizes the matrix into user and product latent embeddings.

---

### 5️⃣ Popularity Baseline

Global ranking of products based on total interaction weight.

Used for:
- Cold-start users
- Stability improvement
- Hybrid blending

---

### 6️⃣ Hybrid Model

Final blended score:

Final Score = 0.6 × Normalized ALS Score
+ 0.4 × Normalized Popularity Score


This balances personalization and global engagement.

---

## 📊 Offline Evaluation

Metric:
- Recall@10
- Precision@10

### 🔹 ALS Model
- Recall@10: 0.0580  
- Precision@10: 0.00580  

### 🔹 Popularity Baseline
- Recall@10: 0.1490  
- Precision@10: 0.0149  

### 🔹 Hybrid Model
- Recall@10: 0.0607  
- Precision@10: 0.00607  

### 📈 Interpretation

- Popularity baseline performs strongly due to global trends.
- ALS captures personalization patterns.
- Hybrid model balances personalization and robustness.

---

## 🖥️ Streamlit Web Application

### 🏠 Overview Page
- Total users
- Average product price
- Most bought product (clickable)
- Most popular category

### 🎯 Recommendations Page
- Select User ID
- Optional category filter
- Top 10 hybrid recommendations
- Download recommendations as CSV

### 📊 Analytics Dashboard
- Donut chart (Top 10 products)
- Percentage labels inside slices
- Category distribution
- Price histogram
- Interactive hover tooltips

---

## 📂 Project Structure

Ecommerce/
│
├── app.py
├── recommender.py
├── train_model.py
├── project.ipynb
├── train.parquet
│
├── als_model.pkl
├── user_item_matrix.pkl
├── user_to_index.pkl
├── index_to_item.pkl
│
├── README.md
├── requirements.txt


---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- SciPy (Sparse Matrices)
- Implicit (ALS)
- Scikit-learn
- Plotly
- Streamlit
- PyArrow

---

## 🔮 Future Improvements

- Hyperparameter tuning
- MAP@K & NDCG evaluation
- Real-time recommendation API
- Docker deployment
- Cloud hosting (AWS / Streamlit Cloud)
- User segmentation

---

## 👤 Author

Nipun Varshneya
- LinkedIn: https://www.linkedin.com/in/nipun-varshneya-5983b0358/
- GitHub: https://github.com/NipunVar

---

