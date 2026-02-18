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

