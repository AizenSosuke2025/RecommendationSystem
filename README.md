# 🍿 Netflix Prize Recommendation Engine: A Hybrid Deep Learning Architecture

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)
![SciPy](https://img.shields.io/badge/SciPy-Matrix_Factorization-8CAAE6.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)

## 📌 Overview
This repository contains a full-scale recommendation system built on the legendary **Netflix Prize Dataset** (100M+ ratings). The goal of this project is to solve the problem of information overload by predicting user movie affinities and generating personalized Top-K lists. 

The primary technical challenge addressed in this project is the **Cold Start Problem** and extreme matrix sparsity. To handle this, the system utilizes a fault-tolerant **"Waterfall Architecture,"** routing data through advanced Deep Learning models, Collaborative Filtering matrices, and Content-Based text engines to ensure 100% prediction coverage without system crashes.

---

## 🏗️ System Architecture: The Waterfall Pipeline

Standard Collaborative Filtering models mathematically collapse when encountering brand-new users or movies (Cold Starts). This system circumvents that by using a multi-tiered fallback pipeline during inference:

* **Tier 1: The Hybrid Ensemble (High Accuracy)**
  * Uses a weighted blend of **Truncated SVD** (Matrix Factorization) to capture linear latent concepts and **Neural Collaborative Filtering (NCF)** via PyTorch to capture complex, non-linear user-item interactions.
  * *Trigger:* Used when both the user and movie exist in the training history.
* **Tier 2: Content-Based Filtering (Cold-Start Resilience)**
  * Uses **TF-IDF Vectorization** and Cosine Similarity on movie metadata. If a user asks for a prediction on a brand-new movie, the system analyzes the text profile of their historical favorites to calculate a match.
  * *Trigger:* Used when a movie is unknown to the Collaborative models.
* **Tier 3: Global Popularity (The Ultimate Safety Net)**
  * If a brand-new user signs up with absolutely zero history, the system dynamically fills their UI with globally trending, highly-rated items to bootstrap their profile.
  * *Trigger:* Total absence of user data.

---

## 📊 Dataset & Preprocessing
The model is trained on the official Netflix Prize dataset. 
* **Scale:** Over 100,000,000 individual ratings across 17,000+ movies.
* **Sparsity:** ~99% sparse matrix.
* **Memory Management:** Because 100M rows exceed standard RAM limits, the data ingestion pipeline utilizes chunked streaming and localized dictionary grouping for `O(1)` memory lookups.

*(Note: Due to size constraints, the dataset is not hosted in this repository. Download it from Kaggle and place it in the `NetflixPrizeDataset/` directory).*

---

## 🚀 How to Run & Reproduce

### 1. Environment Setup
This pipeline was designed to be run in **Google Colab** to utilize free GPU acceleration for the PyTorch NCF model.
1. Upload the dataset to your Google Drive.
2. Clone this repository into your Colab environment or run the notebooks directly.

### 2. Execution
The main pipeline is contained in the unified script. Once the environment is initialized and models are trained, you can interact with the engine using two primary functions:

```python
# Predict a specific user's rating for a specific movie (1-5 Stars)
predicted_score = predict_rating_waterfall(
    user_id=1488844, 
    movie_id=345, 
    svd_matrix=svd_matrix, 
    ncf_model=ncf_model
)

# Generate a Top-10 personalized recommendation list
recommended_movies = get_top_k_recommendations_waterfall(
    user_id=1488844, 
    k=10, 
    svd_matrix=svd_matrix, 
    ncf_model=ncf_model
)
