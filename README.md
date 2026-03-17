# 📰 News Multi-Classification System

[](https://huggingface.co/spaces/Zainch032/News-Classification)


## 🚀 The Problem: Information Overload

In the modern digital age, thousands of news articles are published every minute. Manually sorting these into relevant categories for news aggregators or personalized feeds is slow, expensive, and prone to human error.

## 🎯 The Solution: Automated Categorization

This project provides an **end-to-end NLP pipeline** that instantly classifies raw text into four core sectors: **World, Business, Sports, and Technology**. By automating this process, platforms can achieve:

  * **Real-time content routing** to specific app sections.
  * **Improved user experience** through organized news feeds.
  * **Data-driven insights** into trending topics across industries.

-----

## 🧠 End-to-End ML Pipeline

I designed the system to handle the journey from **noisy, raw text** to **calibrated probability scores**.

### 1\. Advanced Text Preprocessing

Standard cleaning wasn't enough; to reach high accuracy, I implemented a custom pipeline:

  * **Noise Reduction:** Stripping HTML, punctuation, and numeric tokens.
  * **Normalization:** Lowercasing and Snowball Stemming to reduce words to their root form (e.g., "running" → "run").
  * **Stopword Filtering:** Leveraging Scikit-learn's optimized list to remove non-informative high-frequency words.

### 2\. Feature Engineering & Selection

  * **TF-IDF Vectorization:** Extracted 5,000 top-performing features to represent document importance rather than just word counts.
  * **Sparse Mapping:** Utilized sparse matrix representations to ensure the model remains lightweight and fast during inference.

### 3\. Model Benchmarking

I didn't just pick a model; I conducted a **Grid Search/Tournament** of 8 different classifiers. While **LinearSVC** does not natively support probabilities, it provided the best F1-score. To solve the "confidence" issue, I applied a **Softmax function** over the SVC decision scores.

| Model | Performance | Status |
| :--- | :--- | :--- |
| **LinearSVC** | **Highest Weighted F1** | ✅ **Selected** |
| Multinomial NB | Fast, but lower precision | ❌ |
| Logistic Regression | Reliable, but slower inference | ❌ |
| SGD Classifier | Good for big data, overkill here | ❌ |

-----

## 🧪 Tech Stack & Deployment

  * **Core Logic:** `Python`, `Scikit-learn`, `NLTK`, `Pandas`
  * **Visualization:** `Seaborn` & `Matplotlib` (for class distribution & word frequency EDA)
  * **Inference Engine:** `Flask` & `Gunicorn`
  * **Containerization:** `Docker` (ensuring "it works on my machine" translates to "it works in production")
  * **Hosting:** `Hugging Face Spaces`

-----

## 📊 Key Results

  * **Accuracy:** Optimized for high precision in the 'Technology' and 'Business' sectors.
  * **Scalability:** Containerized with Docker, making it ready to be deployed on AWS/GCP/Azure.
  * **UX:** Integrated a custom-built web UI that shows not just the category, but the **confidence percentage** for transparency.

-----

## 👤 About the Author

**Muhammad Zain** *Final Year Data Science Student | Lahore, Pakistan*

  * [**LinkedIn**](https://linkedin.com/in/muhammad-zain-9710692b4)
  * [**GitHub**](https://github.com/Zainch032)

-----
