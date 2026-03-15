---
title: News Classification
emoji: 📰
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# 📰 News Multi-Classification System

> End-to-end NLP pipeline that classifies raw news articles into World, Business, Sports and Technology in real time.

## 🚀 Live Demo
Try the deployed app here: [News Classification – Hugging Face Space](https://huggingface.co/spaces/Zainch032/News-Classification)

---

## 🎯 What This Project Does

Takes any raw news article as input and predicts its category — **World, Business, Sports, or Technology** — using a trained ML model with real-time confidence scores.

---

## 🧠 ML Pipeline

```
Raw Text → Preprocessing → TF-IDF Vectorization → LinearSVC → Prediction + Confidence
```

### Text Preprocessing
Every article goes through a custom cleaning pipeline before reaching the model:
- Lowercasing and tokenization
- Removal of punctuation and numeric tokens
- Stopword filtering using sklearn's English stopword list
- Snowball stemming to normalize word forms

### TF-IDF Feature Engineering
- 5000 features extracted from preprocessed text
- Captures term importance relative to the entire corpus
- Sparse matrix representation for efficient inference

### Model Selection
8 classifiers were trained and benchmarked before selecting the final model:

| Model | Selected |
|---|---|
| **LinearSVC** | ✅ Best F1 + speed |
| Logistic Regression | |
| Multinomial Naive Bayes | |
| Complement Naive Bayes | |
| Bernoulli Naive Bayes | |
| Perceptron | |
| Ridge Classifier | |
| SGD Classifier | |

**Metrics used:** Accuracy, Weighted F1-score, Precision, Confusion Matrix

### Confidence Estimation
LinearSVC does not natively output probabilities — softmax was applied on decision function scores to generate meaningful confidence percentages per class.

---

## 📊 EDA Highlights
- Class distribution analysis across 4 categories
- Top 15 most frequent words per category after preprocessing
- Visualizations generated with Matplotlib and Seaborn

---

## 🧪 Tech Stack

| Layer | Tools |
|---|---|
| **ML / NLP** | scikit-learn, NLTK, numpy |
| **Data** | pandas, AG News dataset |
| **Visualization** | matplotlib, seaborn |
| **Backend** | Flask, Gunicorn |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Deployment** | Docker, Hugging Face Spaces |

---

## 📌 Resume-Ready Points

- Built end-to-end **NLP classification system** using TF-IDF + LinearSVC
- Benchmarked **8 ML models** and selected optimal using weighted F1-score
- Implemented **confidence scores** via softmax on LinearSVC decision function
- Deployed production ML app using **Flask + Docker on Hugging Face Spaces**

---

## 👤 Author

**Muhammad Zain** — Final year Data Science student, Lahore

- GitHub: [Zainch032](https://github.com/Zainch032)
- LinkedIn: [Muhammad Zain](https://linkedin.com/in/muhammad-zain-9710692b4)
- Hugging Face: [Zainch032](https://huggingface.co/Zainch032)
- Portfolio: [zainch12.pythonanywhere.com](https://zainch12.pythonanywhere.com)

---

MIT License
```
