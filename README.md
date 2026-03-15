---
title: News Classification
emoji: 📰
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---
## News Classification App

This project is a Flask-based web application for classifying news articles into categories (World, Sports, Business, Technology). It uses a pre-trained scikit-learn model and TF-IDF vectorizer, and displays simple dataset visualizations using Matplotlib and Seaborn.

### Running locally (without Docker)

1. **Create and activate a virtual environment (optional but recommended).**
2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run the Flask app:**

```bash
python app.py
```

4. Open your browser at `http://127.0.0.1:5000/`.

### Running with Docker

The repository includes a `Dockerfile` that runs the app with Gunicorn on port **7860**.

1. **Build the image:**

```bash
docker build -t news-classification .
```

2. **Run the container (mapping container port 7860 to your host):**

```bash
docker run -p 7860:7860 news-classification
```

3. Open your browser at `http://127.0.0.1:7860/`.

# 📰 News Multi-Classification System

**End-to-End NLP & Machine Learning Project**

🌐 **Live Demo:** [https://zainch12.pythonanywhere.com](https://zainch12.pythonanywhere.com)

---

## 🎯 What This Project Does

Classifies raw news articles into **World, Business, Sports, and Technology** using a trained ML model.

---

## 🧠 ML / Data Science Highlights (Key Focus)

* End-to-end **NLP pipeline** (raw text → prediction)
* Custom **text preprocessing** (stopwords, punctuation, stemming)
* **TF-IDF feature engineering** (5000 features)
* Trained & benchmarked **8 ML classifiers**
* Final model: **LinearSVC** (best F1-score & scalability)
* **Model confidence estimation** using softmax on decision scores
* Full **EDA + visualization** (class distribution, top words)
* Production-ready **Flask deployment**

---

## ⚙️ ML Pipeline (At a Glance)

**Data → Preprocessing → TF-IDF → Model Training → Evaluation → Deployment**

---

## 📊 Models Evaluated

* LinearSVC ⭐ (Selected)
* Logistic Regression
* Multinomial / Complement / Bernoulli Naive Bayes
* Perceptron
* Ridge Classifier
* SGD Classifier

**Metrics:** Accuracy, Weighted F1-score, Precision, Confusion Matrix

---

## 🧪 Tech Stack

* **ML/NLP:** Python, scikit-learn, NLTK
* **Data:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Frontend:** Html,CSS,Java script
* **Deployment:** Flask, Pickle 


---

## 🚀 Run Locally

```bash
cd app
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

---

## 📌 Resume-Ready Points

* Built an **end-to-end NLP classification system** using TF-IDF + LinearSVC
* Benchmarked **multiple ML models** and selected optimal model using F1-score
* Implemented **real-time inference with confidence scores**
* Deployed ML model using **Flask with data visualizations**

---

## 👤 Author

**Muhammad Zain**

* 📧 [zc19398@gmail.com](mailto:zc19398@gmail.com)
* 🐙 [https://github.com/Zainch032](https://github.com/Zainch032)
* 🌐 [https://zainch12.pythonanywhere.com](https://zainch12.pythonanywhere.com)

---


