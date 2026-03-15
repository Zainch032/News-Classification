from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.stem import SnowballStemmer
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib

# Use non-interactive backend for server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Initialize preprocessing resources
punctuation = set(string.punctuation)
stop = set(ENGLISH_STOP_WORDS)
stemmer = SnowballStemmer("english")

# Get absolute paths for PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "Linear_Svc.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "model", "tfidf.pkl")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

STATIC_DIR = os.path.join(BASE_DIR, "static")


# Load model + TF-IDF
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    tfidf = pickle.load(open(TFIDF_PATH, "rb"))
    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"✅ TF-IDF loaded from: {TFIDF_PATH}")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    model = None 
    tfidf = None

def preprocess(value):
    value = value.lower().split()
    value = [w for w in value if w.isalnum()]
    value = [w for w in value if w not in stop and w not in punctuation]
    value = [stemmer.stem(w) for w in value]
    return " ".join(value)

def map_prediction(pred):
    mapping = {1: "World", 2: "Sports", 3: "Business", 4: "Technology"}
    return mapping.get(pred[0], "Unknown")

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    sum_e_x = e_x.sum(axis=0)
    return e_x / sum_e_x if sum_e_x != 0 else np.zeros_like(e_x)

def get_probabilities(model, vector):
    """Get probabilities from model (works with SVC and LinearSVC)."""
    try:
        # Try predict_proba first (for SVC with probability=True)
        return model.predict_proba(vector)[0]
    except AttributeError:
        try:
            # For LinearSVC (no predict_proba)
            decision = model.decision_function(vector)
            
            # Handle different decision_function return shapes
            if len(decision.shape) == 2:
                scores = decision[0]  # Multi-class, one-vs-rest
            else:
                scores = decision  # Binary or weird shape
            
            # Apply softmax
            return softmax(scores)
        except Exception as e:
            print(f"Probability extraction error: {e}")
            return None

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    input_text = ""
    
    if request.method == "POST":
        input_text = request.form.get("input_text", "")
        
        if input_text.strip():
            if model is None:
                result = "Error: Model not loaded. Check server logs."
            elif tfidf is None:
                result = "Error: TF-IDF vectorizer not loaded."
            else:
                try:
                    # Preprocess and transform
                    processed = preprocess(input_text)
                    transformed = tfidf.transform([processed])
                    
                    # Predict
                    prediction = model.predict(transformed)
                    result = map_prediction(prediction)
                    
                    # Get confidence scores
                    probs = get_probabilities(model, transformed)
                    if probs is not None and len(probs) >= 4:
                        confidence = {
                            "World": float(probs[0]) * 100,
                            "Sports": float(probs[1]) * 100,
                            "Business": float(probs[2]) * 100,
                            "Technology": float(probs[3]) * 100,
                        }
                except Exception as e:
                    result = f"Prediction error: {str(e)}"
                    print(f"❌ Prediction error: {e}")
    
    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        input_text=input_text,
    )


if __name__ == "__main__":
    app.run(debug=True)