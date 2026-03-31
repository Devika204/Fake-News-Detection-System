import pickle
import numpy as np
import shap
import os
import re
import nltk
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# PATH (FOLDER - NO DATASET SELECTION)
# -----------------------------
MODEL_DIR = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\models"

# -----------------------------
# AUTO LOAD MODEL + VECTORIZER
# -----------------------------
model_file = None
vectorizer_file = None

for file in os.listdir(MODEL_DIR):

    if file.endswith("_model.pkl") and "nb" not in file and "sgd" not in file and "pa" not in file:
        model_file = file

    if file.endswith("_vectorizer.pkl"):
        vectorizer_file = file

if model_file is None or vectorizer_file is None:
    raise Exception(" Model or Vectorizer not found")

model = pickle.load(open(os.path.join(MODEL_DIR, model_file), "rb"))
vectorizer = pickle.load(open(os.path.join(MODEL_DIR, vectorizer_file), "rb"))

# -----------------------------
# NLTK SETUP
# -----------------------------
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")

try:
    nltk.corpus.wordnet.ensure_loaded()
except:
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# -----------------------------
# PREDICT PROBA (FOR LIME)
# -----------------------------
def predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    return model.predict_proba(vec)

# -----------------------------
# EXPLAIN FUNCTION
# -----------------------------
def explain_text(text, n=10):

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()

    print("\n===== EXPLANATION =====")

    # -----------------------------
    # 🔹 LIME EXPLANATION
    # -----------------------------
    print("\n--- LIME Explanation ---\n")

    try:
        explainer_lime = LimeTextExplainer(class_names=["Fake", "Real"])
        exp = explainer_lime.explain_instance(text, predict_proba, num_features=n)

        for word, weight in exp.as_list():
            print(f"{word}: {weight:.4f}")

    except Exception as e:
        print("LIME error:", e)

    # -----------------------------
    # 🔹 SHAP EXPLANATION
    # -----------------------------
    print("\n--- SHAP Explanation ---\n")

    try:
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]

            words = cleaned.split()
            word_scores = {}

            for word in words:
                if word in feature_names:
                    idx = np.where(feature_names == word)[0][0]
                    word_scores[word] = coefs[idx]

            top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

        else:
            explainer = shap.LinearExplainer(model, vec)
            shap_values = explainer.shap_values(vec)

            values = shap_values[0]
            shap_dict = dict(zip(feature_names, values))

            words = cleaned.split()
            filtered = {k: v for k, v in shap_dict.items() if k in words}

            top_words = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)[:n]

        if not top_words:
            print("No meaningful SHAP features found.")
            return

        for word, val in top_words:
            print(f"{word}: {val:.4f}")

        # -----------------------------
        #  VISUALIZATION (BAR CHART)
        # -----------------------------
        words_plot = [w for w, _ in top_words]
        scores_plot = [v for _, v in top_words]

        plt.figure(figsize=(8, 5))
        colors = ['green' if s > 0 else 'red' for s in scores_plot]

        plt.barh(words_plot, scores_plot, color=colors)
        plt.xlabel("Impact on Prediction")
        plt.title("SHAP Feature Importance")
        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("SHAP error:", e)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    user_text = input("\nEnter news text:\n")
    explain_text(user_text)