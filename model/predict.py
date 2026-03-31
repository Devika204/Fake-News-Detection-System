import pickle
import re
import nltk
import shap
import numpy as np
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer

# download if not present
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# SELECT DATASET (FIX)
# -----------------------------
MODEL_DIR = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\models"

datasets = []

for file in os.listdir(MODEL_DIR):
    if file.endswith("_model.pkl"):
        datasets.append(file.replace("_model.pkl", ""))

print("\nAvailable datasets:")
for i, d in enumerate(datasets):
    print(f"{i+1}. {d}")

choice = int(input("\nSelect dataset number: ")) - 1
dataset_name = datasets[choice]

# -----------------------------
# Load model + vectorizer (FIXED)
# -----------------------------
model = pickle.load(open(os.path.join(MODEL_DIR, f"{dataset_name}_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(MODEL_DIR, f"{dataset_name}_vectorizer.pkl"), "rb"))

# OPTIONAL: load more models if available
models = {
    "LogisticRegression": model
}

try:
    models["NaiveBayes"] = pickle.load(
        open(os.path.join(MODEL_DIR, f"{dataset_name}_nb_model.pkl"), "rb")
    )
    models["SGD"] = pickle.load(
        open(os.path.join(MODEL_DIR, f"{dataset_name}_sgd_model.pkl"), "rb")
    )
    models["PassiveAggressive"] = pickle.load(
        open(os.path.join(MODEL_DIR, f"{dataset_name}_pa_model.pkl"), "rb")
    )
except:
    print("Other models not found, using main model only.")

# -----------------------------
# Preprocessing tools
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# -----------------------------
# LIME setup
# -----------------------------
class_names = ["Fake", "Real"]
explainer = LimeTextExplainer(class_names=class_names)

def predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    return model.predict_proba(vec)

# -----------------------------
# User input
# -----------------------------
news = input("Enter news article:\n")

cleaned_news = clean_text(news)
news_vector = vectorizer.transform([cleaned_news])

# -----------------------------
# MULTI-MODEL PREDICTION
# -----------------------------
print("\n===== MODEL COMPARISON =====")

for name, m in models.items():
    pred = m.predict(news_vector)[0]

    try:
        prob = m.predict_proba(news_vector)[0]
        confidence = max(prob)
    except:
        confidence = "N/A"

    label = "Real" if pred == 1 else "Fake"

    print(f"{name}: {label} (Confidence: {confidence})")

# -----------------------------
# MAIN MODEL RESULT
# -----------------------------
prediction = model.predict(news_vector)[0]
prob = model.predict_proba(news_vector)[0]

if prediction == 0:
    print("\nFinal Prediction: Fake News")
    print(f"Confidence: {prob[0]:.2f}")
else:
    print("\nFinal Prediction: Real News")
    print(f"Confidence: {prob[1]:.2f}")

# -----------------------------
# LIME EXPLAINABILITY
# -----------------------------
print("\nGenerating LIME explanation...")

exp = explainer.explain_instance(
    news,
    predict_proba,
    num_features=10
)

print("\nTop contributing words (LIME):")
for word, weight in exp.as_list():
    print(f"{word}: {weight:.3f}")

exp.save_to_file("explanation.html")

print("\nLIME explanation saved as 'explanation.html'")

# -----------------------------
# SHAP EXPLAINABILITY (FIXED)
# -----------------------------
print("\nGenerating SHAP explanation...")

try:
    # Proper background (small generic samples)
    background = vectorizer.transform([
        "government announces new policy",
        "scientists discover new species",
        "breaking news in world politics",
        "health experts warn about virus",
        "economic growth slows down"
    ])

    # Create SHAP explainer
    explainer_shap = shap.Explainer(model, background)

    # Explain the actual input (IMPORTANT FIX)
    shap_values = explainer_shap(news_vector)

    print("\nTop contributing features (SHAP):")

    feature_names = vectorizer.get_feature_names_out()
    values = shap_values.values[0]

    top_indices = np.argsort(np.abs(values))[-10:]

    for i in reversed(top_indices):
        print(f"{feature_names[i]}: {values[i]:.4f}")

except Exception as e:
    print("SHAP failed:", e)