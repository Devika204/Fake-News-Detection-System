import streamlit as st
import pickle
import os
import re
import sys
import nltk
import shap
import hashlib
import numpy as np
import matplotlib.pyplot as plt

from lime.lime_text import LimeTextExplainer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from blockchain import store_hash

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

# -----------------------------
# STYLE
# -----------------------------
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    border-radius: 8px;
    height: 45px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# NLTK SETUP
# -----------------------------
try:
    stopwords.words("english")
except:
    nltk.download('stopwords')

try:
    nltk.corpus.wordnet.ensure_loaded()
except:
    nltk.download('wordnet')

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
# LOAD MODEL (AUTO FROM FOLDER)
# -----------------------------
MODEL_DIR = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\models"

@st.cache_resource
def load_model():
    model_file = None
    vectorizer_file = None

    for file in os.listdir(MODEL_DIR):
        if file.endswith("_model.pkl") and "nb" not in file and "sgd" not in file and "pa" not in file:
            model_file = file
        if file.endswith("_vectorizer.pkl"):
            vectorizer_file = file

    if model_file is None or vectorizer_file is None:
        raise Exception("Model or Vectorizer not found")

    model = pickle.load(open(os.path.join(MODEL_DIR, model_file), "rb"))
    vectorizer = pickle.load(open(os.path.join(MODEL_DIR, vectorizer_file), "rb"))

    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# LIME
# -----------------------------
explainer_lime = LimeTextExplainer(class_names=["Fake", "Real"])

# -----------------------------
# PREDICT
# -----------------------------
def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

    try:
        prob = model.predict_proba(vec)[0][pred]
    except:
        prob = 0

    return pred, prob, cleaned

def predict_proba(texts):
    cleaned = [clean_text(t) for t in texts]
    vec = vectorizer.transform(cleaned)
    return model.predict_proba(vec)

# -----------------------------
# UI
# -----------------------------
st.title("📰 Fake News Detection + Blockchain + XAI")
st.markdown("---")

user_input = st.text_area("Enter News Text:", height=150)
analyze_clicked = st.button("Analyze")

# -----------------------------
# MAIN
# -----------------------------
if analyze_clicked:

    if not user_input.strip():
        st.warning("⚠ Please enter text")

    else:
        pred, prob, cleaned_text = predict(user_input)
        label = "REAL ✅" if pred == 1 else "FAKE ❌"

        # -----------------------------
        # RESULT
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction")
            st.metric(label="Result", value=label)
            st.metric(label="Confidence", value=f"{prob:.2f}")

        with col2:
            st.subheader("Hash Preview")
            news_hash = hashlib.sha256(cleaned_text.encode()).hexdigest()
            st.code(news_hash[:40] + "...")

        st.markdown("---")

        # -----------------------------
        # XAI SECTION
        # -----------------------------
        col3, col4 = st.columns(2)

        # -----------------------------
        # LIME
        # -----------------------------
        with col3:
            st.subheader("LIME Explanation")

            try:
                exp = explainer_lime.explain_instance(user_input, predict_proba, num_features=10)
                st.components.v1.html(exp.as_html(), height=400)
            except Exception as e:
                st.error(f"LIME error: {e}")

        # -----------------------------
        # SHAP (FIXED)
        # -----------------------------
        with col4:
            st.subheader("SHAP Explanation")

            try:
                feature_names = vectorizer.get_feature_names_out()

                if hasattr(model, "coef_"):
                    coefs = model.coef_[0]
                    words = cleaned_text.split()

                    word_scores = {}
                    for word in words:
                        if word in feature_names:
                            idx = np.where(feature_names == word)[0][0]
                            word_scores[word] = coefs[idx]

                    top_features = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                else:
                    vec = vectorizer.transform([cleaned_text])
                    explainer = shap.LinearExplainer(model, vec)
                    shap_values = explainer.shap_values(vec)

                    values = shap_values[0]
                    shap_dict = dict(zip(feature_names, values))

                    words = cleaned_text.split()
                    filtered = {k: v for k, v in shap_dict.items() if k in words}

                    top_features = sorted(filtered.items(), key=lambda x: abs(x[1]), reverse=True)[:10]

                if not top_features:
                    st.warning("No meaningful SHAP features found")
                else:
                    # TEXT OUTPUT
                    for word, val in top_features:
                        st.write(f"{word}: {val:.4f}")

                    # 📊 BAR CHART
                    words_plot = [w for w, _ in top_features]
                    scores_plot = [v for _, v in top_features]

                    fig, ax = plt.subplots()
                    colors = ['green' if s > 0 else 'red' for s in scores_plot]

                    ax.barh(words_plot, scores_plot, color=colors)
                    ax.set_xlabel("Impact")
                    ax.set_title("SHAP Feature Importance")
                    ax.invert_yaxis()

                    st.pyplot(fig)

            except Exception as e:
                st.error(f"SHAP error: {e}")

        st.markdown("---")

        # -----------------------------
        # BLOCKCHAIN
        # -----------------------------
        st.subheader("⛓ Blockchain Verification")

        col5, col6 = st.columns(2)

        with col5:
            if st.button("Store on Blockchain"):
                try:
                    tx = store_hash.store_hash_on_chain(news_hash, label)
                    st.success(f"Stored!\nTX: {tx}")
                except Exception as e:
                    st.error(f"Store error: {e}")

        with col6:
            if st.button("Verify Hash"):
                try:
                    exists = store_hash.verify_hash(news_hash)

                    if exists:
                        prediction_on_chain = store_hash.get_prediction(news_hash)
                        st.success(f"✔ Verified\nStored: {prediction_on_chain}")
                    else:
                        st.error("Not found on blockchain")

                except Exception as e:
                    st.error(f"Verify error: {e}")