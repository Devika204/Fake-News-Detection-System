import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

from imblearn.over_sampling import SMOTE

# DL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# PATHS
# -----------------------------
data_folder = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\dataset"
model_folder = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\models"

os.makedirs(model_folder, exist_ok=True)

# -----------------------------
# EVALUATION FUNCTION
# -----------------------------
def evaluate_model(name, model, X_test_vec, y_test, prefix):

    y_pred = model.predict(X_test_vec)

    print(f"\n===== {prefix} - {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{prefix} - {name} Confusion Matrix")
    plt.savefig(f"{prefix}_{name}_confusion.png")
    plt.close()

    try:
        y_prob = model.predict_proba(X_test_vec)[:, 1]
    except:
        y_prob = model.decision_function(X_test_vec)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.legend()
    plt.title(f"{prefix} - {name} ROC")
    plt.savefig(f"{prefix}_{name}_roc.png")
    plt.close()

# -----------------------------
# COMMON TRAIN FUNCTION
# -----------------------------
def train_all_models(X_train, X_test, y_train, y_test, dataset_name):

    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=3, max_df=0.9)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if len(set(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_vec, y_train)
    else:
        X_train_res, y_train_res = X_train_vec, y_train

    # =============================
    # ML MODELS
    # =============================
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "NaiveBayes": MultinomialNB(),
        "SGD": SGDClassifier(loss='log_loss', max_iter=500),
        "PassiveAggressive": PassiveAggressiveClassifier(max_iter=500)
    }

    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        evaluate_model(name, model, X_test_vec, y_test, dataset_name)

    # =============================
    # LSTM + CNN
    # =============================
    print("\nTraining LSTM & CNN...")

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=200)
    X_test_pad = pad_sequences(X_test_seq, maxlen=200)

    # LSTM
    lstm_model = Sequential([
        Embedding(10000, 128, input_length=200),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train_pad, y_train, epochs=2, batch_size=64)

    # CNN
    cnn_model = Sequential([
        Embedding(10000, 128, input_length=200),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])

    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    cnn_model.fit(X_train_pad, y_train, epochs=2, batch_size=64)

    # SAVE
    with open(os.path.join(model_folder, f"{dataset_name}_model.pkl"), "wb") as f:
        pickle.dump(models["LogisticRegression"], f)

    with open(os.path.join(model_folder, f"{dataset_name}_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f" Saved models for {dataset_name}")


# -----------------------------
# START PROCESSING
# -----------------------------
files = os.listdir(data_folder)

# -----------------------------
# Fake + True
# -----------------------------
if "Fake.csv" in files and "True.csv" in files:

    print("\n Processing: Fake + True")

    fake_df = pd.read_csv(os.path.join(data_folder, "Fake.csv"))
    true_df = pd.read_csv(os.path.join(data_folder, "True.csv"))

    fake_df["clean_text"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
    true_df["clean_text"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df = df[["clean_text", "label"]]
    df = df.drop_duplicates()
    df = df[df["clean_text"].str.len() > 20]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_all_models(X_train, X_test, y_train, y_test, "Fake_True")

# -----------------------------
# Onion dataset
# -----------------------------
if "the_onion.csv" in files and "not_onion.csv" in files:

    print("\n Processing: Onion")

    fake_df = pd.read_csv(os.path.join(data_folder, "the_onion.csv"))
    real_df = pd.read_csv(os.path.join(data_folder, "not_onion.csv"))

    fake_df["clean_text"] = fake_df["title"].fillna("")
    real_df["clean_text"] = real_df["title"].fillna("")

    fake_df["label"] = 0
    real_df["label"] = 1

    df = pd.concat([fake_df, real_df], ignore_index=True)

    df = df[["clean_text", "label"]]
    df = df.drop_duplicates()
    df = df[df["clean_text"].str.len() > 20]

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    train_all_models(X_train, X_test, y_train, y_test, "Onion")

# -----------------------------
# OTHER DATASETS
# -----------------------------
for file in os.listdir(data_folder):

    if file in ["Fake.csv", "True.csv", "the_onion.csv", "not_onion.csv"]:
        continue

    if not file.endswith(".csv"):
        continue

    print(f"\n Processing: {file}")

    df = pd.read_csv(os.path.join(data_folder, file))

    if "text" not in df.columns or "label" not in df.columns:
        continue

    df["clean_text"] = df["text"].fillna("")
    df = df[["clean_text", "label"]]
    df = df.drop_duplicates()
    df = df[df["clean_text"].str.len() > 20]

    if len(df["label"].unique()) < 2:
        print(" Skipping (only one class)")
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    dataset_name = file.replace(".csv", "")
    train_all_models(X_train, X_test, y_train, y_test, dataset_name)

print("\n DONE: All models trained for all datasets!")