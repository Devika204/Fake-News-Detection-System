import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# PATHS
# -----------------------------
data_folder = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\data"
model_folder = r"C:\Users\devik\Desktop\Blockchain_code_Original_2\models"

# -----------------------------
# LOOP THROUGH CLEANED DATASETS
# -----------------------------
for file in os.listdir(data_folder):

    if not file.startswith("cleaned_") or not file.endswith(".csv"):
        continue

    print(f"\n Evaluating: {file}")

    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)

    # -----------------------------
    # BASIC CLEAN CHECK
    # -----------------------------
    df = df.drop_duplicates()
    df = df[df["clean_text"].str.len() > 20]

    print("Dataset shape:", df.shape)
    print("Label distribution:\n", df["label"].value_counts())

    X = df["clean_text"]
    y = df["label"]

    # -----------------------------
    # SPLIT (NO SMOTE HERE )
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    dataset_name = file.replace("cleaned_", "").replace(".csv", "")

    # -----------------------------
    # LOAD MODEL + VECTORIZER
    # -----------------------------
    model_path = os.path.join(model_folder, f"{dataset_name}_model.pkl")
    vectorizer_path = os.path.join(model_folder, f"{dataset_name}_vectorizer.pkl")

    if not os.path.exists(model_path):
        print(f" Model not found for {dataset_name}")
        continue

    model = pickle.load(open(model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

    # -----------------------------
    # TRANSFORM (IMPORTANT)
    # -----------------------------
    X_test_vec = vectorizer.transform(X_test)

    # -----------------------------
    # PREDICT
    # -----------------------------
    y_pred = model.predict(X_test_vec)

    # -----------------------------
    # METRICS
    # -----------------------------
    print("\n===== RESULTS =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1 Score :", f1_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n DONE: Evaluation completed for all datasets!")