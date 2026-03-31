import subprocess

print("========== FAKE NEWS BLOCKCHAIN SYSTEM ==========\n")

# -----------------------------
# Step 1: Preprocessing
# -----------------------------
print("Step 1: Running Data Preprocessing...")
subprocess.run(["python", "preprocessing/preprocess.py"])
print("✅ Preprocessing Completed\n")

# -----------------------------
# Step 2: Train Models (ALL DATASETS)
# -----------------------------
print("Step 2: Training ML Models (per dataset)...")
subprocess.run(["python", "model/train_model.py"])
print("✅ Model Training Completed\n")

# -----------------------------
# Step 3: User Input
# -----------------------------
print("Step 3: Enter News Article\n")
news_text = input("Enter News Article:\n")

# -----------------------------
# Step 4: Prediction + Explainability + Blockchain
# -----------------------------
print("\nStep 4: Prediction + Explainability + Blockchain...\n")

subprocess.run([
    "python",
    "blockchain/store_hash.py",
    news_text
])

print("\n========== PROCESS COMPLETED SUCCESSFULLY ==========")