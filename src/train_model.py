import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ==========================================
# Get Project Base Directory
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# File Paths
# ==========================================

csv_path = os.path.join(BASE_DIR, "data", "raw", "loan_data.csv")

model_path = os.path.join(BASE_DIR, "models", "xgboost_loan_model.pkl")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(csv_path)

print("✅ Dataset Loaded Successfully")
print(df.head())

# ==========================================
# Features & Target
# ==========================================

X = df.drop("approved", axis=1)
y = df["approved"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# Build XGBoost Model
# ==========================================

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

# ==========================================
# Train Model
# ==========================================

model.fit(X_train, y_train)

print("✅ Model Training Completed")

# ==========================================
# Predictions
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Model Accuracy: {accuracy*100:.2f}%")

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, model_path)

print("✅ Model saved successfully")
print(f"📁 Saved At: {model_path}")