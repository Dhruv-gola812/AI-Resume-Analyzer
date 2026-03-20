import os
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset/resume_dataset.csv"
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "resume_category_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "resume_vectorizer.pkl")


# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# DETECT COLUMNS
# -----------------------------
def detect_columns(df):
    possible_text_cols = ["Text", "text", "resume_text", "Resume", "Resume_str", "resume", "resumeText"]
    possible_label_cols = ["Category", "category", "Label", "label", "Job Category", "job_category"]

    text_col = None
    label_col = None

    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            break

    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    return text_col, label_col


# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

text_col, label_col = detect_columns(df)

if text_col is None or label_col is None:
    raise ValueError(
        f"Could not detect required columns.\n"
        f"Found columns: {list(df.columns)}\n"
        f"Need one text column and one category column."
    )

print(f"Using text column   : {text_col}")
print(f"Using label column  : {label_col}")

df = df[[text_col, label_col]].dropna()
df.columns = ["resume_text", "category"]

# Remove duplicates
df = df.drop_duplicates()

# Clean text
df["cleaned_text"] = df["resume_text"].apply(clean_text)

# Remove very short rows
df = df[df["cleaned_text"].str.len() > 30]

print("Cleaned dataset shape:", df.shape)
print("\nCategory counts:\n")
print(df["category"].value_counts())


# -----------------------------
# FEATURES + LABELS
# -----------------------------
X = df["cleaned_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=8000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# -----------------------------
# MODEL TRAINING
# -----------------------------
# Wrap in One-vs-Rest for multiclass support with liblinear.
base_model = LogisticRegression(
    max_iter=2000,
    solver="liblinear"
)
model = OneVsRestClassifier(base_model)

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(VECTORIZER_PATH, "wb") as f:
    pickle.dump(vectorizer, f)

print("\nSaved files:")
print(MODEL_PATH)
print(VECTORIZER_PATH)