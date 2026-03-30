# ==============================
# SUPPORT TICKET CLASSIFICATION
# ==============================

import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. DOWNLOAD NLTK DATA
# ==============================
nltk.download('stopwords')
nltk.download('wordnet')

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("tickets.csv")

print("\nDataset Preview:")
print(df.head())

# ==============================
# 3. DEFINE COLUMNS (IMPORTANT)
# ==============================
TEXT_COL = "Ticket Description"
TARGET_COL = "Ticket Subject"
PRIORITY_COL = "Ticket Priority"

# ==============================
# 4. CLEAN TEXT
# ==============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df[TEXT_COL].apply(clean_text)

# ==============================
# 5. FEATURE EXTRACTION
# ==============================
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

y = df[TARGET_COL]

# ==============================
# 6. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. MODEL TRAINING
# ==============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================
# 8. PREDICTION
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 9. EVALUATION
# ==============================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 10. PRIORITY ANALYSIS (USE EXISTING COLUMN)
# ==============================
print("\nPriority Distribution:")
print(df[PRIORITY_COL].value_counts())

# ==============================
# 11. TEST NEW TICKET
# ==============================
def predict_ticket(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    category = model.predict(vector)[0]
    return category

new_ticket = "My internet is not working, urgent issue"

print("\nNew Ticket Prediction:")
print("Category:", predict_ticket(new_ticket))