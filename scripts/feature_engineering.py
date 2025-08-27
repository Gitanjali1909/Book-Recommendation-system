# scripts/feature_engineering.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack, csr_matrix
import pickle
import os

# --------------------------
# 1️⃣ Load Cleaned Data
# --------------------------
df = pd.read_csv("data/books_cleaned.csv")

# --------------------------
# 2️⃣ Process Genres
# --------------------------
df['Genres'] = df['Genres'].fillna("").apply(
    lambda x: [g.strip() for g in str(x).split(",") if g.strip()]
)

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['Genres'])
genres_sparse = csr_matrix(genres_encoded)

# --------------------------
# 3️⃣ Process Authors
# --------------------------
df['Author'] = df['Author'].fillna("Unknown")
authors_encoded = pd.get_dummies(df['Author'])
authors_sparse = csr_matrix(authors_encoded.values)

# --------------------------
# 4️⃣ Process Descriptions
# --------------------------
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
desc_sparse = tfidf.fit_transform(df['Description'].fillna(""))

# --------------------------
# 5️⃣ Combine All Features
# --------------------------
features = hstack([genres_sparse, authors_sparse, desc_sparse]).tocsr()

# --------------------------
# 6️⃣ Save Features & Encoders
# --------------------------
os.makedirs("outputs", exist_ok=True)

with open("outputs/features.pkl", "wb") as f:
    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("outputs/genre_encoder.pkl", "wb") as f:
    pickle.dump(mlb, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("outputs/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Feature engineering complete!")
print(f"Features shape: {features.shape}")
