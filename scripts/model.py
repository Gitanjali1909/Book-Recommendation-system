import pandas as pd
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def build_similarity_matrix(df):
    """
    Build similarity matrix directly from dataframe (NO external files).
    """

    # Handle missing values
    df["Description"] = df["Description"].fillna("")
    df["Genres"] = df["Genres"].fillna("")
    df["Author"] = df["Author"].fillna("")

    # Combine features
    df["combined_features"] = (
        df["Genres"] + " " + df["Author"] + " " + df["Description"]
    )

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    feature_matrix = tfidf.fit_transform(df["combined_features"])

    # Cosine similarity
    similarity_matrix = cosine_similarity(feature_matrix)

    return similarity_matrix


def load_similarity_matrix(df, path="outputs/similarity_matrix.pkl"):
    """
    Load similarity matrix if exists, else build and save it.
    """

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Build if not found
    similarity_matrix = build_similarity_matrix(df)

    # Save for future use
    os.makedirs("outputs", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(similarity_matrix, f)

    return similarity_matrix


def recommend(book_title, df, similarity_matrix, top_n=5, rank_by_rating=False):
    """
    Recommend similar books
    """

    matches = df[df["Book"].str.lower() == book_title.lower()]

    if matches.empty:
        return [{"error": "Book not found. Please check the title."}]

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the selected book itself
    sim_scores = sim_scores[1: top_n + 1]

    recommended_books = [
        {
            "title": df.loc[i[0], "Book"],
            "author": df.loc[i[0], "Author"],
            "rating": float(df.loc[i[0], "Avg_Rating"]),
            "description": df.loc[i[0], "Description"],
            "url": df.loc[i[0], "URL"] if "URL" in df.columns else "#",
        }
        for i in sim_scores
    ]

    # Optional ranking
    if rank_by_rating:
        recommended_books = sorted(
            recommended_books,
            key=lambda x: x["rating"],
            reverse=True,
        )

    return recommended_books