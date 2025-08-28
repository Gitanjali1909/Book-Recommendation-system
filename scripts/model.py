import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/books_cleaned.csv")

with open("outputs/features.pkl", "rb") as f:
    features = pickle.load(f)

if os.path.exists("outputs/similarity_matrix.pkl"):
    with open("outputs/similarity_matrix.pkl", "rb") as f:
        similarity_matrix = pickle.load(f)
    print("✅ Loaded existing similarity_matrix.pkl")
else:
    print("⚡ Computing similarity matrix...")
    similarity_matrix = cosine_similarity(features)
    with open("outputs/similarity_matrix.pkl", "wb") as f:
        pickle.dump(similarity_matrix, f)
    print("💾 Saved new similarity_matrix.pkl")


def recommend(book_title, top_n=5, rank_by_rating=False):
    """
    Recommend similar books using cosine similarity.

    Parameters:
        book_title (str): Title of the book.
        top_n (int): Number of recommendations.
        rank_by_rating (bool): Whether to rank results by average rating.

    Returns:
        List of dicts with title, author, rating, and description.
    """

    matches = df[df['Book'].str.lower() == book_title.lower()]

    if matches.empty:
        return [{"error": "Book not found. Please check the title."}]
    
    idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]
    
    recommended_books = [
        {
            "title": df.loc[i[0], "Book"],
            "author": df.loc[i[0], "Author"],
            "rating": df.loc[i[0], "Avg_Rating"],
            "description": df.loc[i[0], "Description"],  
            "url": df.loc[i[0], "URL"] if "URL" in df.columns else "#"
        }
        for i in sim_scores
    ]

    if rank_by_rating:
        recommended_books = sorted(recommended_books, key=lambda x: x["rating"], reverse=True)

    return recommended_books
