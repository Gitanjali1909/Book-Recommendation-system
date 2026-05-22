import pickle
from sklearn.metrics.pairwise import cosine_similarity


def load_similarity_matrix(path="outputs/similarity_matrix.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_similarity_matrix(df):
    """
    Build similarity matrix from precomputed features.
    """
    import pickle

    # load features
    with open("outputs/features.pkl", "rb") as f:
        features = pickle.load(f)

    # compute similarity
    similarity_matrix = cosine_similarity(features)

    return similarity_matrix


def recommend(book_title, df, similarity_matrix, top_n=5, rank_by_rating=False):
    matches = df[df["Book"].str.lower() == book_title.lower()]

    if matches.empty:
        return [{"error": "Book not found. Please check the title."}]

    idx = matches.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]

    recommended_books = [
        {
            "title": df.loc[i[0], "Book"],
            "author": df.loc[i[0], "Author"],
            "rating": df.loc[i[0], "Avg_Rating"],
            "description": df.loc[i[0], "Description"],
            "url": df.loc[i[0], "URL"] if "URL" in df.columns else "#",
        }
        for i in sim_scores
    ]

    if rank_by_rating:
        recommended_books = sorted(
            recommended_books,
            key=lambda x: x["rating"],
            reverse=True,
        )

    return recommended_books