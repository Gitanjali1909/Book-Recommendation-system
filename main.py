import streamlit as st
import pandas as pd
import pickle
from scripts.model import recommend  # Import your recommend function

# Load cleaned data
df = pd.read_csv("data/books_cleaned.csv")

# Load similarity matrix (already computed in model.py)
with open("outputs/similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

st.set_page_config(page_title="Book Recommendation System", layout="wide")

st.title("📚 Book Recommendation System")
st.markdown("Find books similar to your favorites using content-based filtering!")

# Input box for book title
book_title = st.text_input("Enter a book title:", "")

# Number of recommendations
top_n = st.slider("Number of recommendations:", 1, 10, 5)

# Rank by rating option
rank_by_rating = st.checkbox("Rank recommendations by rating", value=False)

if st.button("Get Recommendations"):
    if book_title.strip() == "":
        st.warning("Please enter a book title.")
    else:
        results = recommend(book_title, top_n=top_n, rank_by_rating=rank_by_rating)

        if "error" in results[0]:
            st.error(results[0]["error"])
        else:
            st.subheader("Recommended Books:")
            for r in results:
                st.write(f"**{r['title']}** by *{r['author']}* (⭐ {r['rating']})")
