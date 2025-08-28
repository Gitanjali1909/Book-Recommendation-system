import streamlit as st
import pandas as pd
import pickle
from scripts.model import recommend

df = pd.read_csv("data/books_cleaned.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
df.columns = df.columns.str.lower()

with open("outputs/similarity_matrix.pkl", "rb") as f:
    similarity_matrix = pickle.load(f)

st.set_page_config(page_title="📚 Book Recommendation System", layout="wide")
st.title("📚 NextRead")
st.markdown("Discover books you’ll love – personalized recommendations based on your favorites!")

st.sidebar.header("Filters")
rank_by_rating = st.sidebar.checkbox("Rank recommendations by rating", value=False)
top_n = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

book_title = st.selectbox("Select a book:", df["book"].sort_values().unique())

def stars(rating):
    full = "⭐" * int(rating)
    return full

if st.button("Get Recommendations"):
    results = recommend(book_title, top_n=top_n, rank_by_rating=rank_by_rating)
    if results and "error" in results[0]:
        st.error(results[0]["error"])
    else:
        st.subheader("Recommended Books:")
        cols = st.columns(2)
        for i, r in enumerate(results):
            col = cols[i % 2]
            with col.container():
                st.markdown(
                    f"<div style='border:1px solid #ecf0f1; padding:12px; border-radius:10px; margin-bottom:12px; background-color:#fdfdfd;'>"
                    f"<h4 style='margin:0; color:#2c3e50; font-size:18px; line-height:1.2;'>{r['title']}</h4>"
                    f"<p style='margin:0; font-style:italic; color:#7f8c8d; font-size:14px; line-height:1.2;'>{r['author']}</p>"
                    f"<p style='margin:4px 0 4px 0; font-size:14px; line-height:1.2;'>Rating: {stars(r['rating'])} ({r['rating']})</p>"
                    f"</div>", unsafe_allow_html=True
                )
                desc = r.get("description", "")
                snippet = desc[:200] + "..." if len(desc) > 200 else desc
                st.write(snippet if len(desc) > 200 else desc)
                if len(desc) > 200:
                    with st.expander("Read Full Description"):
                        st.write(desc)
                url = r.get("url", "#")
                if url != "#":
                    st.markdown(f"<a href='{url}' target='_blank' style='text-decoration:none; color:#2980b9; font-size:14px;'>Go to Goodreads</a>", unsafe_allow_html=True)
