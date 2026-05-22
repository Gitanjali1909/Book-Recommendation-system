import pandas as pd
import streamlit as st

from scripts.model import load_similarity_matrix, recommend


# -------------------- DATA LOADING --------------------

@st.cache_data
def load_dataset(path="data/books_cleaned.csv"):
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


# -------------------- SIMILARITY --------------------

@st.cache_resource
def load_similarity(df):
    return load_similarity_matrix(df)


# -------------------- LOAD --------------------

df = load_dataset()
similarity_matrix = load_similarity(df)


# -------------------- UI --------------------

st.set_page_config(page_title="NextRead", layout="wide")

st.title("NextRead 📚")
st.markdown("Discover books you'll love — personalized recommendations based on your favorites!")

# Sidebar
st.sidebar.header("Filters")
rank_by_rating = st.sidebar.checkbox("Rank recommendations by rating", value=False)
top_n = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

# Book selection
book_title = st.selectbox("Select a book:", df["Book"].sort_values().unique())


# -------------------- HELPERS --------------------

def stars(rating):
    try:
        return "⭐" * int(float(rating))
    except:
        return "⭐"


# -------------------- ACTION --------------------

if st.button("Get Recommendations"):
    with st.spinner("Finding your next favorite book..."):
        results = recommend(
            book_title,
            df,
            similarity_matrix,
            top_n=top_n,
            rank_by_rating=rank_by_rating,
        )

    if results and "error" in results[0]:
        st.error(results[0]["error"])
    else:
        st.subheader("Recommended Books:")

        cols = st.columns(2)

        for i, r in enumerate(results):
            col = cols[i % 2]

            with col:
                st.markdown(
                    f"""
                    <div style='border:1px solid #ecf0f1; padding:12px; border-radius:10px; margin-bottom:12px; background-color:#fdfdfd;'>
                        <h4 style='margin:0; color:#2c3e50;'>{r['title']}</h4>
                        <p style='margin:0; font-style:italic; color:#7f8c8d;'>{r['author']}</p>
                        <p style='margin:4px 0;'>Rating: {stars(r['rating'])} ({r['rating']})</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                desc = r.get("description", "")
                snippet = desc[:200] + "..." if len(desc) > 200 else desc

                st.write(snippet)

                if len(desc) > 200:
                    with st.expander("Read Full Description"):
                        st.write(desc)

                url = r.get("url", "#")
                if url != "#":
                    st.markdown(
                        f"<a href='{url}' target='_blank' style='text-decoration:none; color:#2980b9;'>Go to Goodreads</a>",
                        unsafe_allow_html=True,
                    )