📖 NextRead
Discover books you’ll love – personalized recommendations based on your favorites!

🌟 About the Project

This project is a Book Recommendation System that suggests books similar to the ones you already like. It uses content-based filtering on genres, authors, and descriptions to recommend titles.

I built it to get hands-on experience with real-world datasets, feature engineering, and recommendation algorithms. Plus, it has a simple Streamlit app so you can try it out interactively.

🛠️ How It Works

-Data & Cleaning
Loaded goodreads_books.csv from Kaggle.
Removed duplicates, handled missing values, parsed genres/authors.
Converted ratings to numbers and saved the clean dataset as books_cleaned.csv.

-Feature Engineering
Encoded genres & authors with one-hot encoding.
Transformed book descriptions with TF-IDF (5000 features).
Combined everything into a single sparse feature matrix.

-Model
Used cosine similarity to measure closeness between books.
Precomputed and saved the similarity matrix for fast lookups.
Wrote a recommend() function that returns similar books (title, author, rating, snippet, and Goodreads link).

-UI (Streamlit)
Dropdown autocomplete for book titles.
Sidebar filters for number of recommendations and rating ranking.
-Clean two-column layout with book cards:
Title (clickable to Goodreads)
Author, rating (⭐ stars)
Short description (expandable for more)

🚀 Demo
-Live App Link
 ()

📂 Project Structure
Book-Recommendation-System/
│
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for EDA
├── scripts/               # Data cleaning, feature engineering, model
├── outputs/               # Saved models, similarity matrix
├── main.py                # Streamlit app
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

📊 Tech Stack
Python (pandas, numpy, scikit-learn)
Streamlit (for interactive UI)
Pickle (for saving models/features)

✨ Features
Search any book title and get recommendations.
Choose number of recommendations (5, 10, etc.).
Option to rank results by rating.
Beautiful, clean UI with book cards and clickable links.

Dataset: Goodreads on Kaggle