# NextRead: Book Recommendation System

Discover books you’ll love – personalized recommendations based on your favorites.

## About the Project
This is a Book Recommendation System that suggests books similar to the ones you already like.  
It uses content-based filtering on genres, authors, and descriptions.  
The goal was to practice working with real datasets, feature engineering, and recommendation algorithms.  

## How It Works
### Data & Cleaning
- Loaded `goodreads_books.csv` from Kaggle  
- Removed duplicates and handled missing values  
- Parsed genres/authors and converted ratings  
- Saved clean dataset as `books_cleaned.csv`

### Feature Engineering
- Encoded genres and authors with one-hot encoding  
- Transformed book descriptions with TF-IDF (5000 features)  
- Combined all features into a single sparse feature matrix  

### Model
- Used cosine similarity to measure closeness between books  
- Precomputed and saved the similarity matrix for fast lookups  
- Implemented `recommend()` to return similar books with title, author, rating, description snippet, and Goodreads link  

### UI
- Built with Streamlit  
- Dropdown autocomplete for book titles  
- Sidebar filters for number of recommendations and rating ranking  
- Two-column layout with book cards showing:  
  - Title (clickable to Goodreads)  
  - Author and rating (stars)  
  - Short description with expandable text  

## Demo
- Kaggle Notebook: [Book Recommendation System](https://www.kaggle.com/code/gitanjalisoni/book-rec)  
- Deployed link: ()  

## Tech Stack
- Python (pandas, numpy, scikit-learn)  
- Streamlit (interactive UI)  
- Pickle (saving models/features)  

## Features
- Search any book and get recommendations  
- Adjustable number of recommendations  
- Option to rank results by rating  
- Clean and simple UI with clickable book cards  

## Dataset
Goodreads dataset from Kaggle  
