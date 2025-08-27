import pandas as pd

# 1. Load raw data
df = pd.read_csv("data/goodreads_data.csv")
print("✅ Loaded raw data:", df.shape)
print("📌 Columns:", df.columns.tolist())

# 2. Drop duplicates
df.drop_duplicates(inplace=True)

# 3. Handle missing values
df.dropna(subset=["Book"], inplace=True)            # remove rows without titles
df["Author"].fillna("Unknown", inplace=True)
df["Description"].fillna("", inplace=True)
df["Genres"].fillna("Unknown", inplace=True)

# 4. Parse/clean columns
# Genres are stored like: "['Classics', 'Fiction']"
df["Genres"] = df["Genres"].apply(lambda x: str(x).strip("[]").replace("'", ""))

# Convert Num_Ratings to integer (remove commas first)
df["Num_Ratings"] = df["Num_Ratings"].astype(str).str.replace(",", "", regex=True)
df["Num_Ratings"] = pd.to_numeric(df["Num_Ratings"], errors="coerce").fillna(0).astype(int)

# 5. Reset index
df.reset_index(drop=True, inplace=True)

# 6. Save cleaned dataset
df.to_csv("data/books_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as books_cleaned.csv:", df.shape)
