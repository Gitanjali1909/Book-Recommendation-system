import pandas as pd


df = pd.read_csv("data/goodreads_data.csv")
print("Loaded raw data:", df.shape)
print("Columns:", df.columns.tolist())

df.drop_duplicates(inplace=True)

df.dropna(subset=["Book"], inplace=True)
df["Author"] = df["Author"].fillna("Unknown")
df["Description"] = df["Description"].fillna("")
df["Genres"] = df["Genres"].fillna("Unknown")

df["Genres"] = df["Genres"].apply(lambda x: str(x).strip("[]").replace("'", ""))

df["Num_Ratings"] = df["Num_Ratings"].astype(str).str.replace(",", "", regex=True)
df["Num_Ratings"] = pd.to_numeric(df["Num_Ratings"], errors="coerce").fillna(0).astype(int)

df = df.sort_values("Num_Ratings", ascending=False).head(5000)
df.reset_index(drop=True, inplace=True)

df.to_csv("data/books_cleaned.csv", index=False)
print("Cleaned dataset saved as books_cleaned.csv:", df.shape)
