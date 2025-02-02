import pandas as pd
import numpy as np
from scipy.linalg import svd
import sys
from tabulate import tabulate

# Constants
TOP_MOVIES = 10
GENRES = {
    1: "Action",
    2: "Romance",
    3: "Thriller",
    4: "War",
    5: "Animation",
    6: "Crime",
    7: "Horror",
    8: "History",
    9: "Adventure",
    10: "Sport"
}

# Display genre options
print("=============================================")
print("=            WELCOME TO CINEMA              =")
print("=============================================")
for num, genre in GENRES.items():
    print(f"{num}. {genre}")

# Get user input
try:
    genre_selection = int(input("Please select a genre (1-10): "))
    if genre_selection not in GENRES:
        raise ValueError("Invalid selection. Please choose a number between 1 and 10.")
except ValueError as e:
    print(e)
    sys.exit(1)

# Get selected genre file
selected_genre = GENRES[genre_selection]
file_name = f"{selected_genre}.csv"

# Read the CSV file with encoding fixes
try:
    df = pd.read_csv(file_name, encoding='utf-8', encoding_errors='replace')  # ✅ Fixed encoding
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    df_numeric["Movie"] = df["Movie"]  # Preserve movie names
except Exception as e:
    print(f"An error occurred while reading {file_name}: {e}")
    sys.exit(1)

# Perform Singular Value Decomposition (SVD)
U, s, Vt = svd(df_numeric.drop(columns=["Movie"]), full_matrices=False)
S = np.diag(s)
reconstructed_matrix = np.dot(U, np.dot(S, Vt))

# Create a DataFrame from the reconstructed matrix
reconstructed_df = pd.DataFrame(reconstructed_matrix, index=df_numeric.index, columns=df_numeric.columns[:-1])

# Calculate average ratings and recommend top movies
average_ratings = reconstructed_df.mean(axis=1)
recommended_movies = average_ratings.sort_values(ascending=False).head(TOP_MOVIES)

# Display recommendations
print(f"\nTop Recommended {selected_genre} Movies:")

table_data = []
for idx, movie_index in enumerate(recommended_movies.index, start=1):
    movie_name = df.loc[movie_index, "Movie"]
    director = df.loc[movie_index, 'Director']
    runtime = df.loc[movie_index, 'Runtime']
    release_date = df.loc[movie_index, 'Release']
    table_data.append([idx, movie_name, director, runtime, release_date])

# Print results
print(tabulate(table_data, headers=["#", "Movie", "Director", "Runtime", "Release Date"], tablefmt="pretty"))
