# Import Libraries
import pandas as pd
import numpy as np
import difflib
from scipy.linalg import svd
import sys
from tabulate import tabulate

# Constants
TOP_MOVIES_PER_GENRE = 3

# Load movie data from database
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

# Function to find the closest matching movies
def find_closest_movies(df, search_query):
    movie_titles = df["movie"].dropna().unique().tolist()  # Ensure unique titles
    matches = difflib.get_close_matches(search_query, movie_titles, n=10, cutoff=0.3)
    return matches

# Function to search for a movie
def search_movie():
    search_query = input("\n🎥 Enter the movie name to search: ").strip().lower()

    all_movies = []

    # Load movie data from CSV files
    for genre, file_name in GENRES.items():
        try:
            df = pd.read_csv(f"{file_name}.csv", encoding="utf-8", encoding_errors="replace")
            df.columns = df.columns.str.lower().str.strip()
            df["genre"] = file_name  # Add genre for reference
            all_movies.append(df)
        except Exception as e:
            print(f"⚠ Error loading {file_name}.csv: {e}")

    if not all_movies:
        print("❌ No movie data available.")
        return

    all_movies_df = pd.concat(all_movies, ignore_index=True)

    # Find exact matches
    matches = all_movies_df[all_movies_df["movie"].str.lower().str.contains(search_query, na=False)]

    if matches.empty:
        close_matches = find_closest_movies(all_movies_df, search_query)
        if close_matches:
            print(f"\n🔍 No exact match found for '{search_query}', but here are some similar movies:")
            for idx, match in enumerate(close_matches, start=1):
                print(f"{idx}. {match}")

            choice = input("\nEnter the number of the movie you meant (or press Enter to exit): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(close_matches):
                search_query = close_matches[int(choice) - 1]
                matches = all_movies_df[all_movies_df["movie"].str.lower() == search_query.lower()]
            else:
                print("❌ No valid selection. Exiting search.")
                return

    # Print found movies
    print("\n🎥 Found Movie(s):")
    print(tabulate(matches[["movie", "director", "runtime", "release", "genre"]], headers="keys", tablefmt="fancy_grid", showindex=False))

    # Extract the genre and director of the first found movie
    first_movie = matches.iloc[0]
    movie_genre = first_movie["genre"]
    movie_director = first_movie["director"]

    # Find related movies by genre and director
    related_movies = all_movies_df[(all_movies_df["genre"] == movie_genre) | (all_movies_df["director"] == movie_director)]

    # Remove already found movies from related movies and limit to 10
    related_movies = related_movies[~related_movies["movie"].isin(matches["movie"])].head(10)

    if not related_movies.empty:
        print("\n🎞️ Related Movies:")
        print(tabulate(related_movies[["movie", "director", "runtime", "release", "genre"]], headers="keys", tablefmt="fancy_grid", showindex=False))

# Function to recommend movies based on user choice
def recommend_movies():
    print("\n🎬 𝗠𝗼𝘃𝗶𝗲 𝗥𝗲𝗰𝗼𝗺𝗺𝗲𝗻𝗱𝗮𝘁𝗶𝗼𝗻 𝗦𝘆𝘀𝘁𝗲𝗺")
    print("════════════════════════════")

    # Group genres into two columns for better readability
    genre_list = [f" {num}. {genre}" for num, genre in GENRES.items()]
    column_width = max(len(line) for line in genre_list) + 5  # Adjust spacing

    print("\n📌 **Choose a Genre for Recommendations:**\n")

    # Print genres in two columns
    for i in range(0, len(genre_list), 2):
        left_col = genre_list[i]
        right_col = genre_list[i + 1] if i + 1 < len(genre_list) else ""
        print(f"{left_col.ljust(column_width)} {right_col}")

    try:
        genre_selection = int(input("\n🎯 **Enter Your Choice:** "))
        if genre_selection not in GENRES:
            raise ValueError("❌ Invalid selection. Please choose a valid genre number.")
    except ValueError as e:
        print(f"⚠ {e}")
        return

    selected_genre = GENRES[genre_selection]
    file_name = f"{selected_genre}.csv"

    try:
        df = pd.read_csv(file_name, encoding="utf-8", encoding_errors="replace")
        df.columns = df.columns.str.lower().str.strip()

        df_numeric = df.select_dtypes(include=[np.number]).copy()
        df_numeric["movie"] = df["movie"]

        # Perform Singular Value Decomposition (SVD) for better recommendations
        U, s, Vt = svd(df_numeric.drop(columns=["movie"]), full_matrices=False)
        S = np.diag(s)
        reconstructed_matrix = np.dot(U, np.dot(S, Vt))

        reconstructed_df = pd.DataFrame(reconstructed_matrix, index=df_numeric.index, columns=df_numeric.columns[:-1])

        # Get the top 5 recommended movies
        average_ratings = reconstructed_df.mean(axis=1)
        recommended_movies = average_ratings.sort_values(ascending=False).head(5)

        # Prepare table for display
        print(f"\n🎥 **Top 5 Recommended {selected_genre} Movies:**")
        table_data = []
        for idx, movie_index in enumerate(recommended_movies.index, start=1):
            movie_name = df.iloc[movie_index]["movie"]
            runtime = df.iloc[movie_index]["runtime"] if "runtime" in df.columns else "Unknown"
            director = df.iloc[movie_index]["director"] if "director" in df.columns else "Unknown"
            release_date = df.iloc[movie_index]["release"] if "release" in df.columns else "Unknown"

            table_data.append([idx, movie_name, director, runtime, release_date])

        print(tabulate(table_data, headers=["#", "Movie", "Director", "Runtime", "Release Date"], tablefmt="fancy_grid"))

    except Exception as e:
        print(f"⚠ Error reading {file_name}: {e}")


# Load all genres and get top 20 movies
print("\n=============================================")
print("🎬  WELCOME TO THE MOVIE RECOMMENDATION SYSTEM 🎬")
print("=============================================\n")

top_movies = []
for genre, file_name in GENRES.items():
    try:
        df = pd.read_csv(f"{file_name}.csv", encoding="utf-8", encoding_errors="replace")
        df.columns = df.columns.str.lower().str.strip()

        df_numeric = df.select_dtypes(include=[np.number]).copy()
        df_numeric["movie"] = df["movie"]

        # Perform SVD
        U, s, Vt = svd(df_numeric.drop(columns=["movie"]), full_matrices=False)
        S = np.diag(s)
        reconstructed_matrix = np.dot(U, np.dot(S, Vt))

        reconstructed_df = pd.DataFrame(reconstructed_matrix, index=df_numeric.index, columns=df_numeric.columns[:-1])

        # Get top 2 movies from each genre
        average_ratings = reconstructed_df.mean(axis=1)
        top_2_movies = average_ratings.sort_values(ascending=False).head(TOP_MOVIES_PER_GENRE)

        for movie_index in top_2_movies.index:
            movie_name = df.iloc[movie_index]["movie"]
            runtime = df.iloc[movie_index]["runtime"] if "runtime" in df.columns else "Unknown"
            director = df.iloc[movie_index]["director"] if "director" in df.columns else "Unknown"
            release_date = df.iloc[movie_index]["release"] if "release" in df.columns else "Unknown"

            top_movies.append([movie_name, director, runtime, release_date, file_name])

    except Exception as e:
        print(f"⚠ Error loading {file_name}.csv: {e}")

# Display the top 20 movies grouped by genre
if top_movies:
    genre_movies = {genre: [] for genre in GENRES.values()}  # Dictionary to store top movies per genre

    for movie in top_movies:
        genre_movies[movie[4]].append(movie[0])  # movie[4] = Genre, movie[0] = Movie name

    # Prepare data for tabulation
    table_data = []
    for genre, movies in genre_movies.items():
        movies_list = movies[:TOP_MOVIES_PER_GENRE]  # Only take the top 3 movies per genre
        movies_list += ["-"] * (TOP_MOVIES_PER_GENRE - len(movies_list))  # Fill empty slots with "-"
        table_data.append([genre] + movies_list)

    # Print the table
    print("\n🎬 Best Recommended movies")
    headers = ["Genre", "Movie 1", "Movie 2", "Movie 3"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))


# User choices
while True:
    print("\nWhat would you like to do next?")
    print("1️⃣   Search for a Movie")
    print("2️⃣   Get a Movie Recommendation")
    print("3️⃣   Exit")

    choice = input("Your choice: ").strip()
    if choice == "1":
        search_movie()
    elif choice == "2":
        recommend_movies()
    elif choice == "3":
        print("🎥 Goodbye! Enjoy your movie!")
        break
    else:
        print("❌ Invalid option. Please try again.")