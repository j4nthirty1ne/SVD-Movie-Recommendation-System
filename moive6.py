import pandas as pd
import numpy as np
import difflib
from scipy.linalg import svd
from tabulate import tabulate

TOP_MOVIES_PER_GENRE = 3

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

def find_closest_movies(df, search_query):
    movie_titles = df["movie"].dropna().unique().tolist()
    return difflib.get_close_matches(search_query, movie_titles, n=10, cutoff=0.3)

def search_movie():
    search_query = input("\n🎥 Enter the movie name to search: ").strip().lower()
    all_movies = []

    for genre, file_name in GENRES.items():
        try:
            df = pd.read_csv(f"{file_name}.csv", encoding="utf-8", encoding_errors="replace")
            df.columns = df.columns.str.lower().str.strip()
            df["genre"] = file_name
            all_movies.append(df)
        except Exception as e:
            print(f"⚠ Error loading {file_name}.csv: {e}")

    if not all_movies:
        print("❌ No movie data available.")
        return

    all_movies_df = pd.concat(all_movies, ignore_index=True)
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

    print("\n🎥 Found Movie(s):")
    print(tabulate(matches[["movie", "director", "runtime", "release", "genre"]], headers="keys", tablefmt="fancy_grid", showindex=False))

    first_movie = matches.iloc[0]
    movie_genre = first_movie["genre"]
    movie_director = first_movie["director"]
    related_movies = all_movies_df[(all_movies_df["genre"] == movie_genre) | (all_movies_df["director"] == movie_director)]
    related_movies = related_movies[~related_movies["movie"].isin(matches["movie"])].head(10)

    if not related_movies.empty:
        print("\n🎞️ Related Movies:")
        print(tabulate(related_movies[["movie", "director", "runtime", "release", "genre"]], headers="keys", tablefmt="fancy_grid", showindex=False))

def recommend_movies():
    print("\n🎬 Movie Recommendation System")
    print("════════════════════════════")

    genre_list = [f" {num}. {genre}" for num, genre in GENRES.items()]
    column_width = max(len(line) for line in genre_list) + 5

    print("\n📌 Choose a Genre for Recommendations:\n")
    for i in range(0, len(genre_list), 2):
        left_col = genre_list[i]
        right_col = genre_list[i + 1] if i + 1 < len(genre_list) else ""
        print(f"{left_col.ljust(column_width)} {right_col}")

    try:
        genre_selection = int(input("\n🎯 Enter Your Choice: "))
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

        U, s, Vt = svd(df_numeric.drop(columns=["movie"]), full_matrices=False)
        S = np.diag(s)
        reconstructed_matrix = np.dot(U, np.dot(S, Vt))
        reconstructed_df = pd.DataFrame(reconstructed_matrix, index=df_numeric.index, columns=df_numeric.columns[:-1])

        average_ratings = reconstructed_df.mean(axis=1)
        recommended_movies = average_ratings.sort_values(ascending=False).head(5)

        print(f"\n🎥 Top 5 Recommended {selected_genre} Movies:")
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

        U, s, Vt = svd(df_numeric.drop(columns=["movie"]), full_matrices=False)
        S = np.diag(s)
        reconstructed_matrix = np.dot(U, np.dot(S, Vt))
        reconstructed_df = pd.DataFrame(reconstructed_matrix, index=df_numeric.index, columns=df_numeric.columns[:-1])

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

if top_movies:
    genre_movies = {genre: [] for genre in GENRES.values()}
    for movie in top_movies:
        genre_movies[movie[4]].append(movie[0])

    table_data = []
    for genre, movies in genre_movies.items():
        movies_list = movies[:TOP_MOVIES_PER_GENRE]
        movies_list += ["-"] * (TOP_MOVIES_PER_GENRE - len(movies_list))
        table_data.append([genre] + movies_list)

    print("\n🎬 Best Recommended movies")
    headers = ["Genre", "Movie 1", "Movie 2", "Movie 3"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

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