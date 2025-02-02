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
            movie_name = df.loc[movie_index, "movie"]
            runtime = df.loc[movie_index, "runtime"] if "runtime" in df.columns else "Unknown"
            director = df.loc[movie_index, "director"] if "director" in df.columns else "Unknown"
            release_date = df.loc[movie_index, "release"] if "release" in df.columns else "Unknown"

            top_movies.append([movie_name, director, runtime, release_date, file_name])
    
    except Exception as e:
        print(f"⚠ Error loading {file_name}.csv: {e}")

# Display the top 20 movies
if top_movies:
    print("\n🎬 Top 20 Movies Across All Genres:")
    print(tabulate(top_movies, headers=["Movie", "Director", "Runtime", "Release Date", "Genre"], tablefmt="pretty"))