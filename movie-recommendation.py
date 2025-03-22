import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kagglehub

# Example dataset
# data = {
#     'title': ['The Matrix', 'John Wick', 'Avengers', 'Iron Man', 'Batman Begins'],
#     'description': [
#         'A computer hacker learns about the true nature of reality and his role in the war against its controllers.',
#         'An ex-hitman comes out of retirement to track down the gangsters that killed his dog.',
#         'Earth’s mightiest heroes must come together to stop a global threat.',
#         'A billionaire industrialist and genius inventor builds a high-tech suit to fight crime.',
#         'After training with his mentor, Batman begins his fight to free Gotham.'
#     ]
# }
# df = pd.DataFrame(data)
# df['description'] = df['description'].str.lower()

path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
csv_path = f"{path}/tmdb_5000_movies.csv"
df = pd.read_csv(csv_path)
#print(df)
df['overview'] = df['overview'].fillna("").str.lower()

#TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, num_recommendations=1):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

print("Recommendations for 'The Matrix':")
print(recommend('Man of Steel'))
