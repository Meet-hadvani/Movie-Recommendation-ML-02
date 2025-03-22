# Movie-Recommendation-02

TF-IDF and Cosine Similarity 

#SIMA

⚙️ Step-by-Step Recommendation Logic
1. Input: overview text (movie plot summary)

Each movie has a description like:

"A young man with a strong sense of justice becomes a superhero after his parents are killed..."
This becomes the feature we use to represent the movie.

2. Vectorization: TF-IDF

TF-IDF transforms the text into a numeric vector.
It gives higher weight to unique and important words, and down-weights common words.

3. Similarity Measurement: Cosine Similarity

Cosine similarity calculates the angle between two TF-IDF vectors.
A smaller angle (closer to 1.0) means the movies are more similar.
For each movie, the system:
Finds the cosine similarity between the input movie and all others.
Ranks them in descending order of similarity.

4. Output: Top-N Similar Movies

The top N similar movies (excluding itself) are returned.
These are the ones whose textual descriptions are most similar.

