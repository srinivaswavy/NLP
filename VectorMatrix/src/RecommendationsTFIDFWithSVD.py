import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD

df = pd.read_csv("tmdb_5000_movies.csv")
movieDoc = df.iloc[0]


def get_keywords_and_genres(document):
    genres = ' '.join(''.join(g['name'].split()) for g in json.loads(document['genres']))
    keywords = ' '.join(''.join(g['name'].split()) for g in json.loads(document['keywords']))
    return "%s %s" % (genres, keywords)


df['data'] = df.apply(get_keywords_and_genres, axis=1)

tf_idf_vectorizer = TfidfVectorizer(max_features=2000)

truncated_SVD = TruncatedSVD(n_components=200)

X = tf_idf_vectorizer.fit_transform(df['data'])

X = truncated_SVD.fit_transform(X)

print(X.shape)

movie_docs_indices = np.random.choice(df.shape[0], 5)

movies = df.iloc[movie_docs_indices]

for index in movie_docs_indices:
    print("Source movie:\n============================\n", df.iloc[index]["title"] + " -- " + df.iloc[index]["data"])
    print()
    scores = cosine_similarity([X[index]], X)
    scores = scores.flatten()
    recommended_indices = (-scores).argsort()[0:6]
    print(recommended_indices)
    print("recommendations:\n============================\n")
    for recomm_index in recommended_indices:
        print(df.iloc[recomm_index]["title"] + " -- " + df.iloc[recomm_index]["data"])
