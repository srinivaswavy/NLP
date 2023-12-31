import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

df = pd.read_csv("tmdb_5000_movies.csv")
# print(df.head())

x = df.iloc[0]

# print(x)
#
# print(x["genres"])
# print(x["keywords"])

genresJson = json.loads(x["genres"])

# print(genresJson)

# print(' '.join(''.join(gj["name"].split()) for gj in genresJson))


def genres_and_keywords_to_string(row):
    genres = json.loads(row['genres'])
    genres = ' '.join(''.join(j['name'].split()) for j in genres)

    keywords = json.loads(row['keywords'])
    keywords = ' '.join(''.join(j['name'].split()) for j in keywords)
    return "%s %s" % (genres, keywords)


df['string'] = df.apply(genres_and_keywords_to_string, axis=1)

# print(df.head())

tfIdf = TfidfVectorizer(max_features=2000)

X = tfIdf.fit_transform(df["string"])

# print(len(X.toarray()[0]))

movieToIndex = pd.Series(df.index,index=df["title"])
index = movieToIndex["Scream 3"]
query = X[index]
# print(query)

scores = cosine_similarity(query,X)
scores = scores.flatten()

plt.plot(scores);

plt.show()

plt.plot( scores[(-scores).argsort()])
plt.show()

recommendedIndices = (-scores).argsort()[1:6]

print(recommendedIndices)

print(df["title"].iloc[recommendedIndices])


