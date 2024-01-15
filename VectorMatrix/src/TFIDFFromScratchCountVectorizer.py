import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('bbc_text_cls.csv')
count_vectorizer = CountVectorizer()

tf = count_vectorizer.fit_transform(df["text"])

#convert sparse matrix to dense matrix
#tf = tf.toarray()

# Get the list of words
words = count_vectorizer.get_feature_names_out()

print(words, 'words')
print(tf, 'tf')

N = len(df["text"])
V = len(words)

print((N, V))

idf = np.log(N / np.sum(tf > 0, axis=0))

# covert above idf to a vector
idf = np.squeeze(np.asarray(idf))

print(idf.shape, 'idf')

# multiply a sparse matrix with a dense vector
tf_idf = tf.multiply(idf)

random_i = np.random.choice(N)

print("tf_idf_shape", tf_idf.shape)


# convert coo_matrix to csr_matrix
tf_idf = tf_idf.tocsr()

tf_idf_vector = tf_idf[random_i]

tf_idf_vector = tf_idf_vector.toarray().flatten()

print(tf_idf_vector, 'tf_idf_vector')

print("random_i", random_i)

doc = df.iloc[random_i]

print("news", doc['text'].split("\n")[0])

print("label", doc['labels'])

print("Top 5 items: ")

indices_for_frequent_terms = (-tf_idf_vector).argsort()[0:5]

print(indices_for_frequent_terms, 'indices_for_frequent_terms')

for i in indices_for_frequent_terms:
    print(words[i], sep=" ")
