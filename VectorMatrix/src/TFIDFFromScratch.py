import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize

df = pd.read_csv('bbc_text_cls.csv')

# nltk.download("punkt")

# print(df.head())

word_to_index = {}
tokenized_docs = []

i = 0

for doc in df["text"]:
    tokens = word_tokenize(doc.lower())
    doc_as_int = []
    for token in tokens:
        if token not in word_to_index:
            word_to_index[token] = i
            i = i + 1

        doc_as_int.append(word_to_index[token])
    tokenized_docs.append(doc_as_int)

words = list(word_to_index.keys())

# print(tokenized_docs[1])

N = len(df["text"])
V = len(words)

print((N, V))

tf = np.zeros((N, V))

for i, doc in enumerate(tokenized_docs):
    for j in doc:
        tf[i][j] += 1

print(tf)

idf = np.log(N / np.sum(tf > 0, axis=0))

print(idf)

tf_idf = tf * idf

print(tf_idf)

random_i = np.random.choice(N)

tf_idf_vector = tf_idf[random_i]

print("random_i", random_i)

doc = df.iloc[random_i]

print("news", doc['text'])

print("label", doc['labels'])

print("Top 5 items: ")

indices_for_frequent_terms = (-tf_idf_vector).argsort()[0:5]

for i in indices_for_frequent_terms:
    print(words[i], sep=" ")
