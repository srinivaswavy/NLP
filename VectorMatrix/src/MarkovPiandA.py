import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize

df = pd.read_csv("bbc_text_cls.csv")

print(df.head())
word_to_index = {}
vocabulary = {}

index = 0
for item in df["text"]:
    for word in word_tokenize(item):
        if word not in word_to_index:
            word_to_index[word] = index
            vocabulary[word] = 1
            index += 1
        else:
            vocabulary[word] += 1

print(word_to_index["the"])

M = len(word_to_index.keys())

print('M', M)

A = np.zeros((M, M))
Pi = np.zeros(M)

for item in df["text"]:
    tokens = word_tokenize(item)
    Pi[word_to_index[tokens[0]]] += 1
    for i in range(len(tokens) - 1):
        A[word_to_index[tokens[i]], word_to_index[tokens[i + 1]]] += 1

for i in range(M):
    A[i] = np.log((A[i] + 1) / (A[i].sum() + M))

Pi = np.log((Pi + 1) / (Pi.sum() + M))

print('A', A)
print('Pi', Pi)

i = word_to_index['The']

print('the', Pi[i])

i = word_to_index['India']
print('India', Pi[i])

i = word_to_index['US']
print('US', Pi[i])
