from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import word_tokenize
import numpy as np


def MarKovProbability(Poem, A, Pi, word_to_index):
    tokens = word_tokenize(Poem)
    P = Pi[word_to_index[tokens[0]]]
    for i in range(len(tokens) - 1):
        P += A[word_to_index[tokens[i]], word_to_index[tokens[i + 1]]]
    return P


robert_word_to_index = {}
index = 0
robert_matrix = []
with open('/Users/csriniv6/Downloads/robert_frost.txt') as f:
    for line in f.readlines():
        if line.strip() != "":
            tokens = word_tokenize(line)
            index_vector = []
            for token in tokens:
                if token not in robert_word_to_index:
                    robert_word_to_index[token] = index
                    index += 1
                index_vector.append(robert_word_to_index[token])
            robert_matrix.append(index_vector)

robert_M = len(robert_word_to_index.keys())

index_to_word = np.array(list(robert_word_to_index.keys()))

print(robert_M)

print(" ".join(index_to_word[robert_matrix[0]]))

robert_A = np.zeros((robert_M, robert_M))
robert_Pi = np.zeros(robert_M)

print("matrix", robert_matrix)

for i in range(len(robert_matrix)):
    robert_Pi[robert_matrix[i][0]] += 1
    for j in range(len(robert_matrix[i]) - 1):
        robert_A[robert_matrix[i][j]][robert_matrix[i][j + 1]] += 1

print("M", robert_M)
print("Pi", robert_Pi)

print("A", robert_A)
