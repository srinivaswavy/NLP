from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import numpy as np
import string


def MarKovProbability(Poem, A, Pi, word_to_index):
    tokens = word_tokenize(Poem)
    logP = Pi[get_index_from_word(word_to_index, tokens[0])]
    for i in range(len(tokens) - 1):
        logP += A[get_index_from_word(word_to_index, tokens[i]), get_index_from_word(word_to_index, tokens[i + 1])]
    return logP


def get_index_from_word(word_to_index, word):
    if word in word_to_index:
        return word_to_index[word]
    else:
        return word_to_index["unknown999"]


def TokenizePoems(filePath):
    word_to_index = {}
    index = 0
    matrix = []
    with open(filePath) as f:
        lines = f.readlines()
        print('total lines', len(lines))
        XTrain, XTest = train_test_split(lines, random_state=123, test_size=0.15)
        print('train set size', len(XTrain))
        print('test set size', len(XTest))
        for line in XTrain:
            line = line.strip().lower()
            if line:
                line = line.translate(str.maketrans('', '', string.punctuation))
                tokens = word_tokenize(line)
                index_vector = []
                for token in tokens:
                    if token not in word_to_index:
                        word_to_index[token] = index
                        index += 1
                    index_vector.append(word_to_index[token])
                matrix.append(index_vector)
    word_to_index['unknown999'] = index
    return word_to_index, matrix, XTrain, XTest


def get_markov_probabilities(vector_matrix, vocabulary_size, epsilon=0.1):
    A = np.ones((vocabulary_size, vocabulary_size)) * epsilon
    Pi = np.ones(vocabulary_size) * epsilon
    for i in range(len(vector_matrix)):
        Pi[vector_matrix[i][0]] += 1
        for j in range(len(vector_matrix[i]) - 1):
            A[vector_matrix[i][j]][vector_matrix[i][j + 1]] += 1

    A /= A.sum(axis=1, keepdims=True)
    Pi /= Pi.sum()

    return np.log(A), np.log(Pi)


robert_word_to_index, robert_matrix, Xtrain, XTest = TokenizePoems('/Users/csriniv6/Downloads/robert_frost.txt')
edgar_word_to_index, edgar_matrix, edgar_Xtrain, edgar_XTest = TokenizePoems(
    '/Users/csriniv6/Downloads/edgar_allan_poe.txt')

print('XTrain', robert_matrix)
print('XTrain size', len(robert_matrix))

robert_M = len(robert_word_to_index.keys())
edgar_M = len(edgar_word_to_index.keys())
index_to_word = np.array(list(robert_word_to_index.keys()))

print(robert_M)

print(" ".join(index_to_word[robert_matrix[0]]))

print("matrix", robert_matrix)

robert_A, robert_Pi = get_markov_probabilities(robert_matrix, robert_M, epsilon=0.5)
edgar_A, edgar_Pi = get_markov_probabilities(edgar_matrix, edgar_M, epsilon=0.5)

print("Pi", robert_Pi)

print("A", robert_A)

print(MarKovProbability('I blah drew jug to say'.lower().translate(str.maketrans('', '', string.punctuation)), robert_A,
                        robert_Pi, robert_word_to_index))
poem = """as just as fair"""
processed_poem = poem.lower().translate(str.maketrans('', '', string.punctuation))
print(MarKovProbability(processed_poem, robert_A, robert_Pi, robert_word_to_index))

print(MarKovProbability(
    'Two roads diverged in a yellow wood,'.lower().translate(str.maketrans('', '', string.punctuation)), robert_A,
    robert_Pi, robert_word_to_index))
print(MarKovProbability(
    'Two roads diverged in a yellow wood,'.lower().translate(str.maketrans('', '', string.punctuation)), edgar_A,
    edgar_Pi, edgar_word_to_index))

print(MarKovProbability(
    'There shrines, and palaces, and towers'.lower().translate(str.maketrans('', '', string.punctuation)), robert_A,
    robert_Pi, robert_word_to_index))
print(MarKovProbability(
    'There shrines, and palaces, and towers'.lower().translate(str.maketrans('', '', string.punctuation)), edgar_A,
    edgar_Pi, edgar_word_to_index))

total = 0
errors = 0
for x in edgar_Xtrain:
    x = x.lower().translate(str.maketrans('', '', string.punctuation))
    print(x)
    if x.strip() != "":
        total += 1
        robertP = MarKovProbability(
            x, robert_A,
            robert_Pi, robert_word_to_index)
        edgarP = MarKovProbability(
            x, edgar_A,
            edgar_Pi, edgar_word_to_index)
        if robertP < edgarP:
            errors += 1

print(total, errors)
