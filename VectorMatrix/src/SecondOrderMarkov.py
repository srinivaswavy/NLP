import string
from nltk import word_tokenize


def tokenize(file):
    with open(file, 'r') as file:
        data = file.readlines()
    return data


def get_markov_probabilities(data):
    pi = {}
    a = {}
    a2 = {}
    for line in data:
        line = line.strip().lower()
        if line:
            # remove punctuation
            line = line.translate(str.maketrans('', '', string.punctuation))
            tokens = line.split()
            if tokens[0] in pi:
                pi[tokens[0]] += 1
            else:
                pi[tokens[0]] = 1

            for i in range(len(tokens) - 1):
                if tokens[i] in a:
                    if tokens[i + 1] in a[tokens[i]]:
                        a[tokens[i]][tokens[i + 1]] += 1
                    else:
                        a[tokens[i]][tokens[i + 1]] = 1
                else:
                    a[tokens[i]] = {}
                    a[tokens[i]][tokens[i + 1]] = 1

            for i in range(len(tokens) - 2):
                if tokens[i] in a2:
                    if tokens[i + 1] in a2[tokens[i]]:
                        if tokens[i + 2] in a2[tokens[i]][tokens[i + 1]]:
                            a2[tokens[i]][tokens[i + 1]][tokens[i + 2]] += 1
                        else:
                            a2[tokens[i]][tokens[i + 1]][tokens[i + 2]] = 1
                    else:
                        a2[tokens[i]][tokens[i + 1]] = {}
                        a2[tokens[i]][tokens[i + 1]][tokens[i + 2]] = 1
                else:
                    a2[tokens[i]] = {}
                    a2[tokens[i]][tokens[i + 1]] = {}
                    a2[tokens[i]][tokens[i + 1]][tokens[i + 2]] = 1

    for word in pi:
        pi[word] = pi[word] / sum(pi.values())

    for word in a:
        for word2 in a[word]:
            a[word][word2] = a[word][word2] / sum(a[word].values())
    for word in a2:
        for word2 in a2[word]:
            for word3 in a2[word][word2]:
                a2[word][word2][word3] = a2[word][word2][word3] / sum(a2[word][word2].values())

    return pi, a, a2


pi, a, a2 = get_markov_probabilities(tokenize('/Users/csriniv6/Downloads/robert_frost.txt'))

print('pi', pi)
print('a', a)
print('a2', a2)
