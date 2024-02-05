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
            tokens = word_tokenize(line)
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

    return pi, a, a2


pi, a, a2 = get_markov_probabilities(tokenize('/Users/csriniv6/Downloads/robert_frost.txt'))

print('pi', pi)
print('a', a)
print('a2', a2)
