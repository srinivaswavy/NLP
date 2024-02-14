import string
import numpy as np

pi = {}
a1 = {}
a2 = {}


def tokenize(file):
    with open(file, 'r') as file:
        lines = file.readlines()
        data = [line.strip().lower().translate(str.maketrans('', '', string.punctuation)).split() for line in lines if
                line.strip()]
    return data


poem_lines = tokenize('/Users/csriniv6/Downloads/robert_frost.txt')

#print(poem_lines[0:10])

for line in poem_lines:
    T = len(line)
    for i in range(T):
        word = line[i]
        if i == 0:
            if word not in pi:
                pi[word] = 0
            pi[word] += 1
        if i == 1:
            if line[i - 1] not in a1:
                a1[line[i - 1]] = {}
            if word not in a1[line[i - 1]]:
                a1[line[i - 1]][word] = 0
            a1[line[i - 1]][word] += 1
        if i == T - 1:
            if (line[i - 1], word) not in a2:
                a2[(line[i - 1], word)] = {}
                a2[(line[i - 1], word)]['END'] = 1
        if i > 1:
            if (line[i - 2], line[i - 1]) not in a2:
                a2[(line[i - 2], line[i - 1])] = {}
            if word not in a2[(line[i - 2], line[i - 1])]:
                a2[(line[i - 2], line[i - 1])][word] = 0
            a2[(line[i - 2], line[i - 1])][word] += 1

for word in pi:
    pi[word] = pi[word] / sum(pi.values())

for word in a1:
    s = sum(a1[word].values())
    for word2 in a1[word]:
        a1[word][word2] = a1[word][word2] / s

for word in a2:
    s = sum(a2[word].values())
    for word2 in a2[word]:
        a2[word][word2] = a2[word][word2] / s

# print('pi', pi)
# print('a1', a1)
# print('a2', a2)


def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for term, p in dictionary.items():
        cumulative += p
        if cumulative > p0:
            return term
    assert False


def generate():
    poem = []
    for i in range(4):
        poem_line = []
        prev_word = sample_word(pi)
        poem_line.append(prev_word)
        word = sample_word(a1[prev_word])
        while True:
            if word == 'END':
                break
            poem_line.append(word)
            next_word = sample_word(a2[(prev_word, word)])
            prev_word = word
            word = next_word

        poem.append(' '.join(poem_line))
    print("-----generated poem------")
    for line in poem:
        print(line)


generate()
