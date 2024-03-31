import random

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np

df = pd.read_csv('bbc_text_cls.csv')
print(df.head())
inputs = df[df['labels'] == 'business']['text']

a2 = {}

for line in inputs:
    tokens = word_tokenize(line.lower().rstrip())
    if len(tokens) < 3:
        continue
    for i in range(1, len(tokens) - 2):
        if (tokens[i - 1], tokens[i + 1]) not in a2:
            a2[(tokens[i - 1], tokens[i + 1])] = {}
        a2[(tokens[i - 1], tokens[i + 1])][tokens[i]] = a2[(tokens[i - 1], tokens[i + 1])].get(tokens[i], 0) + 1

for k, v in a2.items():
    sum_counts = sum(v.values())
    for k1, v1 in v.items():
        v[k1] = v1 / sum_counts

for k, v in list(a2.items())[0:10]:
    print(k, v)


def sample_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for term, p in dictionary.items():
        cumulative += p
        if cumulative > p0:
            return term
    assert False


detokenizer = TreebankWordDetokenizer()


def spin_article(article):
    lines = article.split("\n")
    spun_article = []
    for line in lines:
        spun_article.append(spin_line(line))
    return detokenizer.detokenize(tokens=spun_article)


def spin_line(line):
    tokens = word_tokenize(line.lower().rstrip())
    if len(tokens) < 3:
        return line
    i = 0
    spun_line = [tokens[0]]
    while i < len(tokens) - 2:
        t_1 = tokens[i]
        t_2 = tokens[i + 1]
        t_3 = tokens[i + 2]
        probs = a2.get((t_1, t_3))

        if not (probs is None or len(probs.values()) < 2) and np.random.randn() < 0.3:
            spun_line.append(t_2)
            spun_line.append('<' + sample_word(probs) + '>')
            spun_line.append(t_3)
            i += 2
        else:
            spun_line.append(t_2)
            i += 1
    return detokenizer.detokenize(tokens=spun_line)


random_article = inputs.iloc[random.randint(0, len(inputs))]

print("Random article: ", random_article)

print("Spun article: ", spin_article(random_article))

#
# spinned_article = []
#
# random_index = random.randint(0, len(inputs))
# print("Random index: ", random_index)
#
# random_article = df["text"].values[random_index]
# print("Random article: ", random_article)
#
# tokens = word_tokenize(random_article.lower().rstrip())
# if len(tokens) < 3:
#     print("Article is too short")
#     exit(1)
#
# i = 0
#
# while i < len(tokens) - 2:
#     t_1 = tokens[i]
#     t_2 = tokens[i + 1]
#     t_3 = tokens[i + 2]
#     probs = a2.get((t_1, t_3))
#
#     if not (probs is None or len(probs.values()) < 2):
#         spinned_article.append(t_1)
#         spinned_article.append(sample_word(probs))
#         spinned_article.append(t_3)
#         i += 2
#
#
#
# detokenizer = TreebankWordDetokenizer()
#
# print('spinned article',  detokenizer.detokenize(tokens=spinned_article))
