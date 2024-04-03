import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import random
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("bbc_text_cls.csv")

articles = df["text"]

article = random.choice(articles)
# article = articles[0]

sentenses = sent_tokenize(article)

sentenses = [sentense.lower() for sentense in sentenses]
title = sentenses[0]

sentenses = sentenses[1:]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        words_and_tags = nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word, get_wordnet_pos(pos)) for word, pos in words_and_tags]


tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words='english')

X = tfidf.fit_transform(sentenses)
numner_of_sentences = X.shape[0]

similarity_matrix = cosine_similarity(X)

# for i in range(numner_of_sentences):
#     for j in range(numner_of_sentences):
#         similarity_matrix[i, j] = cosine_similarity(X[i], X[j])

U = (np.zeros((numner_of_sentences, numner_of_sentences)) + (1 / numner_of_sentences))

similarity_matrix = 0.15 * U + 0.85 * (similarity_matrix / similarity_matrix.sum(axis=1, keepdims=True))

assert np.allclose(similarity_matrix.sum(axis=1), 1)

# calculate eigenvector of similarity matrix for eigen value 1

eigen_values, eigen_vectors = np.linalg.eig(similarity_matrix.T)

print("eigen values", eigen_values)
print("eigen vectors", eigen_vectors)

index = np.where(np.isclose(eigen_values, 1))

print("index", index)

eigen_vector = eigen_vectors[:, index].flatten()

print("eigen vector", eigen_vector)

print("eigen vector", eigen_vector.dot(similarity_matrix))

top_sentences_indices = np.argsort(-eigen_vector)[:5]

print("top_sentences_indices", top_sentences_indices)

top_sentences_indices = np.sort(top_sentences_indices)

print('title: ', title)
print('sentenses: \n')
i = 0
for sentense in sentenses:
    print(i, sentense, "\n")
    i += 1

print("top_sentences_indices: ", top_sentences_indices)

print("summary: \n", "\n".join(np.take(sentenses, top_sentences_indices)))
