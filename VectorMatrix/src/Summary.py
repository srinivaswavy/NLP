import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
import random
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("bbc_text_cls.csv")

articles = df["text"]

article = random.choice(articles)
print('article', article)

sentenses = sent_tokenize(article)
print('sentenses', sentenses)


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


tfidf = TfidfVectorizer(tokenizer=LemmaTokenizer())

X = tfidf.fit_transform(sentenses)
non_zero_counts_per_row = np.diff(X.indptr)
sums = np.squeeze(np.asarray(X.sum(axis=1)))

average_non_zero_elements_per_row = np.array(
    [-1 if count == 0 else sum / count for sum, count in zip(sums, non_zero_counts_per_row)])

print(average_non_zero_elements_per_row)

top_sentences_indices = np.argsort(-average_non_zero_elements_per_row)[:5]

print("summary", "".join(np.take(sentenses, top_sentences_indices)))
