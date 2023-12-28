import matplotlib
import nltk
import numpy
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("omw-1.4")

df = pd.read_csv('bbc_text_cls.csv')

print(df.head())

inputs = df['text']
labels = df['labels']

# labels.hist(figsize=(10,5));
#
# matplotlib.pyplot.show()

inputs_train, inputs_test, YTrain, YTest = train_test_split(inputs, labels, random_state=123)

vectorizer = CountVectorizer()

Xtrain = vectorizer.fit_transform(inputs_train)
XTest = vectorizer.transform(inputs_test)

print("Simple count vectorizer dimentionality:", Xtrain.shape)

print((Xtrain != 0).sum())

print((Xtrain != 0).sum() / numpy.prod(Xtrain.shape))

model = MultinomialNB()
model.fit(Xtrain, YTrain)
print("train score:", model.score(Xtrain, YTrain))
print("test score:", model.score(XTest, YTest))

print(model.predict(vectorizer.transform(["I finally earned good dollars"])))
print(model.predict(vectorizer.transform(["Why politicians often lie?"])))

vectorizer = CountVectorizer(stop_words="english")
Xtrain = vectorizer.fit_transform(inputs_train)
XTest = vectorizer.transform(inputs_test)

print(Xtrain.shape)

model.fit(Xtrain, YTrain)
print("train score:", model.score(Xtrain, YTrain))
print("test score:", model.score(XTest, YTest))

print("Stopwords count vectorizer dimentionality:", Xtrain.shape)


def get_wordnet_pos(treebank_tag: str):
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
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in words_and_tags]


vectorizer = CountVectorizer(tokenizer=LemmaTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
XTest = vectorizer.transform(inputs_test)

model = MultinomialNB()
model.fit(Xtrain, YTrain)
print("Lemma tokenizer model train score:", model.score(Xtrain, YTrain))
print("Lemma tokenizer model test score:", model.score(XTest, YTest))
print("LemmaTokenizer dimensionality:", Xtrain.shape)


class PorterStemmerTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()

    def __call__(self, doc):
        tokens = word_tokenize(doc)
        return [self.porter.stem(token) for token in tokens]


vectorizer = CountVectorizer(tokenizer=PorterStemmerTokenizer())
Xtrain = vectorizer.fit_transform(inputs_train)
XTest = vectorizer.transform(inputs_test)

model = MultinomialNB()
model.fit(Xtrain, YTrain)
print("Stemmertokenizer model train score:", model.score(Xtrain, YTrain))
print("Stemmertokenizer model test score:", model.score(XTest, YTest))
print("Stemmertokenizer dimentionality:", Xtrain.shape)
