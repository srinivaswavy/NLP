import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

lemmatizer = WordNetLemmatizer()
vectorizer = CountVectorizer()


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


def vectorize(s: str):
    words = s.lower().split()
    words_and_tags = nltk.pos_tag(words)
    return [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in words_and_tags]


with open('data', 'r') as datafile:
    data = map(
        lambda s: " ".join(vectorize(s)), datafile.readlines()
    )
    X = vectorizer.fit_transform(list(data))
vocabulary = vectorizer.get_feature_names_out()

print(X)

print(X.toarray())
print(vocabulary)
