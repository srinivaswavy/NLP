import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_curve
from wordcloud import WordCloud, STOPWORDS
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import string

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


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
        return [self.wnl.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in words_and_tags]


df = pd.read_csv('/Users/csriniv6/Downloads/AirlineTweets.csv')
data = df[['text', 'airline_sentiment']]

data = data[data['airline_sentiment'] != 'neutral']

data['text'] = data['text'].apply(lambda x: x.lower())

wordcloud = WordCloud().generate(data[data['airline_sentiment'] == 'negative']['text'].str.cat(sep=' '))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

data['sentiment'] = data['airline_sentiment'].apply(lambda x: 1 if x == 'positive' else (0 if x == 'neutral' else -1))
data = data[['text', 'sentiment']]
Y = data['sentiment']
# remove punctuation
X = data['text'] #.apply(lambda line: line.lower().translate(str.maketrans('', '', string.punctuation)))
inputs_train, inputs_test, Y_Train, Y_Test = train_test_split(X, Y, random_state=123)

#vectorizer = CountVectorizer(stop_words='english', tokenizer=LemmaTokenizer(), ngram_range=(1, 2))
vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())
X_Train = vectorizer.fit_transform(inputs_train)
X_Test = vectorizer.transform(inputs_test)

model = LogisticRegression(max_iter=500,class_weight='balanced')

model.fit(X_Train, Y_Train)

# print weights of words

weights = model.coef_[0]

vocabulary = vectorizer.get_feature_names_out()

word_weights = dict(zip(vocabulary, weights))

for word, weight in sorted(word_weights.items(), key=lambda x: x[1])[0:30]:
    print(word, weight)

confusion_matrix = metrics.confusion_matrix(Y_Test,
                                            [1 if i > 0.80 else -1 for i in model.predict_proba(X_Test)[:, 1]])

for i in range(len(inputs_test)):
    actual = Y_Test.array[i]
    predicted = model.predict(X_Test[i])
    if actual != predicted:
        print(inputs_test.array[i], actual, predicted)

f1score = metrics.f1_score(Y_Test, model.predict(X_Test), average='weighted')

fpr, tpr, thresholds = roc_curve(Y_Test, model.predict(X_Test))

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

plt.show()

print(f1score)

print(confusion_matrix)
