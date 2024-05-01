import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD

nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

df = pd.read_csv('/Users/csriniv6/Downloads/AirlineTweets.csv')

df = df[['text', 'airline_sentiment']]
df = df[df['airline_sentiment'] != 'neutral']
df = df.rename(columns={'airline_sentiment': 'Y'})
df['targets'] = df['Y'].astype("category").cat.codes

df_train, df_test = train_test_split(df, random_state=42, test_size=0.15)

inputs_train = df_train['text']
inputs_test = df_test['text']

targets_train = df_train['targets']
targets_test = df_test['targets']


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

truncated_svd = TruncatedSVD(n_components=100)

tf_idf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=LemmaTokenizer())
# X_train = truncated_svd.fit_transform(tf_idf_vectorizer.fit_transform(inputs_train))
# X_test = truncated_svd.transform( tf_idf_vectorizer.transform(inputs_test))

X_train = tf_idf_vectorizer.fit_transform(inputs_train)
X_test = tf_idf_vectorizer.transform(inputs_test)

# count_vectorizer = CountVectorizer(stop_words='english')
# X_train = count_vectorizer.fit_transform(inputs_train)
# X_test = count_vectorizer.transform(inputs_test)

X_train = X_train.toarray()
X_test = X_test.toarray()

K = df['targets'].max() + 1

print("K", K)

D = X_train.shape[1]

i = Input((D,), batch_size=120)

x = Dense(K, 'relu')(i)
x = Dense(K, 'softmax')(x)
# x = Dense(K, 'softmax')(x)

# x = Dense(1)(i)


model = Model(i, x)

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

r = model.fit(
    x=X_train,
    y=targets_train,
    batch_size=120,
    epochs=30,
    validation_data=(X_test, targets_test)
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

P_train = np.apply_along_axis(np.argmax, 1, model.predict(X_train))
P_test = np.apply_along_axis(np.argmax, 1, model.predict(X_test))

cm = confusion_matrix(targets_train, P_train)

print(cm)

cm = confusion_matrix(targets_test, P_test)

print(cm)

print("Train F1:", f1_score(targets_train, P_train, average='weighted'))
print("Test F1:", f1_score(targets_test, P_test, average="weighted"))
