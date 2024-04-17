import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF as nmf
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import numpy as np
import textwrap

nltk.download('stopwords')
df = pd.read_csv('bbc_text_cls.csv')
print(df.head())
tf_idf_vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

tf_idf_vectorizer_model = tf_idf_vectorizer.fit(df['text'])

X = tf_idf_vectorizer_model.transform(df['text'])
print(X)


def plot_top_words(model, feature_names, n_top_words=10):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle('LDA', fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


NMF_model = nmf(n_components=10,
                beta_loss='kullback-leibler',
                solver='mu',
                random_state=0)
NMF_model.fit(X)

feature_names = tf_idf_vectorizer.get_feature_names_out()
plot_top_words(NMF_model, feature_names);

Z = NMF_model.transform(X)

np.random.seed(0)
i = np.random.choice(len(df))
z = Z[i]
topics = np.arange(10) + 1

fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels']);


def wrap(x):
    return textwrap.fill(x, replace_whitespace=False, fix_sentence_endings=True)


print(wrap(df.iloc[i]['text']))

i = np.random.choice(len(df))
z = Z[i]

fig, ax = plt.subplots()
ax.barh(topics, z)
ax.set_yticks(topics)
ax.set_title('True label: %s' % df.iloc[i]['labels'])

plt.show()
