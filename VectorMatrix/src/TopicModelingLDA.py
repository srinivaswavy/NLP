import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as lda
from nltk.corpus import stopwords

nltk.download('stopwords')
df = pd.read_csv('bbc_text_cls.csv')
print(df.head())
count_vectorizer = CountVectorizer(stop_words=stopwords.words('english'))

count_vectorizer_model = count_vectorizer.fit(df['text'])

X = count_vectorizer_model.transform(df['text'])
print(X)

lda_model = lda(n_components=10, random_state=0)
lda_model.fit(X)

print(lda_model.components_)

lda_model.transform(X)

