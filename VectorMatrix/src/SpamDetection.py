import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# read from csv with more rich character encoding

df = pd.read_csv("/Users/csriniv6/Downloads/spam.csv", encoding='latin-1')

print(df['v1'].head())
print(df['v2'].head())

tfidf = TfidfVectorizer()
count_vectorizer = CountVectorizer()

X = count_vectorizer.fit_transform(df['v2'])
#X = tfidf.fit_transform(df['v2'])
Y = df['v1']

print(X.shape)

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, random_state=123)

model = MultinomialNB()

model.fit(XTrain, YTrain)

print("train score:", model.score(XTrain, YTrain))

print("test score:", model.score(XTest, YTest))
