from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
with open('data','r') as datafile:
    X = vectorizer.fit_transform(datafile)
vocabolary = vectorizer.get_feature_names_out()
print(X.toarray())
print(vocabolary)