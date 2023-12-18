from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
datafile = open('data','r')
X = vectorizer.fit_transform(datafile)
vocabolary = vectorizer.get_feature_names_out()
print(X.toarray())
print(vocabolary)