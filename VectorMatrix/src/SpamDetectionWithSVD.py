import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


df = pd.read_csv("/Users/csriniv6/Downloads/spam.csv", encoding='latin-1')
sample_email = df.iloc[0]
df = df.rename(columns={"v2": "text", "v1": "Y"})
df["Y"] = df["Y"].apply(lambda x: 1 if x == 'spam' else 0)
X = df['text']
Y = df["Y"]

inputs_train, inputs_test, y_train, y_test = train_test_split(X, Y, random_state=123)

count_vectorizer = CountVectorizer()

truncated_svd = TruncatedSVD(n_components=500)

X_train = count_vectorizer.fit_transform(inputs_train)
X_test = count_vectorizer.transform(inputs_test)

# X_train = truncated_svd.fit_transform(count_vectorizer.fit_transform(inputs_train))
# X_test = truncated_svd.transform(count_vectorizer.transform(inputs_test))

model = MultinomialNB()

# model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

predicted_Y = model.predict(X_test)

print('f1_score:', f1_score(y_test, predicted_Y))

print("confusion matrix:\n", confusion_matrix(y_test, predicted_Y, labels=[1, 0]))

print("precision:", precision_score(y_test, predicted_Y))
print("recall:", recall_score(y_test, predicted_Y))
fpr, tpr, thresholds = roc_curve(y_test, predicted_Y)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

predict_probabilities = model.predict_proba(X_test)

print(predict_probabilities[:10])

predicted_labels = (predict_probabilities[:, 1] >= 0.85).astype(int)

print("F1 score:", f1_score(y_test, predicted_labels, pos_label=1))


