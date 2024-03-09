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
Y = df['v1'].apply(lambda x: 1 if x == 'spam' else 0)

print(X.shape)

XTrain, XTest, YTrain, YTest = train_test_split(X, Y, random_state=123)

model = MultinomialNB()

model.fit(XTrain, YTrain)

print("train score:", model.score(XTrain, YTrain))

#print F1 score

from sklearn.metrics import f1_score

print("F1 score:", f1_score(YTest, model.predict(XTest), pos_label=1))

#print confusion matrix

from sklearn.metrics import confusion_matrix

print("Confusion matrix:", confusion_matrix(YTest, model.predict(XTest),labels=[1, 0]))

#print recall and precision

from sklearn.metrics import recall_score, precision_score

print("Recall:", recall_score(YTest, model.predict(XTest), pos_label=1))
print("Precision:", precision_score(YTest, model.predict(XTest), pos_label=1))

#AUC-ROC

from sklearn.metrics import roc_auc_score

print("AUC-ROC:", roc_auc_score(YTest, model.predict(XTest)))


print("test score:", model.score(XTest, YTest))

#plot ROC curve

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(YTest, model.predict(XTest))

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#selecting the best threshold

# from sklearn.metrics import precision_recall_curve
#
# precision, recall, thresholds = precision_recall_curve(YTest, model.predict(XTest))
#
# plt.plot(thresholds, precision[:-1], label="Precision")
# plt.plot(thresholds, recall[:-1], label="Recall")
# plt.xlabel('Threshold')
# plt.legend()
# plt.show()

#predict probabilities

probabilities = model.predict_proba(XTest)

print(probabilities[:10])

predicted_labels = (probabilities[:, 1] >= 0.941520).astype(int)

print("F1 score:", f1_score(YTest, predicted_labels, pos_label=1))
