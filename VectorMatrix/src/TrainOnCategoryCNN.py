import random

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from keras.layers import Dense, Input, Conv1D, Embedding, GlobalMaxPooling1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

from matplotlib import pyplot as plt

df = pd.read_csv('bbc_text_cls.csv')
print(df.head())
df["Y"] = df["labels"].astype("category").cat.codes

K = df["Y"].max() + 1

print("Category count", K)

print(df.head())
df_train, df_test = train_test_split(df, random_state=123)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(df_train["text"])

V = tokenizer.num_words

X_train = pad_sequences(tokenizer.texts_to_sequences(df_train["text"]))
Y_train = df_train["Y"]

T = X_train.shape[1]

X_test = pad_sequences(tokenizer.texts_to_sequences(df_test["text"]), T)
Y_test = df_test["Y"]

print("X Test", X_test[0:2])

print(tokenizer.sequences_to_texts(X_train[0:10]))

print(X_train[0:3])

i = Input((T,))

D = 50

x = Embedding(V, D)(i)
x = Conv1D(32, 3, activation="relu")(x)
x = MaxPooling1D(3)(x)
x = Conv1D(64, 3, activation="relu")(x)
# x = MaxPooling1D(3)(x)
# x = Conv1D(128, 3, activation="relu")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(K, activation="softmax")(x)

model = Model(i, x)

print(model.summary())

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"]
)

r = model.fit(
    X_train,
    Y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, Y_test)
)

plt.plot(r.history["loss"], label="training loss")
plt.plot(r.history["val_loss"], label="validation loss")
plt.legend()
plt.show()

plt.plot(r.history["accuracy"], label="training accuracy")
plt.plot(r.history["val_accuracy"], label="val accuracy")
plt.legend()
plt.show()

P_train = np.apply_along_axis(np.argmax, 1, model.predict(X_train))
P_test = np.apply_along_axis(np.argmax, 1, model.predict(X_test))

cm = confusion_matrix(Y_train, P_train)

print(cm)

cm = confusion_matrix(Y_test, P_test)

print(cm)

print("Train F1:", f1_score(Y_train, P_train, average='weighted'))
print("Test F1:", f1_score(Y_test, P_test, average='weighted'))

model.save("/Users/csriniv6/Desktop/CategoryModel")
model.save_weights("/Users/csriniv6/Desktop/CategoryModel_weights")

sample = df.sample(10)

for index in range(10):
    row = sample.iloc[index]
    print("=================================================")
    print("article:", row["text"])
    print("label:", row["labels"])
    print("Y", row["Y"])
    predictions = model.predict(pad_sequences(tokenizer.texts_to_sequences([row["text"]]), maxlen=T))
    print("prediction:", predictions)
    print("predicted label:", np.argmax(predictions))
