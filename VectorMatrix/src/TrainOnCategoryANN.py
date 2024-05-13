import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np

from matplotlib import pyplot as plt

df = pd.read_csv('bbc_text_cls.csv')

print(df.head())
df['targets'] = df['labels'].astype("category").cat.codes

df_train, df_test = train_test_split(df, random_state=42)

inputs_train = df_train['text']
inputs_test = df_test['text']

targets_train = df_train['targets']
targets_test = df_test['targets']

tf_idf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train = tf_idf_vectorizer.fit_transform(inputs_train)
X_test = tf_idf_vectorizer.transform(inputs_test)

X_train = X_train.toarray()
X_test = X_test.toarray()

D = X_train.shape[1]

# number of classes
K = df['targets'].max() + 1

i = Input((D,), batch_size=120)

x = Dense(200, activation="relu")(i)
x = Dense(K)(x)

model = Model(i, x)

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
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

print("Train F1:", f1_score(targets_train, P_train, average="weighted"))
print("Test F1:", f1_score(targets_test, P_test, average="weighted"))
