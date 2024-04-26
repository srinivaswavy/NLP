import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import seaborn as sn

df = pd.read_csv("/Users/csriniv6/Downloads/spam.csv", encoding='latin-1')
sample_email = df.iloc[0]
df = df.rename(columns={"v2": "text", "v1": "Y"})
df["Y"] = df["Y"].apply(lambda x: 1 if x == 'spam' else 0)
inputs = df['text']
targets = df["Y"]

inputs_train, inputs_test, y_train, y_test = train_test_split(inputs, targets, random_state=42)

tf_idf_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')

X_train = tf_idf_vectorizer.fit_transform(inputs_train)
X_test = tf_idf_vectorizer.transform(inputs_test)

X_train = X_train.toarray()
X_test = X_test.toarray()

D = X_train.shape[1]

i = Input(shape=(D,))
x = Dense(1,activation='sigmoid')(i)

model = Model(i, x)

print(model.summary())

model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss=BinaryCrossentropy(from_logits=False),
    metrics=['accuracy']
)

r = model.fit(
    x=X_train,
    y=y_train.to_numpy(),
    batch_size=120,
    epochs=50,
    validation_data=(X_test, y_test.to_numpy())
)

plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend();
plt.show()

P_train = ((model.predict(X_train) >= 0.5) * 1.0).flatten()
P_test = ((model.predict(X_test) >= 0.5) * 1.0).flatten()

cm = confusion_matrix(y_train, P_train)

print(cm)

cm = confusion_matrix(y_test, P_test)

print(cm)

print("Train F1:", f1_score(y_train, P_train))
print("Test F1:", f1_score(y_test, P_test))




