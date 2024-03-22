import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("mice"))
print(lemmatizer.lemmatize("replacement", pos=wordnet.ADJ))
print(lemmatizer.lemmatize("is", pos=wordnet.NOUN))
print(lemmatizer.lemmatize("blowing", pos=wordnet.VERB))
print(lemmatizer.lemmatize("going"))
print(lemmatizer.lemmatize("went", pos=wordnet.VERB))
