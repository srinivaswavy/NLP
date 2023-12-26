from nltk import PorterStemmer

porterStemmer = PorterStemmer()

print(porterStemmer.stem("Walking"))
print(porterStemmer.stem("Replacement"))