from numpy import *

data = array(["Hi! How are you?", "I am Srini", "After that it's all blah blah blah"])

flattenedWordData = list(map(lambda item: item.split(" "), data))

uniqueWords = unique([item for row in flattenedWordData for item in row])

vocabulary = {}
i=0
for item in uniqueWords:
    vocabulary[item] = i
    i+=1

print(vocabulary)

def transformTextToVector( str:str):
    vector = zeros(len(vocabulary),int)
    for word in str.split(" "):
        index = vocabulary[word]
        vector[index] = vector[index]+1
    return vector

vectors = []

for sentence in data:
    vectors.append(transformTextToVector(sentence))

print(array(vectors))







