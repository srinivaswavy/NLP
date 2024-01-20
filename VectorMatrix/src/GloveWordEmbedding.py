import numpy as np

# import cosine_similarity from sklearn.metrics.pairwise

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

vector_array = []


def loadGloveVectors(filePath: str):
    word_to_vec = {}
    with open(filePath) as f:
        for line in f.readlines():
            words = line.split()
            vector_array.append(words[1:])
            word_to_vec[words[0]] = np.array(words[1:]).astype(float)
    return word_to_vec


word_to_vec = loadGloveVectors('/Users/csriniv6/Downloads/glove.6B/glove.6B.100d.txt')

index_to_word = list(word_to_vec.keys())

for w in index_to_word[0:5]:
    print(w, word_to_vec[w])


def find_nearest_neighbors(word_to_vec, word, n_neighbors):
    scores = cosine_similarity(word, word_to_vec)
    scores = scores.flatten()
    neighbors_index = (-scores).argsort()[0:n_neighbors]
    neighbors = [index_to_word[index] for index in neighbors_index]

    for neighbor in neighbors:
        print(neighbor)


find_nearest_neighbors(vector_array, [word_to_vec["doctor"]], 5)

print('Analogy for woman with King and man')

# King - man = ? - woman
# ? = King + woman - man

king_vector = np.array(word_to_vec["king"])
print('king', king_vector)

aunt_vector = np.array(word_to_vec["aunt"])
print('aunt', king_vector)

wife_vector = np.array(word_to_vec["wife"])
print('wife', wife_vector)

man_vector = np.array(word_to_vec["man"])
print('man', man_vector)

woman_vector = np.array(word_to_vec["woman"])
print('woman', woman_vector)

actor_vector = np.array(word_to_vec["actor"])
print('actor', actor_vector)

# man-woman = actor - actress

# man-woman = husband - wife

# man-woman = uncle - aunt

# man-woman = king - queen

#find_analogies('japan', 'japanese', 'italian')
#japan - japanese = italy - italian

japan_vector = np.array(word_to_vec["japan"])
japanese_vector = np.array(word_to_vec["japanese"])
italian_vector = np.array(word_to_vec["italian"])

print('japan', japan_vector)
print('japanese', japanese_vector)
print('italian', italian_vector)


result_vector = japan_vector + italian_vector - japanese_vector

print('result', result_vector)

find_nearest_neighbors(vector_array,
                       [result_vector],
                       1)
