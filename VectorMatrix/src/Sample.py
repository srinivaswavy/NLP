# import numpy as np
#
# # Let's assume the following is your numpy array
# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# # You can divide each element of a row by the sum of elements in that row as follows:
# result = a / a.sum(axis=1,keepdims=True)
#
# print(result)
#
# print(a.sum(axis=1,keepdims=True))
# print(a.sum(axis=1,keepdims=False))
#
# result = a / a.sum(axis=1,keepdims=False)
#
# print(result)

# import itertools
#
# number = ["A"] + [str(r) for r in range(2, 11)] + ["J", "Q", "K"]
# suit = ["Clubs", "Diamonds", "Hearts", "Spades"]
#
# cards_tuple = list(itertools.product(number, suit))
#
# print(cards_tuple)
#
# cards = [n + " of " + s for n in number for s in suit]
#
# print(cards)


class Calculator:
    def __init__(self):
        pass

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        else:
            return a / b
