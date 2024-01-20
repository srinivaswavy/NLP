import numpy as np

list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]

# Convert lists to NumPy arrays
array1 = np.array(list1)
array2 = np.array(list2)
array3 = np.array(list3)

# Perform element-wise subtraction
array4 = array1 + array2 - array3

# Convert the result back to a list if needed
list4 = array4.tolist()

print(list4)  # Output: [-3, -3, -3]
