import numpy as np 

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases = [2, 3, 0.5]

biases2 = [-1, 2, -0.5]

# weights must be as deep as inputs 1D vector is long
    # 1 * 0.2
    # 2 * 0.5
    # 3 * -0.26
    # 2.5 * (See the problem?)

# transpose: swap rows and columns
    # this can help fix size issues (shape errors)

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# layer1_outputs now become the input for layer 2 

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#output = np.dot(inputs, np.array(weights).T) + biases # .dot takes a matrix OR vector as arg1, arg2 only takes a vector. Thus, reversing this shown order will create an error
    # We transpose weights down here, notice the .T for transpose

    # When biases is added, its adding to the same indexes of the output. i.e. ind 0 -> 0, 1 -> 1

print(layer2_outputs)

#matrix product

# a = [[1,2,3,4], [5,6,7,8]]

# b = [[9,10,11,12], [13,14,15,16]]

# a * b = dot([1,5], [9,10])
    #notice, we are doing the dot prod of a column by a row 