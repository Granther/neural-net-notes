import numpy as np 

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]


#E = 2.71828182846

exp_values = np.exp(layer_outputs) 
    # exp_values is a matix of the same shape as what is passed in exp

'''
exp_values = []

for output in layer_outputs:
    exp_values.append(E**output)
'''
print(exp_values)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=true) 
    # axis argument is not required, axis = 0 sums columns, 1 does rows
    # keepdims allows it to sum as separate vectors instead of the entire matrix 

'''
norm_base = sum(exp_values)

norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)
'''
print(norm_values)
print(sum(norm_values))

# Pipeline
    # input value --> e ^ input_value --> normalize --> output

# Softmax is the process of exponentiation and normalizing 


# How wrong is a network?
    # Negative values
        # ReLU 'clips' negatives, making them 0
        # -9 is not the same value as +9, so why make it?
        # Data needs to represented as a positive number and is proportional to other signed data

# y = e^x
    # output = eulers_number ^ input
    # y will always be positive

# Overflows can be a problem with exponentials
    # We can combat this by subtracting every value to a vector or matrix of pre-exponentiated numbers by the largest value
    # This results in the the largest number being 0 and everything else being negative
    # But after exponentiation it doesn't really matter
        # Now our range of values can only be between 0 and 1 after expo

# normalization
    # single_neuron_output / sum of all other neuron outputs in that layer
