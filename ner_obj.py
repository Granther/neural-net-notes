import numpy as np 
import cd

np.random.seed(0)

# note: In terms of the weights, we try to keep them between -1 and 1 

X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

#X, y = cd.create_data(100, 3)

'''

# rectified linear...

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []

for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)
'''

class Layer_Dense:
    def __init__ (self, n_inputs, n_neurons): # n_inputs os how many features are in each 1D vector, the tail number of the shape value (3, 4)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) # this takes the x y of the shape
            # generating a matrix of shape (n_inputs, n_neurons) populated with random numbers multiplied by 0.10 to sort of normalize between -1 and 1
                # note that the generated matrix is already Transposed compared to the weights in ner2.py
        self.biases = np.zeros((1, n_neurons)) # this takes a tuple of the shape
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs) # different way to do ReLU

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

'''
layer1 = Layer_Dense(2, 5) 
    # when making layer2's inputs dependent on layer1, layer2 must take layer1's shape as its input, arg2 of layer1 = arg1 of layer2
#layer2 = Layer_Dense(5, 2)

activation1 = Activation_ReLU()

layer1.forward(X)

activation1.forward(layer1.output)

#layer2.forward(layer1.output)

print(activation1.output)

''' 

X, y = cd.create_data(points=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])


# All activation functions should change or filter data passing from neuron to neuron
    # Linear functions do NOT do this, in a linear function y = x. Input = Output, so data is passed, unchanged

# step function
    # function that runs after weight*input + bias for each neuron. 
        # value > 0 = 1
        # value <= 0 = 0

# sigmoid function is similar, but instead spits out a value between 0 or 1, essentually a bit more granularity. 1 = 100% 0 = 0% 0.7 = 70%

# Both of these functions are the new input of the next layer

# Rectified Linear Activation Function
    # if value > 0,  value = value
    # if value <= 0, value = 0
        # cant be less than 0, can be granular if over 0
        # simpler calculation than sigmoid
        # The activation point can be offset by tweaking the bias 

# study lscpu

