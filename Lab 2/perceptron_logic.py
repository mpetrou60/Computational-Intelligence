# Import the NumPy library for matrix math
import numpy as np

# A single perceptron function
def perceptron(inputs_list, weights_list, bias):
    # Convert the inputs list into a numpy array
    inputs = np.array(inputs_list)
    
    # Convert the weights list into a numpy array
    weights = np.array(weights_list)
    
    # Calculate the dot product
    summed = np.dot(inputs, weights)
    
    # Add in the bias
    summed = summed + bias
    
    # Calculate output
    # N.B this is a ternary operator, neat huh?
    output = 1 if summed > 0 else 0
    
    return output

# Our main code starts here

# Test the perceptron
inputs = [1.0, 1.0]
weights = [-2.0, -2.0]
bias = 3

print('Inputs: ', inputs)
print('Weights: ',weights)
print('Bias: ', bias)
print('Result: ', perceptron(inputs, weights, bias))

# OR                    NOR
# Truth Table:          Truth Table: 
# X\Y | 0 1             X\Y | 0 1
# -----------           -----------
#  0  | 0 1              0  | 1 0
#  1  | 1 1              1  | 0 0
# Weights = [2, 2]     Weights = [-1, -1]
# Bias = -1            Bias = 1/0.5

# AND                   NAND  
# Truth Table:          Truth Table: 
# X\Y | 0 1             X\Y | 0 1
# -----------           -----------
#  0  | 0 0              0  | 1 1
#  1  | 0 1              1  | 1 0
# Weights = [1, 1]      Weights = [-2, -2]
# Bias = -1             Bias = 3
# Weights = [-1, -1]
# Bias = 1

# XOR 
# Truth Table: 
# X\Y | 0 1
# -----------
#  0  | 0 1
#  1  | 1 0
# this is impossible to do with just one perceptron.
# you would need a three perceptrons for this two work!

