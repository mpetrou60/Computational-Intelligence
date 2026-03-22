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
inputs = [1.0, 0.0]
weights = [-2.0, -2.0]
bias = 1

print('Inputs: ', inputs)
print('Weights: ',weights)
print('Bias: ', bias)
print('Result: ', perceptron(inputs, weights, bias))

# Question 2: This creates an AND function
# Truth Table: 
# X\Y | 0 1
# -----------
#  0  | 0 0
#  1  | 0 1