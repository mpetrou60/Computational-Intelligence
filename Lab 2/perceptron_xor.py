# Import the NumPy library for matrix math
import numpy as np

# Import the matplotlib pyplot library
# it has a very long name, so import it as the name plt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

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
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
weights1 = [1.0, 1.0]
weights2 = [-1.0, 1.0]
bias1 = -1.5
bias2 = -0.5
bias3  = 0

print('Inputs: ', inputs)
print('Weights: ',weights1)
print('Bias: ', bias1)
outputs = []
for input in inputs:
    p1 = perceptron(input, weights1, bias1)
    p2 = perceptron(input, weights1, bias2)
    print('Result: ', perceptron([p1, p2], weights2, bias3))
    outputs.append(perceptron([p1, p2], weights2, bias3))

print(outputs)

# Make a new plot (XKCD style)
# fig = plt.xkcd()

# Add points as scatters - scatter (x, y, size, colour)
# zorder determines the drawing order, set to 3 to make the
# grid lines appear behind the scatter points
i = 0
for out in outputs:
    if out == 1:
        plt.scatter(inputs[i][0],inputs[i][1], s=50, color='aquamarine', zorder=3)
    else:
        plt.scatter(inputs[i][0],inputs[i][1], s=50, color='lightsalmon', zorder=3)
    i = i+1

# set the axis limits
plt.xlim(-2, 2)
plt.ylim(-2, 2)
x = np.linspace(-2,2,50)
linear_separator1 = []
linear_separator2 = []
for m in range(len(x)):
    linear_separator1.append((-weights1[0]/weights1[1])*x[m] - (bias1/weights1[1]))
    linear_separator2.append((-weights1[0]/weights1[1])*x[m] - (bias2/weights1[1]))

# plt.plot(x, linear_separator1)
# plt.plot(x, linear_separator2)

sns.lineplot(x=x, y=linear_separator1, color="lavender", linewidth=2.5)
sns.lineplot(x=x, y=linear_separator2, color="lavender", linewidth=2.5)

# Label the plot
plt.xlabel("Input 1", fontsize=20)
plt.ylabel("Input 2", fontsize=20)
plt.title("State Space of Input Vector for XOR", fontsize=20)

# Turn on grid lines
plt.grid(True, linewidth=1, linestyle=':')

# Autosize (stops the labels getting cut off)
plt.tight_layout()

# Show the plot
plt.show()