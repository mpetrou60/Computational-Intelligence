# import scipy.special for the sigmoid function expit()
import scipy.special, numpy
import matplotlib.pyplot as plt

# Neural network class definition
class NeuralNetwork:
    # Init the network, this gets run whenever we make a new instance of this class
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set the number of nodes in each input, hidden and output layer
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes

        # Weight matrices, with (input -> hidden) and who (hidden -> output)
        self.wih = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        self.who = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))

        # Set the learning rate
        self.lr = learning_rate

        # Set the activation function, the logistic sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    # Train the network using back-propagation of errors
    def train(self, inputs_list, targets_list):
        # Convert inputs into 2D arrays
        inputs_array = numpy.array(inputs_list, ndmin=2).T
        targets_array = numpy.array(targets_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array)

        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Current error is (target - actual)
        output_errors = targets_array - final_outputs

        # Hidden layer errors are the output errors, split by the weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # Update the weights for the links between the input and hidden layers
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0 - hidden_outputs)), numpy.transpose(inputs_array))

    # Query the network
    def query(self, inputs_list):
        # Converts the inputs list into a 2D array
        inputs_array = numpy.array(inputs_list, ndmin=2).T

        # Calcualte the signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs_array) 

        # Calculate output from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # Calculate outputs from the final layer
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


# Exercise 1:
nn1 = NeuralNetwork(2, 2, 1, 0.1)
nn1.query([1.0, 0.0])
# outputs are completely random as the weights are given 
# random values and have only been run once so there has 
# no influence from the training to converge onto a 
# specific solution

# Exercise 2/3:
nn2 = NeuralNetwork(2, 2, 1, 0.1)
inputs = [[0,0], [0,1], [1,0], [1,1]]
targets = [[0], [0], [0], [1]]
nn2.train(inputs, targets)
nn2.query(inputs)
# needs to train more than once for it to work

# Exercise 4/5/6:
nn3 = NeuralNetwork(2, 5, 1, 0.25)
# inputs = [[0,0], [0,1], [1,0], [1,1]]
# targets = [[0], [0], [0], [1]]
for j in range(1000):
        nn3.train(inputs, targets)
        nn3.query(inputs)
nn3.query(inputs)
# Reducing the learning rate reduces the number of iterations required
# Increasing the hidden layers increases the number of iterations required
# for AND to work with the same number of iterations and learning rate I 
# had to increase the number of hidden layers.
# For XOR to work need to increase iterations to 100000 and increase hidden
# layers to 10. This is because XOR is a more complex gate to model so will 
# take longer to learn.
# Larger networks will be required for more inputs as this will create a more
# complex relationship

print(nn3.wih)
print(nn3.who)
outputs = [[0], [1], [1], [0]]

# # Create the figure
# fig = plt.figure(num=1, clear=True)
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# # set the axis limits
# plt.xlim(-2, 2)
# plt.ylim(-2, 2)
# x = numpy.linspace(-2,2,50)
# linear_separator1 = []
# linear_separator2 = []
# linear_separator3 = []
# linear_separator4 = []
# linear_separator5 = []
# for m in range(len(x)):
#     linear_separator1.append((-nn3.wih[0][0]/nn3.wih[0][1])*x[m] - (1/-nn3.wih[0][1]))
#     linear_separator2.append((-nn3.wih[1][0]/nn3.wih[1][1])*x[m] - (1/-nn3.wih[1][1]))
#     linear_separator3.append((-nn3.wih[2][0]/nn3.wih[2][1])*x[m] - (1/-nn3.wih[2][1]))
#     linear_separator4.append((-nn3.wih[3][0]/nn3.wih[3][1])*x[m] - (1/-nn3.wih[3][1]))
#     linear_separator5.append((-nn3.wih[4][0]/nn3.wih[4][1])*x[m] - (1/-nn3.wih[4][1]))

# ax.plot_surface(numpy.asarray(x), numpy.asarray(linear_separator1), numpy.asarray(linear_separator2))
# ax.hold(True)
# ax.plot_surface(numpy.asarray(x), numpy.asarray(linear_separator2), numpy.asarray(linear_separator3), alpha=0.2)
# ax.plot_surface(numpy.asarray(x), numpy.asarray(linear_separator3), numpy.asarray(linear_separator4), alpha=0.2)
# ax.plot_surface(numpy.asarray(x), numpy.asarray(linear_separator4), numpy.asarray(linear_separator5), alpha=0.2)
# ax.plot_surface(numpy.asarray(x), numpy.asarray(linear_separator5), numpy.asarray(linear_separator1), alpha=0.2)

# # plt.plot(x, linear_separator1)
# # plt.plot(x, linear_separator2)
# # plt.plot(x, linear_separator3)
# # plt.plot(x, linear_separator4)
# # plt.plot(x, linear_separator5)

# i = 0
# for out in outputs:
#     if out == 1:
#         ax.scatter(inputs[i][0],inputs[i][1],0, s=50, color='aquamarine', zorder=3)
#     else:
#         ax.scatter(inputs[i][0],inputs[i][1],0, s=50, color='lightsalmon', zorder=3)
#     i = i+1



# # Label the plot
# plt.xlabel("Input 1", fontsize=20)
# plt.ylabel("Input 2", fontsize=20)
# plt.title("State Space of Input Vector for XOR", fontsize=20)

# # Turn on grid lines
# plt.grid(True, linewidth=1, linestyle=':')

# # Autosize (stops the labels getting cut off)
# plt.tight_layout()

# # Show the plot
# plt.show()