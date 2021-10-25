# Import the NumPy library for matrix math
import numpy

# A single perceptron function
def perceptron(inputs_list, weights_list, bias):

	# Convert the inputs list into a numpy array
	inputs = numpy.array(inputs_list)

	# Convert the weights list into a numpy array
	weights = numpy.array(weights_list)

	# Calculate the dot product
	summed = numpy.dot(inputs, weights)

	# Add in the bias
	summed = summed + bias

	# Calculate output
	# N.B this is a ternary operator, neat huh?
	output = 1 if summed > 0 else 0
	return output

# Our main code starts here

# Test the perceptron
inputs = ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])
outputs = []

def Perceptron_function(inputs, weights, bias):
    print("Inputs: ", inputs)
    print("Weights: ",weights)
    print("Bias: ", bias)

    #store the outputs for each input
    outputs = []
    for x in range(4):
	    print("Result: ", perceptron(inputs[x], weights, bias))
	    outputs.append(perceptron(inputs[x], weights, bias))
    print(outputs)
    return outputs

gate = input("Enter desired gate: ")
if(gate == "AND"):
    weights = [1.0, 1.0]
    bias = -1.5
    outputs = Perceptron_function(inputs, weights,bias)
if(gate == "NAND"):
    weights = [-1.0, -1.0]
    bias = 2
    outputs = Perceptron_function(inputs, weights,bias)
if(gate == "OR"):
    weights = [2.0, 2.0]
    bias = -1.0
    outputs = Perceptron_function(inputs, weights,bias)
if(gate == "XOR"):
    weights = [2.0, 2.0]
    bias = -1.0
    OR_outputs = Perceptron_function(inputs, weights,bias)
    weights = [-1.0, -1.0]
    bias = 2
    NAND_outputs = Perceptron_function(inputs, weights,bias)
    inputs = ([NAND_outputs[0],OR_outputs[0]], [NAND_outputs[1],OR_outputs[1]], [NAND_outputs[2],OR_outputs[2]], [NAND_outputs[3],OR_outputs[3]])
    print(inputs)
    weights = [1.0, 1.0]
    bias = -1.0
    outputs = Perceptron_function(inputs, weights, bias)
    print("XOR")

#print("Inputs: ", inputs)
#print("Weights: ",weights)
#print("Bias: ", bias)

#store the outputs for each input
#outputs = []
#for x in range(4):
#	print("Result: ", perceptron(inputs[x], weights, bias))
#	outputs.append(perceptron(inputs[x], weights, bias))

#print(outputs)

# Import the matplotlib pyplot library
# It has a very long name, so import it as the name plt
import matplotlib.pyplot as plt

# Make a new plot (XKCD style)
fig = plt.xkcd()

#convert the output to the correct colour for graph
colour=["","","",""]
for x in range(4):
	if outputs[x] == 0:
		colour[x] = "red"
	else:
		colour[x] = "green"

print(colour)

inputs = ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])
# Add points as scatters - scatter(x, y, size, color)
# zorder determines the drawing order, set to 3 to make the
# grid lines appear behind the scatter points
plt.scatter(inputs[0][0], inputs[0][1], s=50, color=colour[0], zorder=3)
plt.scatter(inputs[1][0], inputs[1][1], s=50, color=colour[1], zorder=3)
plt.scatter(inputs[2][0], inputs[2][1], s=50, color=colour[2], zorder=3)
plt.scatter(inputs[3][0], inputs[3][1], s=50, color=colour[3], zorder=3)

if(gate == "XOR"):
    weights = [2.0, 2.0]
    bias = -1.0
    #calculate and plot linear seperator for the perceptron
    xrange = -2
    y = []
    x = []
    while xrange <= 2:
	    y.append(((-weights[0]/weights[1])*xrange)-(bias/weights[1]))
	    x.append(xrange)
	    xrange = xrange + 1

    print(x,y)
    plt.plot(x,y,'b')

    weights = [1.0, 1.0]
    bias = -1.5
    #calculate and plot linear seperator for the perceptron
    xrange = -2
    y = []
    x = []
    while xrange <= 2:
	    y.append(((-weights[0]/weights[1])*xrange)-(bias/weights[1]))
	    x.append(xrange)
	    xrange = xrange + 1

    print(x,y)
    plt.plot(x,y,'b')
else:
    #calculate and plot linear seperator for the perceptron
    xrange = -2
    y = []
    x = []
    while xrange <= 2:
	    y.append(((-weights[0]/weights[1])*xrange)-(bias/weights[1]))
	    x.append(xrange)
	    xrange = xrange + 1

    print(x,y)
    plt.plot(x,y,'b')


# Set the axis limits
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Label the plot
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Input Vector")

# Turn on grid lines
plt.grid(True, linewidth=1, linestyle=":")

# Autosize (stops the labels getting cut off)
plt.tight_layout()

# Show the plot
plt.show()
