from statistics import mean
from backpropagation.utils.helper_computations import random_weights, dot_product, sigmoid


def initialize_net_weights(net, weights_from_file=None):
    """
    Initialize net using weights from file or randomly weights
    :param net: artificial neural network
    :param weights_from_file: List of list where each sublist has weights for single neuron
    """
    if weights_from_file:
        i = 0
        for layer in net[1:]:
            for neuron in layer:
                neuron.weights = weights_from_file[i]
                i += 1
    else:
        for layer in net:
            for neuron in layer:
                neuron.weights = random_weights(min_value=-0.5, max_value=0.5, num_weights=len(neuron.weights))


class Neuron:
    """
    Neuron - Single Unit of Multiple Layer Neural Network
    """
    def __init__(self, weights=None):
        self.weights = weights or []
        self.value = None


def network(input_units, hidden_layer_sizes, output_units):
    """
    Create Directed Acyclic Network of given number layers.
    :param input_units: number of neurons in the input network layer
    :param hidden_layer_sizes: List number of neuron units in each hidden layer
    :param output_units: number of neurons in the output net layer
    excluding input and output layers
    """
    layers_sizes = [input_units] + hidden_layer_sizes + [output_units]

    net = [[Neuron() for _ in range(size)] for size in layers_sizes]
    n_layers = len(net)

    # Give each neuron the right amount of weights, where each weight is 0
    for layer in range(1, n_layers):
        prev_layer = [0 for _ in net[layer - 1]]
        for n in net[layer]:
            n.weights = prev_layer

    return net


def prediction(net):
    def predict(example):
        """Pass the example through the neural network and return the result"""

        # input neurons
        i_neurons = net[0]

        # activate input layer
        for v, n in zip(example, i_neurons):
            n.value = v

        n_layers = len(net)
        # forward pass
        for layer in range(1, n_layers):
            for neuron in net[layer]:
                inc = [n.value for n in net[layer-1]]
                in_val = dot_product(inc, neuron.weights)
                neuron.value = sigmoid(in_val)

        # hypothesis
        o_neurons = net[-1]
        return o_neurons.index(max(o_neurons, key=lambda neuron: neuron.value))

    return predict


def err_ratio(net):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    """
    predict = prediction(net)

    def count_err_ratio(examples):
        if len(examples) == 0:
            return 0.0
        target = len(examples[0]) - 1
        right = 0
        for example in examples:
            desired = example[target]
            output = predict(example)
            if output == desired:
                right += 1
        return 1 - (right / len(examples))

    return count_err_ratio


def grade_learner(predict, tests):
    """
    Grades the given learner based on how many tests it passes.
    tests is a list with each element in the form: (values, output).
    """
    return mean(int(predict(X) == y) for X, y in tests)
