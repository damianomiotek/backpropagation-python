#!/usr/bin/python3

import random
import matplotlib.pyplot as plt
import errno
import os
from backpropagation.config.settings import settings
from backpropagation.utils import files_operations as fo
from backpropagation.utils.dataset_operations import DataSet
from backpropagation.utils.helper_computations import sigmoid, sigmoid_derivative, init_examples, dot_product, \
    vector_add, scalar_vector_product
from backpropagation.neural_network import network, initialize_net_weights, err_ratio, prediction, grade_learner


def neural_net_learner(examples, backprop_stop_cond="epochs", hidden_layer_sizes=None, learning_rate=0.01,
                       validation_set_size=10, weights_from_file=None, cohesion=None, epochs_number=300,
                       file_to_write_errors=None):
    """
    Layered feed-forward network. Build and teach a raw artificial neural network using backpropagation algorithm.
    :param examples: List with examples to learning net
    :param backprop_stop_cond: String with stop condition for learning network: epochs or cohesion
    :param hidden_layer_sizes: List of number of hidden units per hidden layer
    :param learning_rate: Learning rate of gradient descent
    :param validation_set_size: Size of validation data set
    :param weights_from_file: List of lists where each sublist contains weights for single neuron,
    except neurons in input layer
    :param cohesion: Number of cohesion rate as stop condition for network learning, if backprop_stop_cond
    equals "cohesion"
    :param epochs_number: Number of passes over the dataset, if backprop_stop_cond equals "epochs"
    :param file_to_write_errors: Absolute path to file where errors will be written
    :return net: learned net
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = [3]
    i_units = target_index = len(examples[0]) - 1
    o_units = len(set([example[target_index] for example in examples]))

    # construct a network
    raw_net = network(i_units, hidden_layer_sizes, o_units)
    # learn network
    return backpropagation(examples, raw_net, backprop_stop_cond, learning_rate, validation_set_size,
                           weights_from_file, cohesion, epochs_number, file_to_write_errors)


def backpropagation(examples, net, stop_cond="epochs", learning_rate=0.01, validation_set_size=10, weights_from_file=None,
                    cohesion=None, epochs_number=300, file_to_write_errors=None):
    """
    The back-propagation algorithm for multilayer networks.
    :param examples: List with examples to learning net
    :param net: raw artificial neural network
    :param stop_cond: String with stop condition for learning network: epochs or cohesion
    :param learning_rate: Learning rate of gradient descent
    :param validation_set_size: Size of validation data set
    :param weights_from_file: List of lists where each sublist contains weights for single neuron,
    except neurons in inputs layer
    :param cohesion: Number of cohesion rate as stop condition for network learning, if backprop_stop_cond
    equals "cohesion:
    :param epochs_number: Number of passes over the dataset, if backprop_stop_cond equals "epochs"
    :param file_to_write_errors: Absolute path to file where errors will be written
    :return net: learned net
    """
    # initialise weights
    initialize_net_weights(net, weights_from_file)

    # initialize some variables for further computations
    o_neurons = net[-1]
    i_neurons = net[0]
    o_units = len(o_neurons)
    idx_t = len(examples[0]) - 1
    idx_i = [i for i in range(idx_t)]
    n_layers = len(net)
    epoch = 0

    file_with_results = None
    if file_to_write_errors is not None:
        try:
            file_with_results = open(file_to_write_errors, 'w')
        except IOError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_to_write_errors)

    count_err_ratio = err_ratio(net)
    validation_examples = examples[:validation_set_size]
    errors = list()
    errors.append(count_err_ratio(validation_examples))

    def stop_after_epochs():
        if epoch > epochs_number:
            return False
        return True

    def stop_after_cohesion():
        if errors[-1] < cohesion:
            return False
        return True

    if stop_cond == "cohesion":
        learning = stop_after_cohesion
    else:
        learning = stop_after_epochs

    while learning():
        random.shuffle(examples)
        validation_examples = examples[:validation_set_size]
        training_examples = examples[validation_set_size:]
        # iterate over each example
        inputs, targets = init_examples(training_examples, idx_i, idx_t, o_units)
        for e in range(len(training_examples)):
            i_val = inputs[e]
            t_val = targets[e]

            # activate input layer
            for v, n in zip(i_val, i_neurons):
                n.value = v

            # forward pass
            for layer in range(1, n_layers):
                inc = [n.value for n in net[layer - 1]]
                for neuron in net[layer]:
                    in_val = dot_product(inc, neuron.weights)
                    neuron.value = sigmoid(in_val)

            # initialize delta
            delta = [[] for _ in range(n_layers)]

            # compute outer layer delta

            # error for the MSE cost function
            err = [t_val[i] - o_neurons[i].value for i in range(o_units)]

            # calculate delta at output
            delta[-1] = [sigmoid_derivative(o_neurons[i].value) * err[i] for i in range(o_units)]

            # backward pass
            h_layers = n_layers - 2
            for i in range(h_layers, 0, -1):
                layer = net[i]
                h_units = len(layer)
                nx_layer = net[i + 1]

                # weights from each ith layer neuron to each i + 1th layer neuron
                w = [[neuron.weights[k] for neuron in nx_layer] for k in range(h_units)]

                delta[i] = [sigmoid_derivative(layer[j].value) * dot_product(w[j], delta[i + 1])
                            for j in range(h_units)]

            # update weights
            for i in range(1, n_layers):
                layer = net[i]
                inc = [neuron.value for neuron in net[i - 1]]
                units = len(layer)
                for j in range(units):
                    layer[j].weights = vector_add(layer[j].weights,
                                                  scalar_vector_product(learning_rate * delta[i][j], inc))
        errors.append(count_err_ratio(validation_examples))
        if file_with_results is not None:
            file_with_results.write(f'Error after {epoch} epoch is {errors[-1]}\n')
        epoch += 1

    # First error in list was compute before learning(before while loop)
    errors.pop(0)

    # draw learning plot using Matplotlib
    draw_learning_plot(errors, epoch)

    if file_with_results is not None:
        file_with_results.close()
    return net


def draw_learning_plot(errors, n_epoch):
    """Draw plot of progress of learning neural network
    :param errors: list with errors values. One error for one epoch.
    :param n_epoch: Epoch number
    return: True if function has drawn plot, otherwise False
    """
    if n_epoch > 0:
        epochs = [i + 1 for i in range(n_epoch)]
        plt.plot(epochs, errors, lw=2, color='blue')
        plt.show()
        return True
    else:
        print("The chart wasn't drawn because the neural network has been already trained")
        return None


if __name__ == '__main__':
    data_set = DataSet(fo.read_inputs_data(settings['data inputs file']), settings['validation set ratio'],
                       settings['test set ratio'])

    # Using neural_net_learner function as wrapper for backpropagation function.
    # neural_net_learner construct raw neural network for us and calls backpropagation algorithm
    learned_net = neural_net_learner(data_set.learning_set, settings['stop condition'], settings['hidden layer sizes'],
                                     settings['learning rate'], data_set.validation_set_size,
                                     fo.read_weights(settings['weights read file']), settings['cohesion'],
                                     settings['epochs number'], settings['errors while training'])

    predict = prediction(learned_net)
    print(f'After learning, neural network has {grade_learner(predict, data_set.test_set)} efficiency')
    fo.write_weights(learned_net, settings['weights write file'])
