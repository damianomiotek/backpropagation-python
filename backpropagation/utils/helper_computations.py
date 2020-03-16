import operator
import random
import numpy as np


def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


def random_weights(min_value, max_value, num_weights):
    """Return list with num_weights randomly weights"""
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


def dot_product(x, y):
    """Return the sum of the element-wise product of vectors x and y."""
    return sum(_x * _y for _x, _y in zip(x, y))


def sigmoid(x):
    """Return activation value of x with sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(value):
    """Return derivative of neuron activation value"""
    return value * (1 - value)


def init_examples(examples, idx_i, idx_t, o_units):
    """Return two dictionaries, both contains integers as keys.
    "inputs" dictionary contains input values for neural network, "targets" contains targets values
    of neural network"""
    inputs, targets = {}, {}

    for i, e in enumerate(examples):
        # input values of e
        inputs[i] = [e[i] for i in idx_i]

        if o_units > 1:
            # one-hot representation of e's target
            t = [0 for i in range(o_units)]
            t[e[idx_t]] = 1
            targets[i] = t
        else:
            # target value of e
            targets[i] = [e[idx_t]]

    return inputs, targets


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return list(map(operator.add, a, b))


def scalar_vector_product(x, y):
    """Return vector as a product of a scalar and a vector"""
    return np.multiply(x, y)
