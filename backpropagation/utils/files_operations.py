import csv
import errno
import os
from pathlib import Path
from backpropagation.utils.helper_computations import num_or_str


def read_inputs_data(file_path):
    """
    Read inputs data from file. File is in csv format. Returns list of list,
    where each sublist is single input example.
    :param file_path: File that contains input data
    """
    if Path(file_path).exists():
        with open(file_path) as csv_file:
            csv_reader = csv.reader(csv_file)
            return [list(map(num_or_str, row)) for row in csv_reader]
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)


def write_weights(net, file_path):
    """
    Write weights of neural network to file except input layer.
    :param net: artificial neural network
    :param file_path: File where weights of neural network will be written
    :return: None if name of file doesn't exist
    otherwise True
    """
    if file_path != "":
        try:
            with open(file_path, 'w') as csv_file:
                csv_writer = csv.writer(csv_file)
                for layer in net[1:]:
                    for neuron in layer:
                        csv_writer.writerow(neuron.weights)
            return True
        except IOError:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    return None


def read_weights(file_path):
    """
    Read weights from file.
    :return: List of list where each sublist has weights for each single neuron
     except neurons in input layer. Return None if name of file doesn't exist
     :param file_path: File from which weights for neural network will be read
    """
    if file_path != "":
        if Path(file_path).exists():
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file)
                return [list(map(num_or_str, row)) for row in csv_reader]
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)
    return None
