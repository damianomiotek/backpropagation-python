import random


class DataSet:
    """
    Stores data set and operates on it
    """
    def __init__(self, examples, validation_set_ratio=0.09, test_set_ratio=0.09):
        self.examples = examples
        self.learning_set = None                    # learning set = training set + validation set
        self.test_set = None
        self.target = len(self.examples[0]) - 1
        self.validation_set_size = int(validation_set_ratio * len(examples))
        self.test_set_size = int(test_set_ratio * len(examples))

        # Prepares examples for the back propagation algorithm by changing targets names to integers.
        # Divides examples into appropriate sets
        self.classes_to_numbers()
        self.divide_examples()

    def classes_to_numbers(self):
        """Converts class/targets names to numbers."""
        classes = sorted({item[self.target] for item in self.examples})
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def divide_examples(self):
        """
        Divides all examples into training and learning sets
        """
        random.shuffle(self.examples)

        self.test_set = self.examples[:self.test_set_size]
        self.learning_set = self.examples[self.test_set_size:]

        self.test_set = [(example[:self.target], example[self.target]) for example in self.test_set]
