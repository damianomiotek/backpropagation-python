settings = {
  "data inputs file": "",                       # absolute path to file that contains inputs data
  "weights read file": "",                      # absolute path to file from which weights may be reading
  "weights write file": "",                     # absolute path to file in which weights will be written
  "errors while training": "",                  # absolute path to file where errors while training will be written
  "validation set ratio": 0.1,                  # ratio of the validation set to the entire data set
  "test set ratio": 0.1,                        # ratio of the test set to the entire data set
  "hidden layer sizes": [3, 2],                 # list of neurons numbers in each hidden layer
  "learning rate": 0.01,                        # Learning rate of gradient descent
  "stop condition": "epochs",                   # stop condition for learning neural network: epochs or cohesion
  "cohesion": 0.01,                             # if above selected are epochs, value for "cohesion" key might be 0
  "epochs number": 300                          # if selected is cohesion, value "epochs number" might be 0
}