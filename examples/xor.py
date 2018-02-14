import random
import sys

sys.path.append("../src")  # Adds source directory to python modules path.
from nn import neural_network


def main():

    # Training data containing the 4 posible outputs
    trainig = [{
        'inputs': [0, 0],
        'output': [0]
    },
        {
        'inputs': [0, 1],
        'output': [1]
    },
        {
        'inputs': [1, 0],
        'output': [1]
    }, {
        'inputs': [1, 1],
        'output': [0]
    }]

    # Neural Network with 2 input nodes, 4 nodes in the hidden layer and 1 ouput node
    nn = neural_network(2, 4, 1)
    print('Training NN with 20000 iterations')
    for x in range(20000):
        # Use random data from the training set in every iteration
        data = random.choice(trainig)
        # Training with that data
        nn.train(data['inputs'], data['output'])
        # To print the progress every 10%
        if x % 2000 is 0:
            print((x / 2000 * 10), '%')

    # List of all the posible inputs
    tests = [[0, 0], [0, 1], [1, 0], [1, 1]]
    for test in tests:
        # Get the NN prediction of each input
        prediction = nn.predict(test)[0]
        # Print the results
        print('XOR:', test, 'PREDICTION:', round(
            prediction), 'PRECISION', round(prediction, 4))


if __name__ == '__main__':
    main()
