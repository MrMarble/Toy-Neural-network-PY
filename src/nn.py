import numpy as np

from matrix import Matrix


class activation_function():
    def __init__(self, x_func, y_func):
        self.x = x_func
        self.y = y_func

    @staticmethod
    def sigmoid():
        return activation_function(
            lambda x, i, j: 1 / (1 + np.exp(-x)),
            lambda y, i, j: y * (1-y)
        )

    @staticmethod
    def tanh():
        return activation_function(
            lambda x, i, j: np.tanh(x),
            lambda y, i, j: 1-(y*y)
        )


class neural_network():
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()
        self.set_learning_rate()

        self.set_activation_function()

    def predict(self, input_list):
        # Generating the Hidden Outputs
        inputs = Matrix.from_list(input_list)
        hidden = Matrix.static_multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # Activation function!
        hidden.map(self.activation_function.x)

        # Generating the output's output!
        output = Matrix.static_multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(self.activation_function.x)

        # Sending back to the caller!
        return output.to_list()

    def set_learning_rate(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate

    def set_activation_function(self, func=activation_function.sigmoid()):
        self.activation_function = func

    def train(self, input_list, target_list):
        # Generating the Hidden Outputs
        inputs = Matrix.from_list(input_list)
        hidden = Matrix.static_multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        # activation function!
        hidden.map(self.activation_function.x)

        # Generating the output's output!
        outputs = Matrix.static_multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(self.activation_function.x)

        # Convert array to matrix object
        targets = Matrix.from_list(target_list)

        # Calculate the error
        # ERROR = TARGETS - OUTPUTS
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate gradient
        gradients = Matrix.static_map(outputs, self.activation_function.y)
        gradients.multiply(output_errors)
        gradients.multiply(self.learning_rate)

        # Calculate deltas
        hidden_T = Matrix.transpose(hidden)
        weight_ho_deltas = Matrix.static_multiply(gradients, hidden_T)

        # Adjust the weights by deltas
        self.weights_ho.add(weight_ho_deltas)
        # Adjust the bias by its deltas(which is just the gradients)
        self.bias_o.add(gradients)

        # Calculate the hidden layer errors
        who_t = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.static_multiply(who_t, output_errors)

        # Calculate hidden gradient
        hidden_gradient = Matrix.static_map(hidden, self.activation_function.y)
        hidden_gradient.multiply(hidden_errors)
        hidden_gradient.multiply(self.learning_rate)

        # Calcuate input->hidden deltas
        inputs_T = Matrix.transpose(inputs)
        weight_ih_deltas = Matrix.static_multiply(hidden_gradient, inputs_T)

        self.weights_ih.add(weight_ih_deltas)
        # Adjust the bias by its deltas(which is just the gradients)
        self.bias_h.add(hidden_gradient)
