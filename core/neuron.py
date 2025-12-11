import numpy as np

class Neuron:
    """
    Represents a single neuron in the graph.
    """
    def __init__(self, neuron_id, activation_function='tanh'):
        self.id = neuron_id
        self.activation = 0.0
        self.state = 0.0  # Internal state (potential)
        self.connections = []  # List of (target_neuron_id, weight) tuples
        self.activation_function = activation_function

    def update(self, input_sum):
        """
        Updates the neuron's state and activation based on input.
        """
        self.state = input_sum
        if self.activation_function == 'tanh':
            self.activation = np.tanh(self.state)
        elif self.activation_function == 'sigmoid':
             self.activation = 1 / (1 + np.exp(-self.state))
        else:
            self.activation = self.state # Linear default

    def add_connection(self, target_id, weight):
        """
        Adds a directed connection to another neuron.
        """
        self.connections.append([target_id, weight])
