import numpy as np
from core.neuron import Neuron
from core.update_rules import adjust_weights

class Network:
    """
    Manages a collection of neurons and their interactions.
    """
    def __init__(self, num_neurons=10, connection_density=0.3):
        self.neurons = [Neuron(i) for i in range(num_neurons)]
        self.num_neurons = num_neurons
        self._initialize_topology(connection_density)

    def _initialize_topology(self, density):
        """
        Randomly connects neurons based on the given density.
        """
        for neuron in self.neurons:
            for target in self.neurons:
                if neuron.id != target.id and np.random.rand() < density:
                    weight = np.random.uniform(-0.5, 0.5)
                    neuron.add_connection(target.id, weight)

    def get_activations(self):
        """
        Returns the current activation vector of the network.
        """
        return np.array([n.activation for n in self.neurons])

    def set_activations(self, activations):
        """
        Forces the activations of neurons to specific values (e.g., for input clamping).
        """
        for i, val in enumerate(activations):
            if i < len(self.neurons):
                self.neurons[i].activation = val

    def update_cycle(self):
        """
        Performs one full update cycle:
        1. Compute new states based on current activations.
        2. Update activations.
        """
        current_activations = self.get_activations()
        
        # Calculate inputs for each neuron
        inputs = np.zeros(self.num_neurons)
        for neuron in self.neurons:
            for target_id, weight in neuron.connections:
                inputs[target_id] += neuron.activation * weight
        
        # Update neurons
        for i, neuron in enumerate(self.neurons):
            neuron.update(inputs[i])

    def forward(self, input_pattern, steps=5):
        """
        Runs the network for a fixed number of steps given an initial input.
        """
        self.set_activations(input_pattern)
        for _ in range(steps):
            self.update_cycle()
        return self.get_activations()
