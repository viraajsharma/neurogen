"""
Dynamic neural network construction from genome.

Builds a PyTorch network from a genome representation.
Supports forward pass and local learning updates.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from core.genome import Genome, Node, Connection


class DynamicNetwork(nn.Module):
    """
    A PyTorch network dynamically constructed from a genome.
    Supports local learning (weight updates during evaluation).
    """
    
    def __init__(self, genome: Genome):
        """
        Build a network from a genome.
        
        Args:
            genome: Genome encoding the network structure
        """
        super().__init__()
        self.genome = genome
        
        # Store activation functions
        self.activations = {
            'linear': lambda x: x,
            'relu': torch.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        
        # Build the network structure
        self._build_network()
    
    def _build_network(self):
        """Construct the network from genome."""
        # Create a mapping from node IDs to their indices
        self.node_id_to_idx = {node.id: i for i, node in enumerate(self.genome.nodes)}
        
        # Store node activation functions
        self.node_activations = {}
        for node in self.genome.nodes:
            self.node_activations[node.id] = self.activations.get(
                node.activation_fn, self.activations['relu']
            )
        
        # Build connection matrix (sparse representation)
        # Store as list of (in_id, out_id, weight_param)
        self.connection_params = nn.ParameterList()
        self.connection_map = []  # List of (in_id, out_id, param_idx)
        
        for i, conn in enumerate(self.genome.connections):
            if conn.enabled:
                # Create a learnable parameter for this connection
                weight_param = nn.Parameter(torch.tensor([conn.weight], dtype=torch.float32))
                self.connection_params.append(weight_param)
                self.connection_map.append((conn.in_id, conn.out_id, len(self.connection_params) - 1))
        
        # Compute topological order for feedforward evaluation
        self._compute_evaluation_order()
    
    def _compute_evaluation_order(self):
        """
        Compute the order in which nodes should be evaluated.
        Uses topological sort for feedforward networks.
        """
        # Build adjacency list
        adj = {node.id: [] for node in self.genome.nodes}
        in_degree = {node.id: 0 for node in self.genome.nodes}
        
        for in_id, out_id, _ in self.connection_map:
            adj[in_id].append(out_id)
            in_degree[out_id] += 1
        
        # Topological sort using Kahn's algorithm
        queue = []
        for node in self.genome.nodes:
            if in_degree[node.id] == 0:
                queue.append(node.id)
        
        self.eval_order = []
        while queue:
            node_id = queue.pop(0)
            self.eval_order.append(node_id)
            
            for neighbor in adj[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Separate input, hidden, and output nodes
        self.input_ids = [n.id for n in self.genome.get_nodes_by_type('input')]
        self.output_ids = [n.id for n in self.genome.get_nodes_by_type('output')]
        self.hidden_ids = [n.id for n in self.genome.get_nodes_by_type('hidden')]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, num_inputs)
        
        Returns:
            Output tensor of shape (batch_size, num_outputs)
        """
        batch_size = x.shape[0]
        
        # Initialize node activations
        node_values = {node.id: torch.zeros(batch_size, dtype=torch.float32) 
                      for node in self.genome.nodes}
        
        # Set input values
        for i, node_id in enumerate(self.input_ids):
            if i < x.shape[1]:
                node_values[node_id] = x[:, i]
        
        # Store pre-activation values for local learning
        self.pre_activations = {}
        self.post_activations = {}
        
        # Evaluate nodes in topological order
        for node_id in self.eval_order:
            if node_id in self.input_ids:
                # Input nodes already set
                self.post_activations[node_id] = node_values[node_id]
                continue
            
            # Compute weighted sum of inputs
            weighted_sum = torch.zeros(batch_size, dtype=torch.float32)
            
            for in_id, out_id, param_idx in self.connection_map:
                if out_id == node_id:
                    weight = self.connection_params[param_idx]
                    weighted_sum += node_values[in_id] * weight
            
            # Store pre-activation
            self.pre_activations[node_id] = weighted_sum
            
            # Apply activation function
            activation_fn = self.node_activations[node_id]
            node_values[node_id] = activation_fn(weighted_sum)
            
            # Store post-activation
            self.post_activations[node_id] = node_values[node_id]
        
        # Collect output values
        outputs = []
        for node_id in self.output_ids:
            outputs.append(node_values[node_id].unsqueeze(1))
        
        return torch.cat(outputs, dim=1) if outputs else torch.zeros(batch_size, 1)
    
    def apply_local_learning(self, learning_rate: float = 0.01):
        """
        Apply local learning rules to update weights.
        Uses Hebbian learning: Δw = η * pre_activation * post_activation
        
        Args:
            learning_rate: Learning rate for weight updates
        """
        if not hasattr(self, 'pre_activations') or not hasattr(self, 'post_activations'):
            return  # No forward pass has been done yet
        
        with torch.no_grad():
            for in_id, out_id, param_idx in self.connection_map:
                # Get pre-synaptic activation (input to connection)
                if in_id in self.post_activations:
                    pre_act = self.post_activations[in_id].mean()  # Average over batch
                else:
                    continue
                
                # Get post-synaptic activation (output of connection)
                if out_id in self.post_activations:
                    post_act = self.post_activations[out_id].mean()  # Average over batch
                else:
                    continue
                
                # Hebbian update: Δw = η * pre * post
                delta_w = learning_rate * pre_act * post_act
                
                # Update weight
                self.connection_params[param_idx].add_(delta_w)
                
                # Clamp weights to prevent explosion
                self.connection_params[param_idx].clamp_(-5.0, 5.0)
    
    def get_weights(self) -> List[float]:
        """Get all connection weights as a list."""
        return [param.item() for param in self.connection_params]
    
    def set_weights(self, weights: List[float]):
        """Set connection weights from a list."""
        with torch.no_grad():
            for i, weight in enumerate(weights):
                if i < len(self.connection_params):
                    self.connection_params[i].fill_(weight)
    
    def __repr__(self):
        return (f"DynamicNetwork(nodes={len(self.genome.nodes)}, "
                f"connections={len(self.connection_map)})")
