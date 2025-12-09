"""
Genome representation for NEUROGEN.

A genome encodes a neural network structure with:
- Nodes (input, hidden, output)
- Connections (weighted edges between nodes)
- Mutation operations
"""

import random
import copy
from typing import List, Dict, Tuple, Optional


class Node:
    """Represents a single node in the network."""
    
    def __init__(self, node_id: int, node_type: str, activation_fn: str = 'relu'):
        """
        Args:
            node_id: Unique identifier for this node
            node_type: One of 'input', 'hidden', 'output'
            activation_fn: Activation function name ('relu', 'tanh', 'sigmoid', 'linear')
        """
        self.id = node_id
        self.type = node_type
        self.activation_fn = activation_fn
    
    def __repr__(self):
        return f"Node(id={self.id}, type={self.type}, activation={self.activation_fn})"


class Connection:
    """Represents a weighted connection between two nodes."""
    
    def __init__(self, in_id: int, out_id: int, weight: float, enabled: bool = True):
        """
        Args:
            in_id: ID of the source node
            out_id: ID of the destination node
            weight: Connection weight
            enabled: Whether this connection is active
        """
        self.in_id = in_id
        self.out_id = out_id
        self.weight = weight
        self.enabled = enabled
    
    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"Connection({self.in_id}->{self.out_id}, w={self.weight:.3f}, {status})"


class Genome:
    """
    Encodes a neural network structure with nodes and connections.
    Supports mutation operations for evolution.
    """
    
    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize a minimal genome with input and output nodes.
        
        Args:
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.nodes: List[Node] = []
        self.connections: List[Connection] = []
        self.next_node_id = 0
        
        # Create input nodes
        for i in range(num_inputs):
            self.nodes.append(Node(self.next_node_id, 'input', 'linear'))
            self.next_node_id += 1
        
        # Create output nodes
        for i in range(num_outputs):
            self.nodes.append(Node(self.next_node_id, 'output', 'tanh'))
            self.next_node_id += 1
        
        # Create initial connections from all inputs to all outputs
        for in_node in self.get_nodes_by_type('input'):
            for out_node in self.get_nodes_by_type('output'):
                weight = random.uniform(-1.0, 1.0)
                self.connections.append(Connection(in_node.id, out_node.id, weight))
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]
    
    def get_node_by_id(self, node_id: int) -> Optional[Node]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def is_acyclic(self) -> bool:
        """Check if the connection graph is acyclic (feedforward)."""
        # Build adjacency list
        adj = {node.id: [] for node in self.nodes}
        for conn in self.connections:
            if conn.enabled:
                adj[conn.in_id].append(conn.out_id)
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for neighbor in adj.get(node_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in self.nodes:
            if node.id not in visited:
                if has_cycle(node.id):
                    return False
        return True
    
    def add_node(self) -> bool:
        """
        Add a new hidden node by splitting an existing connection.
        Returns True if successful.
        """
        if not self.connections:
            return False
        
        # Pick a random enabled connection
        enabled_conns = [c for c in self.connections if c.enabled]
        if not enabled_conns:
            return False
        
        conn = random.choice(enabled_conns)
        
        # Disable the old connection
        conn.enabled = False
        
        # Create new hidden node
        activation = random.choice(['relu', 'tanh', 'sigmoid'])
        new_node = Node(self.next_node_id, 'hidden', activation)
        self.nodes.append(new_node)
        self.next_node_id += 1
        
        # Add two new connections: in->new and new->out
        # Keep weight=1.0 for in->new and original weight for new->out
        self.connections.append(Connection(conn.in_id, new_node.id, 1.0))
        self.connections.append(Connection(new_node.id, conn.out_id, conn.weight))
        
        return True
    
    def remove_node(self) -> bool:
        """
        Remove a random hidden node and its connections.
        Returns True if successful.
        """
        hidden_nodes = self.get_nodes_by_type('hidden')
        if not hidden_nodes:
            return False
        
        node = random.choice(hidden_nodes)
        
        # Remove the node
        self.nodes = [n for n in self.nodes if n.id != node.id]
        
        # Remove all connections involving this node
        self.connections = [c for c in self.connections 
                          if c.in_id != node.id and c.out_id != node.id]
        
        return True
    
    def add_connection(self) -> bool:
        """
        Add a new random connection between two nodes.
        Ensures the graph remains acyclic.
        Returns True if successful.
        """
        # Get potential source and destination nodes
        # Source: input or hidden, Destination: hidden or output
        sources = self.get_nodes_by_type('input') + self.get_nodes_by_type('hidden')
        destinations = self.get_nodes_by_type('hidden') + self.get_nodes_by_type('output')
        
        if not sources or not destinations:
            return False
        
        # Try multiple times to find a valid connection
        for _ in range(20):
            src = random.choice(sources)
            dst = random.choice(destinations)
            
            # Can't connect node to itself
            if src.id == dst.id:
                continue
            
            # Check if connection already exists
            exists = any(c.in_id == src.id and c.out_id == dst.id 
                        for c in self.connections)
            if exists:
                continue
            
            # Temporarily add connection and check for cycles
            weight = random.uniform(-1.0, 1.0)
            new_conn = Connection(src.id, dst.id, weight)
            self.connections.append(new_conn)
            
            if self.is_acyclic():
                return True
            else:
                # Remove the connection if it creates a cycle
                self.connections.pop()
        
        return False
    
    def remove_connection(self) -> bool:
        """
        Remove a random connection.
        Ensures at least one path from inputs to outputs remains.
        Returns True if successful.
        """
        if len(self.connections) <= self.num_inputs:
            # Keep at least some connections
            return False
        
        enabled_conns = [c for c in self.connections if c.enabled]
        if not enabled_conns:
            return False
        
        conn = random.choice(enabled_conns)
        conn.enabled = False
        
        return True
    
    def perturb_weight(self) -> bool:
        """
        Randomly perturb a connection weight.
        Returns True if successful.
        """
        if not self.connections:
            return False
        
        conn = random.choice(self.connections)
        
        # 90% chance: small perturbation, 10% chance: complete reset
        if random.random() < 0.9:
            conn.weight += random.gauss(0, 0.3)
            # Clamp to reasonable range
            conn.weight = max(-5.0, min(5.0, conn.weight))
        else:
            conn.weight = random.uniform(-1.0, 1.0)
        
        return True
    
    def mutate(self, mutation_rates: Dict[str, float] = None):
        """
        Apply mutations to this genome.
        
        Args:
            mutation_rates: Dictionary of mutation probabilities
                - 'add_node': probability of adding a node
                - 'remove_node': probability of removing a node
                - 'add_connection': probability of adding a connection
                - 'remove_connection': probability of removing a connection
                - 'perturb_weight': probability of perturbing a weight
        """
        if mutation_rates is None:
            mutation_rates = {
                'add_node': 0.03,
                'remove_node': 0.02,
                'add_connection': 0.05,
                'remove_connection': 0.03,
                'perturb_weight': 0.8
            }
        
        if random.random() < mutation_rates['add_node']:
            self.add_node()
        
        if random.random() < mutation_rates['remove_node']:
            self.remove_node()
        
        if random.random() < mutation_rates['add_connection']:
            self.add_connection()
        
        if random.random() < mutation_rates['remove_connection']:
            self.remove_connection()
        
        if random.random() < mutation_rates['perturb_weight']:
            self.perturb_weight()
    
    def copy(self) -> 'Genome':
        """Create a deep copy of this genome."""
        return copy.deepcopy(self)
    
    def __repr__(self):
        return (f"Genome(nodes={len(self.nodes)}, "
                f"connections={len([c for c in self.connections if c.enabled])})")
