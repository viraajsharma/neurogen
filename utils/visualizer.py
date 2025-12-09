"""
Visualization utilities for NEUROGEN.

Visualize network structures and training progress.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Optional
from core.genome import Genome


def visualize_genome(genome: Genome, save_path: Optional[str] = None, show: bool = True):
    """
    Visualize a genome as a directed graph.
    
    Args:
        genome: Genome to visualize
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    node_colors = []
    node_labels = {}
    
    for node in genome.nodes:
        G.add_node(node.id)
        node_labels[node.id] = f"{node.id}\n{node.activation_fn}"
        
        # Color by type
        if node.type == 'input':
            node_colors.append('lightblue')
        elif node.type == 'output':
            node_colors.append('lightcoral')
        else:
            node_colors.append('lightgreen')
    
    # Add edges
    edge_weights = []
    for conn in genome.connections:
        if conn.enabled:
            G.add_edge(conn.in_id, conn.out_id)
            edge_weights.append(abs(conn.weight))
    
    # Create layout
    # Separate nodes by type for better visualization
    input_nodes = [n.id for n in genome.get_nodes_by_type('input')]
    hidden_nodes = [n.id for n in genome.get_nodes_by_type('hidden')]
    output_nodes = [n.id for n in genome.get_nodes_by_type('output')]
    
    pos = {}
    
    # Position input nodes on the left
    for i, node_id in enumerate(input_nodes):
        pos[node_id] = (0, i)
    
    # Position hidden nodes in the middle
    for i, node_id in enumerate(hidden_nodes):
        pos[node_id] = (1, i - len(hidden_nodes) / 2)
    
    # Position output nodes on the right
    for i, node_id in enumerate(output_nodes):
        pos[node_id] = (2, i)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.9)
    
    # Draw edges with varying thickness based on weight
    if edge_weights:
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [3 * (w / max_weight) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=edge_widths, 
                              alpha=0.6, arrows=True, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8)
    
    plt.title(f"Genome Visualization\n{len(genome.nodes)} nodes, "
             f"{len([c for c in genome.connections if c.enabled])} connections")
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved genome visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_fitness_history(best_fitness: List[float], 
                         avg_fitness: List[float],
                         save_path: Optional[str] = None,
                         show: bool = True):
    """
    Plot fitness history over generations.
    
    Args:
        best_fitness: List of best fitness values per generation
        avg_fitness: List of average fitness values per generation
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    plt.figure(figsize=(10, 6))
    
    generations = range(len(best_fitness))
    
    plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    plt.plot(generations, avg_fitness, 'r--', linewidth=1.5, label='Average Fitness')
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.title('Fitness Evolution Over Generations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved fitness plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_network_complexity(generations: List[int],
                           num_nodes: List[int],
                           num_connections: List[int],
                           save_path: Optional[str] = None,
                           show: bool = True):
    """
    Plot network complexity evolution.
    
    Args:
        generations: List of generation numbers
        num_nodes: List of node counts
        num_connections: List of connection counts
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot nodes
    ax1.plot(generations, num_nodes, 'g-', linewidth=2)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    ax1.set_title('Network Complexity Evolution', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot connections
    ax2.plot(generations, num_connections, 'purple', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Number of Connections', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved complexity plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
