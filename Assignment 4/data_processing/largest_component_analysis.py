import networkit as nk
from collections import Counter

def get_largest_connected_component_size(graphml_file="friends_network.graphml"):
    """
    Get the size of the largest connected component in terms of nodes and edges.
    
    Args:
        graphml_file (str): Path to the GraphML file
        
    Returns:
        dict: Contains nodes and edges count for largest component
    """
    print("Loading graph from GraphML file...")
    
    # Load the graph
    G = nk.graphio.readGraph(graphml_file, nk.Format.GraphML)
    print(f"Total graph: {G.numberOfNodes()} nodes, {G.numberOfEdges()} edges")
    
    # Analyze connected components
    print("Analyzing connected components...")
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    
    num_components = cc.numberOfComponents()
    print(f"Number of connected components: {num_components}")
    
    # Get the partition (which component each node belongs to)
    partition = cc.getPartition()
    
    # Count nodes in each component
    component_counts = Counter()
    for node in G.iterNodes():
        component_id = partition[node]
        component_counts[component_id] += 1
    
    # Find largest component
    largest_comp_id = max(component_counts, key=component_counts.get)
    largest_comp_size = component_counts[largest_comp_id]
    
    print(f"Largest component ID: {largest_comp_id}")
    print(f"Largest component size: {largest_comp_size} nodes")
    
    # Get nodes in largest component
    largest_comp_nodes = []
    for node in G.iterNodes():
        if partition[node] == largest_comp_id:
            largest_comp_nodes.append(node)
    
    # Create subgraph of largest component
    largest_subgraph = nk.graphtools.subgraphFromNodes(G, largest_comp_nodes)
    
    # Get results
    result = {
        'nodes': largest_subgraph.numberOfNodes(),
        'edges': largest_subgraph.numberOfEdges(),
        'density': nk.graphtools.density(largest_subgraph),
        'total_components': num_components
    }
    
    # Additional statistics
    degrees = [largest_subgraph.degree(node) for node in largest_subgraph.iterNodes()]
    result['avg_degree'] = sum(degrees) / len(degrees) if degrees else 0
    result['max_degree'] = max(degrees) if degrees else 0
    result['min_degree'] = min(degrees) if degrees else 0
    
    return result

def main():
    """Main function to analyze largest connected component."""
    
    print("="*60)
    print("LARGEST CONNECTED COMPONENT ANALYSIS")
    print("="*60)
    
    try:
        result = get_largest_connected_component_size()
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Largest Connected Component Size:")
        print(f"  • Nodes: {result['nodes']:,}")
        print(f"  • Edges: {result['edges']:,}")
        print(f"  • Density: {result['density']:.8f}")
        print(f"  • Average degree: {result['avg_degree']:.2f}")
        print(f"  • Max degree: {result['max_degree']:,}")
        print(f"  • Min degree: {result['min_degree']:,}")
        print(f"  • Total components in network: {result['total_components']:,}")
        
        # Calculate percentage of total network
        print(f"\nAs percentage of total network:")
        total_nodes = 1820056  # From your output
        total_edges = 3662313  # From your output
        print(f"  • {result['nodes']/total_nodes*100:.1f}% of all nodes")
        print(f"  • {result['edges']/total_edges*100:.1f}% of all edges")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    results = main() 