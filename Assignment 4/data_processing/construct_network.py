import json
import networkit as nk
from collections import defaultdict
import time

def construct_network_from_friends_json(json_file_path):
    """
    Construct a NetworKit graph from friends.json data.
    
    Args:
        json_file_path (str): Path to the friends.json file
    
    Returns:
        tuple: (graph, node_attributes, id_mapping)
    """
    print("Loading and parsing JSON data...")
    start_time = time.time()
    
    # Parse JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Total items in data: {len(data)}")
    
    # Separate nodes and relationships
    nodes = []
    relationships = []
    
    for item in data:
        if item.get('type') == 'node':
            nodes.append(item)
        elif item.get('type') == 'relationship':
            relationships.append(item)
    
    print(f"Found {len(nodes)} nodes and {len(relationships)} relationships")
    
    # Create mapping from original node IDs to sequential node IDs for NetworKit
    original_to_nk_id = {}
    node_attributes = {}
    
    for i, node in enumerate(nodes):
        original_id = node['id']
        original_to_nk_id[original_id] = i
        
        # Extract loccountrycode attribute
        loccountrycode = node.get('properties', {}).get('loccountrycode', 'Unknown')
        node_attributes[i] = {
            'original_id': original_id,
            'loccountrycode': loccountrycode,
            'personaname': node.get('properties', {}).get('personaname', 'Unknown'),
            'realname': node.get('properties', {}).get('realname', 'Unknown')
        }
    
    # Create NetworKit graph
    num_nodes = len(nodes)
    G = nk.Graph(num_nodes, weighted=False, directed=False)
    
    print("Adding edges to the graph...")
    edges_added = 0
    
    for rel in relationships:
        start_id = rel.get('start')
        end_id = rel.get('end')
        
        # Check if both nodes exist in our mapping
        if start_id in original_to_nk_id and end_id in original_to_nk_id:
            nk_start = original_to_nk_id[start_id]
            nk_end = original_to_nk_id[end_id]
            
            # Add edge if it doesn't already exist
            if not G.hasEdge(nk_start, nk_end):
                G.addEdge(nk_start, nk_end)
                edges_added += 1
    
    print(f"Added {edges_added} edges to the graph")
    print(f"Final graph: {G.numberOfNodes()} nodes, {G.numberOfEdges()} edges")
    
    return G, node_attributes, original_to_nk_id

def analyze_network(G, node_attributes):
    """
    Perform basic network analysis.
    
    Args:
        G: NetworKit graph
        node_attributes: Dictionary of node attributes
    """
    print("\n=== Network Analysis ===")
    print(f"Number of nodes: {G.numberOfNodes()}")
    print(f"Number of edges: {G.numberOfEdges()}")
    print(f"Density: {nk.graphtools.density(G):.6f}")
    
    # Check if graph is connected
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    print(f"Number of connected components: {cc.numberOfComponents()}")
    print(f"Size of largest component: {cc.getComponentSizes()[0] if cc.getComponentSizes() else 0}")
    
    # Country code distribution
    country_counts = defaultdict(int)
    for node_id, attrs in node_attributes.items():
        country_counts[attrs['loccountrycode']] += 1
    
    print(f"\nCountry code distribution:")
    sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
    for country, count in sorted_countries[:10]:  # Top 10 countries
        print(f"  {country}: {count} users")
    
    # Basic centrality measures (on largest component if disconnected)
    if cc.numberOfComponents() > 1:
        print("\nNote: Graph has multiple components. Centrality measures will be computed on the largest component.")
        largest_component = cc.getPartition().getMembers(0)  # Get largest component
        subgraph = nk.graphtools.subgraphFromNodes(G, largest_component)
        G_analysis = subgraph
    else:
        G_analysis = G
    
    # Degree centrality
    degrees = [G_analysis.degree(node) for node in G_analysis.iterNodes()]
    print(f"\nDegree statistics:")
    print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Min degree: {min(degrees)}")

def save_network_data(G, node_attributes, output_prefix="network"):
    """
    Save network data in various formats.
    
    Args:
        G: NetworKit graph
        node_attributes: Dictionary of node attributes
        output_prefix: Prefix for output files
    """
    print(f"\nSaving network data...")
    
    # Save as edge list
    edge_list_file = f"{output_prefix}_edges.txt"
    with open(edge_list_file, 'w') as f:
        f.write("source\ttarget\n")
        for u, v in G.iterEdges():
            f.write(f"{u}\t{v}\n")
    print(f"Edge list saved to {edge_list_file}")
    
    # Save node attributes
    attributes_file = f"{output_prefix}_nodes.json"
    with open(attributes_file, 'w') as f:
        json.dump(node_attributes, f, indent=2)
    print(f"Node attributes saved to {attributes_file}")
    
    # Save as GraphML (if needed for other tools)
    try:
        graphml_file = f"{output_prefix}.graphml"
        nk.graphio.writeGraph(G, graphml_file, nk.Format.GraphML)
        print(f"Graph saved as GraphML to {graphml_file}")
    except Exception as e:
        print(f"Could not save as GraphML: {e}")

def main():
    """Main function to construct and analyze the network."""
    json_file = "exports/friends.json"
    
    print("Starting network construction from friends.json...")
    print("This may take a while due to the large file size...")
    
    try:
        # Construct network
        G, node_attributes, id_mapping = construct_network_from_friends_json(json_file)
        
        # Analyze network
        analyze_network(G, node_attributes)
        
        # Save network data
        save_network_data(G, node_attributes, "friends_network")
        
        print("\nNetwork construction completed successfully!")
        
        return G, node_attributes, id_mapping
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please make sure the file exists in the exports/ directory")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    G, node_attributes, id_mapping = main() 