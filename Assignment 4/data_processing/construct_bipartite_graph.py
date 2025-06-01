import json
import networkit as nk
from collections import defaultdict
import time

def construct_bipartite_graph_from_reviews(json_file_path):
    """
    Construct a bipartite NetworKit graph from user-app review data.
    
    Args:
        json_file_path (str): Path to the 50_user_app_review.json file
    
    Returns:
        tuple: (graph, user_attributes, app_attributes, user_id_mapping, app_id_mapping)
    """
    print("Loading and parsing user-app review JSON data...")
    start_time = time.time()
    
    # Parse JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Total items in data: {len(data)}")
    
    # Separate users, apps, and review relationships
    users = []
    apps = []
    reviews = []
    
    for item in data:
        if item.get('type') == 'node':
            if 'User' in item.get('labels', []):
                users.append(item)
            elif 'App' in item.get('labels', []):
                apps.append(item)
        elif item.get('type') == 'relationship' and item.get('label') == 'REVIEWED':
            reviews.append(item)
    
    print(f"Found {len(users)} users, {len(apps)} apps, and {len(reviews)} review relationships")
    
    # Create mappings from original IDs to sequential node IDs for NetworKit
    # Users will have IDs 0 to len(users)-1
    # Apps will have IDs len(users) to len(users)+len(apps)-1
    
    user_original_to_nk_id = {}
    app_original_to_nk_id = {}
    user_attributes = {}
    app_attributes = {}
    
    # Map users
    for i, user in enumerate(users):
        original_id = user['id']
        nk_id = i  # Users get IDs 0 to len(users)-1
        user_original_to_nk_id[original_id] = nk_id
        
        user_attributes[nk_id] = {
            'type': 'User',
            'original_id': original_id,
            'personaname': user.get('properties', {}).get('personaname', 'Unknown'),
            'loccountrycode': user.get('properties', {}).get('loccountrycode', 'Unknown'),
            'realname': user.get('properties', {}).get('realname', 'Unknown'),
            'ident': user.get('properties', {}).get('ident', 'Unknown')
        }
    
    # Map apps
    for i, app in enumerate(apps):
        original_id = app['id']
        nk_id = len(users) + i  # Apps get IDs starting after all users
        app_original_to_nk_id[original_id] = nk_id
        
        app_attributes[nk_id] = {
            'type': 'App',
            'original_id': original_id,
            'name': app.get('properties', {}).get('name', 'Unknown'),
            'ident': app.get('properties', {}).get('ident', 'Unknown'),
            'is_free': app.get('properties', {}).get('is_free', False),
            'app_type': app.get('properties', {}).get('type', 'game')
        }
    
    # Create NetworKit graph (bipartite)
    total_nodes = len(users) + len(apps)
    G = nk.Graph(total_nodes, weighted=False, directed=False)
    
    print("Adding review edges to the bipartite graph...")
    edges_added = 0
    skipped_reviews = 0
    
    for review in reviews:
        user_original_id = review.get('start')  # User ID
        app_original_id = review.get('end')     # App ID
        
        # Check if both user and app exist in our mappings
        if (user_original_id in user_original_to_nk_id and 
            app_original_id in app_original_to_nk_id):
            
            user_nk_id = user_original_to_nk_id[user_original_id]
            app_nk_id = app_original_to_nk_id[app_original_id]
            
            # Add edge if it doesn't already exist
            if not G.hasEdge(user_nk_id, app_nk_id):
                G.addEdge(user_nk_id, app_nk_id)
                edges_added += 1
        else:
            skipped_reviews += 1
    
    print(f"Added {edges_added} review edges to the bipartite graph")
    if skipped_reviews > 0:
        print(f"Skipped {skipped_reviews} reviews due to missing user/app nodes")
    
    print(f"Final bipartite graph: {G.numberOfNodes()} nodes, {G.numberOfEdges()} edges")
    print(f"  - {len(users)} user nodes (IDs 0-{len(users)-1})")
    print(f"  - {len(apps)} app nodes (IDs {len(users)}-{total_nodes-1})")
    
    return G, user_attributes, app_attributes, user_original_to_nk_id, app_original_to_nk_id

def analyze_bipartite_graph(G, user_attributes, app_attributes):
    """
    Analyze the bipartite graph.
    
    Args:
        G: NetworKit bipartite graph
        user_attributes: Dictionary of user node attributes
        app_attributes: Dictionary of app node attributes
    """
    print("\n=== Bipartite Graph Analysis ===")
    
    num_users = len(user_attributes)
    num_apps = len(app_attributes)
    
    print(f"Number of user nodes: {num_users}")
    print(f"Number of app nodes: {num_apps}")
    print(f"Total nodes: {G.numberOfNodes()}")
    print(f"Total edges (reviews): {G.numberOfEdges()}")
    
    # Calculate degree statistics for users and apps separately
    user_degrees = []
    app_degrees = []
    
    for node_id in G.iterNodes():
        degree = G.degree(node_id)
        if node_id < num_users:  # User node
            user_degrees.append(degree)
        else:  # App node
            app_degrees.append(degree)
    
    print(f"\nUser degree statistics (number of apps reviewed per user):")
    if user_degrees:
        print(f"  Average: {sum(user_degrees) / len(user_degrees):.2f}")
        print(f"  Max: {max(user_degrees)}")
        print(f"  Min: {min(user_degrees)}")
    
    print(f"\nApp degree statistics (number of reviews per app):")
    if app_degrees:
        print(f"  Average: {sum(app_degrees) / len(app_degrees):.2f}")
        print(f"  Max: {max(app_degrees)}")
        print(f"  Min: {min(app_degrees)}")
    
    # Find most reviewed apps
    app_review_counts = []
    for node_id in range(num_users, num_users + num_apps):
        app_name = app_attributes[node_id]['name']
        review_count = G.degree(node_id)
        app_review_counts.append((app_name, review_count))
    
    app_review_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost reviewed apps:")
    for i, (app_name, count) in enumerate(app_review_counts[:5]):
        print(f"  {i+1}. {app_name}: {count} reviews")
    
    # Find most active users
    user_review_counts = []
    for node_id in range(num_users):
        user_name = user_attributes[node_id]['personaname']
        review_count = G.degree(node_id)
        user_review_counts.append((user_name, review_count))
    
    user_review_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost active users:")
    for i, (user_name, count) in enumerate(user_review_counts[:5]):
        print(f"  {i+1}. {user_name}: {count} reviews")
    
    # Check connectivity
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    print(f"\nConnectivity:")
    print(f"  Number of connected components: {cc.numberOfComponents()}")
    
    # Bipartite density
    max_possible_edges = num_users * num_apps
    density = G.numberOfEdges() / max_possible_edges if max_possible_edges > 0 else 0
    print(f"  Bipartite density: {density:.8f}")

def save_bipartite_graph_data(G, user_attributes, app_attributes, output_prefix="bipartite"):
    """
    Save bipartite graph data in various formats.
    
    Args:
        G: NetworKit bipartite graph
        user_attributes: Dictionary of user node attributes
        app_attributes: Dictionary of app node attributes
        output_prefix: Prefix for output files
    """
    print(f"\nSaving bipartite graph data...")
    
    # Combine attributes for all nodes
    all_attributes = {**user_attributes, **app_attributes}
    
    # Save as edge list with node types
    edge_list_file = f"{output_prefix}_edges.txt"
    with open(edge_list_file, 'w', encoding='utf-8') as f:
        f.write("user_id\tapp_id\tuser_name\tapp_name\n")
        for u, v in G.iterEdges():
            if u < len(user_attributes):  # u is user, v is app
                user_id, app_id = u, v
            else:  # v is user, u is app
                user_id, app_id = v, u
            
            user_name = user_attributes[user_id]['personaname']
            app_name = app_attributes[app_id]['name']
            f.write(f"{user_id}\t{app_id}\t{user_name}\t{app_name}\n")
    print(f"Edge list saved to {edge_list_file}")
    
    # Save node attributes
    attributes_file = f"{output_prefix}_nodes.json"
    with open(attributes_file, 'w', encoding='utf-8') as f:
        json.dump(all_attributes, f, indent=2, ensure_ascii=False)
    print(f"Node attributes saved to {attributes_file}")
    
    # Save as GraphML
    try:
        graphml_file = f"{output_prefix}.graphml"
        nk.graphio.writeGraph(G, graphml_file, nk.Format.GraphML)
        print(f"Bipartite graph saved as GraphML to {graphml_file}")
    except Exception as e:
        print(f"Could not save as GraphML: {e}")

def main():
    """Main function to construct and analyze the bipartite graph."""
    json_file = "exports/50_user_app_review.json"
    
    print("Starting bipartite graph construction from user-app review data...")
    
    try:
        # Construct bipartite graph
        G, user_attrs, app_attrs, user_mapping, app_mapping = construct_bipartite_graph_from_reviews(json_file)
        
        # Analyze bipartite graph
        analyze_bipartite_graph(G, user_attrs, app_attrs)
        
        # Save bipartite graph data
        save_bipartite_graph_data(G, user_attrs, app_attrs, "user_app_bipartite")
        
        print("\nBipartite graph construction completed successfully!")
        
        return G, user_attrs, app_attrs, user_mapping, app_mapping
        
    except FileNotFoundError:
        print(f"Error: Could not find {json_file}")
        print("Please make sure the file exists in the exports/ directory")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    G, user_attrs, app_attrs, user_mapping, app_mapping = main() 