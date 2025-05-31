import json
import networkit as nk
from collections import defaultdict
import time
from construct_combined_graph import construct_combined_graph

def extract_user_friendship_subgraph(G, num_users):
    """
    Extract only the User-User friendship edges from the combined graph.
    
    Args:
        G: Combined NetworKit graph
        num_users: Number of user nodes
    
    Returns:
        NetworKit graph with only friendship edges
    """
    print("Extracting User-User friendship subgraph...")
    
    # Create a new graph with only user nodes
    friendship_graph = nk.Graph(num_users, weighted=False, directed=False)
    
    friendship_edges = 0
    for u, v in G.iterEdges():
        # Only include User-User edges (both nodes < num_users)
        if u < num_users and v < num_users:
            friendship_graph.addEdge(u, v)
            friendship_edges += 1
    
    print(f"Friendship subgraph: {friendship_graph.numberOfNodes()} users, {friendship_edges} friendship edges")
    return friendship_graph

def find_largest_component_users(friendship_graph):
    """
    Find users in the largest connected component of the friendship graph.
    
    Args:
        friendship_graph: NetworKit graph with only friendship edges
    
    Returns:
        set: Set of user IDs in the largest connected component
    """
    print("Finding largest connected component in friendship network...")
    
    cc = nk.components.ConnectedComponents(friendship_graph)
    cc.run()
    
    num_components = cc.numberOfComponents()
    component_sizes_dict = cc.getComponentSizes()
    
    # Convert to sorted list of sizes
    component_sizes = sorted(component_sizes_dict.values(), reverse=True)
    
    print(f"Friendship network has {num_components} connected components")
    print(f"Largest component size: {component_sizes[0]} users")
    print(f"Component size distribution (top 5): {component_sizes[:5]}")
    
    # Get users in the largest component (component 0)
    largest_component_users = set(cc.getPartition().getMembers(0))
    
    return largest_component_users

def filter_apps_by_component_users(G, num_users, largest_component_users, app_attributes):
    """
    Find apps that are reviewed by users in the largest component.
    
    Args:
        G: Combined NetworKit graph
        num_users: Number of user nodes
        largest_component_users: Set of user IDs in largest component
        app_attributes: App attributes dictionary
    
    Returns:
        set: Set of app IDs that are reviewed by largest component users
    """
    print("Finding apps reviewed by largest component users...")
    
    relevant_apps = set()
    total_apps = len(app_attributes)
    
    # Check all User-App edges
    for u, v in G.iterEdges():
        # User-App edge (one node is user, other is app)
        if (u < num_users and v >= num_users) or (v < num_users and u >= num_users):
            user_id = u if u < num_users else v
            app_id = v if v >= num_users else u
            
            # If the user is in the largest component, include this app
            if user_id in largest_component_users:
                relevant_apps.add(app_id)
    
    print(f"Found {len(relevant_apps)} apps reviewed by largest component users (out of {total_apps} total apps)")
    return relevant_apps

def create_filtered_recommender_graph(G, user_attributes, app_attributes, 
                                    largest_component_users, relevant_apps):
    """
    Create a filtered graph containing only:
    - Users from the largest component
    - Apps reviewed by those users
    - All edges between these nodes
    
    Returns:
        tuple: (filtered_graph, filtered_user_attrs, filtered_app_attrs, user_mapping, app_mapping)
    """
    print("Creating filtered graph for recommender system...")
    
    # Create new mappings for the filtered graph
    filtered_users = sorted(list(largest_component_users))
    filtered_apps = sorted(list(relevant_apps))
    
    # New sequential IDs: Users 0 to len(filtered_users)-1, Apps len(filtered_users) to total-1
    old_to_new_user_id = {old_id: new_id for new_id, old_id in enumerate(filtered_users)}
    old_to_new_app_id = {old_id: len(filtered_users) + new_id for new_id, old_id in enumerate(filtered_apps)}
    
    # Create filtered attributes
    filtered_user_attrs = {}
    for new_id, old_id in enumerate(filtered_users):
        filtered_user_attrs[new_id] = user_attributes[old_id].copy()
    
    filtered_app_attrs = {}
    for new_id, old_id in enumerate(filtered_apps):
        new_app_id = len(filtered_users) + new_id
        filtered_app_attrs[new_app_id] = app_attributes[old_id].copy()
    
    # Create filtered graph
    total_filtered_nodes = len(filtered_users) + len(filtered_apps)
    filtered_G = nk.Graph(total_filtered_nodes, weighted=False, directed=False)
    
    friendship_edges = 0
    review_edges = 0
    
    # Add edges
    for u, v in G.iterEdges():
        u_is_relevant_user = u in largest_component_users
        v_is_relevant_user = v in largest_component_users
        u_is_relevant_app = u in relevant_apps
        v_is_relevant_app = v in relevant_apps
        
        # User-User edge (both in largest component)
        if u_is_relevant_user and v_is_relevant_user:
            new_u = old_to_new_user_id[u]
            new_v = old_to_new_user_id[v]
            if not filtered_G.hasEdge(new_u, new_v):
                filtered_G.addEdge(new_u, new_v)
                friendship_edges += 1
        
        # User-App edge (user in component, app is relevant)
        elif ((u_is_relevant_user and v_is_relevant_app) or 
              (v_is_relevant_user and u_is_relevant_app)):
            
            if u_is_relevant_user:
                new_user = old_to_new_user_id[u]
                new_app = old_to_new_app_id[v]
            else:
                new_user = old_to_new_user_id[v]
                new_app = old_to_new_app_id[u]
            
            if not filtered_G.hasEdge(new_user, new_app):
                filtered_G.addEdge(new_user, new_app)
                review_edges += 1
    
    print(f"Filtered graph created:")
    print(f"  Users: {len(filtered_users)}")
    print(f"  Apps: {len(filtered_apps)}")
    print(f"  Total nodes: {filtered_G.numberOfNodes()}")
    print(f"  Friendship edges: {friendship_edges}")
    print(f"  Review edges: {review_edges}")
    print(f"  Total edges: {filtered_G.numberOfEdges()}")
    
    return (filtered_G, filtered_user_attrs, filtered_app_attrs, 
            old_to_new_user_id, old_to_new_app_id)

def analyze_recommender_graph(G, user_attributes, app_attributes):
    """Analyze the filtered graph for recommender system insights."""
    print("\n=== Recommender Graph Analysis ===")
    
    num_users = len(user_attributes)
    num_apps = len(app_attributes)
    
    print(f"Graph composition:")
    print(f"  Users: {num_users}")
    print(f"  Apps: {num_apps}")
    print(f"  Total nodes: {G.numberOfNodes()}")
    
    # Count edge types
    friendship_edges = 0
    review_edges = 0
    
    for u, v in G.iterEdges():
        if u < num_users and v < num_users:
            friendship_edges += 1
        else:
            review_edges += 1
    
    print(f"  Friendship edges: {friendship_edges}")
    print(f"  Review edges: {review_edges}")
    print(f"  Total edges: {G.numberOfEdges()}")
    
    # User activity analysis
    user_degrees = []
    user_friendship_counts = []
    user_review_counts = []
    
    for user_id in range(num_users):
        total_degree = G.degree(user_id)
        user_degrees.append(total_degree)
        
        friendship_count = 0
        review_count = 0
        
        for neighbor in G.iterNeighbors(user_id):
            if neighbor < num_users:
                friendship_count += 1
            else:
                review_count += 1
        
        user_friendship_counts.append(friendship_count)
        user_review_counts.append(review_count)
    
    print(f"\nUser statistics:")
    print(f"  Average friends per user: {sum(user_friendship_counts) / len(user_friendship_counts):.2f}")
    print(f"  Average apps reviewed per user: {sum(user_review_counts) / len(user_review_counts):.2f}")
    print(f"  Max friends: {max(user_friendship_counts)}")
    print(f"  Max reviews: {max(user_review_counts)}")
    
    # App popularity analysis
    app_review_counts = []
    app_names = []
    
    for app_id in range(num_users, num_users + num_apps):
        review_count = G.degree(app_id)
        app_review_counts.append(review_count)
        app_names.append(app_attributes[app_id]['name'])
    
    if app_review_counts:
        print(f"\nApp statistics:")
        print(f"  Average reviews per app: {sum(app_review_counts) / len(app_review_counts):.2f}")
        print(f"  Most reviewed app: {max(app_review_counts)} reviews")
        print(f"  Least reviewed app: {min(app_review_counts)} reviews")
        
        # Show most popular apps
        app_popularity = list(zip(app_names, app_review_counts))
        app_popularity.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nMost popular apps:")
        for i, (name, count) in enumerate(app_popularity[:min(5, len(app_popularity))]):
            print(f"  {i+1}. {name}: {count} reviews")
    
    # Density analysis
    max_friendship_edges = num_users * (num_users - 1) // 2
    max_review_edges = num_users * num_apps
    
    friendship_density = friendship_edges / max_friendship_edges if max_friendship_edges > 0 else 0
    review_density = review_edges / max_review_edges if max_review_edges > 0 else 0
    
    print(f"\nDensity analysis:")
    print(f"  Friendship network density: {friendship_density:.8f}")
    print(f"  User-App review density: {review_density:.8f}")
    
    # Connectivity check (should be connected since we took largest component)
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    print(f"  Connected components: {cc.numberOfComponents()}")

def save_recommender_graph(G, user_attributes, app_attributes, output_prefix="recommender"):
    """Save the filtered recommender graph."""
    print(f"\nSaving recommender graph data...")
    
    num_users = len(user_attributes)
    
    # Save edge list with edge types
    edge_list_file = f"{output_prefix}_edges.txt"
    with open(edge_list_file, 'w', encoding='utf-8') as f:
        f.write("source_id\ttarget_id\tsource_type\ttarget_type\tedge_type\tsource_name\ttarget_name\n")
        
        for u, v in G.iterEdges():
            if u < num_users and v < num_users:
                # Friendship edge
                edge_type = "friendship"
                source_type = "User"
                target_type = "User"
                source_name = user_attributes[u]['personaname']
                target_name = user_attributes[v]['personaname']
            else:
                # Review edge
                edge_type = "review"
                if u < num_users:
                    user_id, app_id = u, v
                else:
                    user_id, app_id = v, u
                
                source_type = "User"
                target_type = "App"
                source_name = user_attributes[user_id]['personaname']
                target_name = app_attributes[app_id]['name']
                u, v = user_id, app_id
            
            f.write(f"{u}\t{v}\t{source_type}\t{target_type}\t{edge_type}\t{source_name}\t{target_name}\n")
    
    print(f"Edge list saved to {edge_list_file}")
    
    # Save node attributes
    all_attributes = {**user_attributes, **app_attributes}
    attributes_file = f"{output_prefix}_nodes.json"
    with open(attributes_file, 'w', encoding='utf-8') as f:
        json.dump(all_attributes, f, indent=2, ensure_ascii=False)
    print(f"Node attributes saved to {attributes_file}")

def main():
    """Main function to create recommender system graph."""
    friends_file = "exports/friends.json"
    reviews_file = "exports/user_app_review.json"
    
    print("Creating recommender system graph from largest connected component...")
    print("Using complete user_app_review.json dataset...")
    
    try:
        # Step 1: Load the combined graph
        print("Step 1: Loading combined graph...")
        G, user_attrs, app_attrs, user_mapping, app_mapping = construct_combined_graph(
            friends_file, reviews_file
        )
        
        num_users = len(user_attrs)
        
        # Step 2: Extract friendship subgraph
        print("\nStep 2: Extracting friendship network...")
        friendship_graph = extract_user_friendship_subgraph(G, num_users)
        
        # Step 3: Find largest connected component
        print("\nStep 3: Finding largest connected component...")
        largest_component_users = find_largest_component_users(friendship_graph)
        
        # Step 4: Filter apps by component users
        print("\nStep 4: Filtering apps...")
        relevant_apps = filter_apps_by_component_users(G, num_users, largest_component_users, app_attrs)
        
        # Step 5: Create filtered graph
        print("\nStep 5: Creating filtered recommender graph...")
        (filtered_G, filtered_user_attrs, filtered_app_attrs, 
         user_id_mapping, app_id_mapping) = create_filtered_recommender_graph(
            G, user_attrs, app_attrs, largest_component_users, relevant_apps
        )
        
        # Step 6: Analyze the recommender graph
        print("\nStep 6: Analyzing recommender graph...")
        analyze_recommender_graph(filtered_G, filtered_user_attrs, filtered_app_attrs)
        
        # Step 7: Save the recommender graph
        print("\nStep 7: Saving recommender graph...")
        save_recommender_graph(filtered_G, filtered_user_attrs, filtered_app_attrs, "recommender_graph")
        
        print("\n" + "="*60)
        print("RECOMMENDER SYSTEM GRAPH READY!")
        print("="*60)
        print("The filtered graph contains:")
        print(f"- {len(filtered_user_attrs)} users from the largest connected component")
        print(f"- {len(filtered_app_attrs)} apps reviewed by these users")
        print(f"- All friendship and review edges between these nodes")
        print("\nThis graph is optimal for collaborative filtering recommendations!")
        
        return (filtered_G, filtered_user_attrs, filtered_app_attrs, 
                user_id_mapping, app_id_mapping)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    (filtered_G, filtered_user_attrs, filtered_app_attrs, 
     user_mapping, app_mapping) = main() 