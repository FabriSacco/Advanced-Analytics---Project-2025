import json
import pickle
import networkit as nk
from datetime import datetime

def import_scores_to_graph(scores_file, mapping_file, graph_file=None, 
                          score_column='sentiment_score', output_prefix="scored_graph"):
    """
    Import sentiment scores back into the graph as edge attributes.
    
    Args:
        scores_file (str): Path to JSON file with sentiment scores from Colab
        mapping_file (str): Path to edge mapping file
        graph_file (str): Path to existing graph pickle file (optional)
        score_column (str): Column name containing the sentiment scores
        output_prefix (str): Prefix for output files
    """
    print("Loading sentiment scores...")
    with open(scores_file, 'r', encoding='utf-8') as f:
        scores_data = json.load(f)
    
    print("Loading edge mappings...")
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    # Create score lookup by edge_id
    score_lookup = {}
    if isinstance(scores_data, list):
        # Format: [{"edge_id": 0, "sentiment_score": 4.2, ...}, ...]
        for record in scores_data:
            edge_id = record.get('edge_id')
            score = record.get(score_column)
            if edge_id is not None and score is not None:
                score_lookup[edge_id] = score
    elif isinstance(scores_data, dict):
        # Format: {"0": 4.2, "1": 3.8, ...} or {"edge_id": {"0": 4.2, ...}}
        if score_column in scores_data:
            score_lookup = {int(k): v for k, v in scores_data[score_column].items()}
        else:
            # Assume direct edge_id -> score mapping
            score_lookup = {int(k): v for k, v in scores_data.items()}
    
    print(f"Loaded {len(score_lookup)} sentiment scores")
    
    # Load or construct graph
    if graph_file and graph_file.endswith('.pkl'):
        print(f"Loading existing graph from {graph_file}...")
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
            G = graph_data['graph']
            user_attributes = graph_data['user_attributes']
            app_attributes = graph_data['app_attributes']
    else:
        print("Constructing new graph...")
        # Import the graph construction function
        from construct_combined_graph import construct_combined_graph
        G, user_attributes, app_attributes, _, _ = construct_combined_graph(
            'exports/friends.json', 
            'exports/user_app_review.json'
        )
    
    # Create edge mapping lookup
    edge_mappings = {}
    for mapping in mapping_data['edge_mappings']:
        edge_id = mapping['edge_id']
        start_node = mapping['start_node']
        end_node = mapping['end_node']
        edge_mappings[edge_id] = (start_node, end_node)
    
    # Load original data to get node mappings
    print("Loading original data to get node mappings...")
    with open('exports/user_app_review.json', 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # Build node mapping (original_id -> graph_node_id)
    user_original_to_nk_id = {}
    app_original_to_nk_id = {}
    
    # Get users and apps from original data
    users = [item for item in original_data if item.get('type') == 'node' and 'User' in item.get('labels', [])]
    apps = [item for item in original_data if item.get('type') == 'node' and 'App' in item.get('labels', [])]
    
    # Map users (same order as in construct_combined_graph)
    for i, user in enumerate(users):
        user_original_to_nk_id[user['id']] = i
    
    # Map apps (offset by number of users)
    for i, app in enumerate(apps):
        app_original_to_nk_id[app['id']] = len(users) + i
    
    # Add sentiment scores as edge attributes
    print("Adding sentiment scores to edges...")
    edges_updated = 0
    edges_not_found = 0
    
    # Create edge attributes dictionary
    edge_attributes = {}
    
    for edge_id, score in score_lookup.items():
        if edge_id in edge_mappings:
            start_original, end_original = edge_mappings[edge_id]
            
            # Convert to graph node IDs
            if (start_original in user_original_to_nk_id and 
                end_original in app_original_to_nk_id):
                
                start_nk = user_original_to_nk_id[start_original]
                end_nk = app_original_to_nk_id[end_original]
                
                # Check if edge exists in graph
                if G.hasEdge(start_nk, end_nk):
                    edge_key = (min(start_nk, end_nk), max(start_nk, end_nk))
                    edge_attributes[edge_key] = {
                        'sentiment_score': score,
                        'edge_id': edge_id
                    }
                    edges_updated += 1
                else:
                    edges_not_found += 1
            else:
                edges_not_found += 1
        else:
            edges_not_found += 1
    
    print(f"Edges updated with sentiment scores: {edges_updated}")
    print(f"Edges not found in graph: {edges_not_found}")
    
    # Save updated graph data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as pickle for NetworkX compatibility
    output_data = {
        'graph': G,
        'user_attributes': user_attributes,
        'app_attributes': app_attributes,
        'edge_attributes': edge_attributes,
        'sentiment_scores_info': {
            'total_scores': len(score_lookup),
            'edges_updated': edges_updated,
            'edges_not_found': edges_not_found,
            'score_column_used': score_column,
            'timestamp': timestamp
        }
    }
    
    output_file = f'exports/{output_prefix}_{timestamp}.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    
    # Save summary as JSON
    summary = {
        'timestamp': timestamp,
        'input_files': {
            'scores_file': scores_file,
            'mapping_file': mapping_file,
            'graph_file': graph_file
        },
        'statistics': {
            'total_sentiment_scores': len(score_lookup),
            'edges_updated': edges_updated,
            'edges_not_found': edges_not_found,
            'total_nodes': G.numberOfNodes(),
            'total_edges': G.numberOfEdges()
        },
        'sample_scores': list(score_lookup.items())[:10]  # First 10 for verification
    }
    
    summary_file = f'exports/{output_prefix}_summary_{timestamp}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== IMPORT COMPLETE ===")
    print(f"Updated graph saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nGraph statistics:")
    print(f"  - Total nodes: {G.numberOfNodes()}")
    print(f"  - Total edges: {G.numberOfEdges()}")
    print(f"  - Edges with sentiment scores: {edges_updated}")
    
    return output_file, summary_file, edge_attributes

if __name__ == "__main__":
    # Example usage
    print("Usage: import_scores_to_graph(scores_file, mapping_file)")
    print("Example: import_scores_to_graph('exports/scored_reviews.json', 'exports/edge_mapping_20241221_143022.json')") 