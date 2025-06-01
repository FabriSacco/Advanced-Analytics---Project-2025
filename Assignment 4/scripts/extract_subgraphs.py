import json
import time

def load_combined_graph_data():
    """Load the combined graph data from the final outputs."""
    print("Loading combined graph data...")
    
    # Load nodes
    with open("outputs/final_recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    print(f"Loaded {len(nodes)} nodes")
    
    # Separate users and apps
    users = {}
    apps = {}
    
    for node_id, node_data in nodes.items():
        node_id = int(node_id)  # Convert string keys to int
        if node_data['type'] == 'User':
            users[node_id] = node_data
        elif node_data['type'] == 'App':
            apps[node_id] = node_data
    
    print(f"  - Users: {len(users)}")
    print(f"  - Apps: {len(apps)}")
    
    return users, apps

def extract_user_user_graph():
    """Extract User-User friendship graph."""
    print("\n=== Extracting User-User Friendship Graph ===")
    
    users, apps = load_combined_graph_data()
    
    # Read edges and filter for User-User friendships
    friendship_edges = []
    total_edges = 0
    
    print("Processing edges from final_recommender_graph_edges.txt...")
    
    with open("outputs/final_recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        
        batch_count = 0
        for line in f:
            total_edges += 1
            parts = line.strip().split('\t')
            
            if len(parts) >= 5:
                source_id = int(parts[0])
                target_id = int(parts[1])
                source_type = parts[2]
                target_type = parts[3]
                edge_type = parts[4]
                
                # Only keep User-User friendships
                if edge_type == "friendship" and source_type == "User" and target_type == "User":
                    friendship_edges.append({
                        'source_id': source_id,
                        'target_id': target_id,
                        'source_name': parts[6] if len(parts) > 6 else 'Unknown',
                        'target_name': parts[7] if len(parts) > 7 else 'Unknown'
                    })
            
            # Progress tracking
            batch_count += 1
            if batch_count % 500000 == 0:
                print(f"  Processed {batch_count:,} edges, found {len(friendship_edges):,} friendships")
    
    print(f"Extraction complete:")
    print(f"  Total edges processed: {total_edges:,}")
    print(f"  Friendship edges found: {len(friendship_edges):,}")
    
    # Save User-User graph
    print("\nSaving User-User friendship graph...")
    
    # Save nodes (users only)
    user_nodes_file = "outputs/user_user_graph_nodes.json"
    with open(user_nodes_file, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)
    print(f"User nodes saved to {user_nodes_file}")
    
    # Save edges
    user_edges_file = "outputs/user_user_graph_edges.txt"
    with open(user_edges_file, 'w', encoding='utf-8') as f:
        f.write("source_id\ttarget_id\tsource_name\ttarget_name\n")
        for edge in friendship_edges:
            f.write(f"{edge['source_id']}\t{edge['target_id']}\t{edge['source_name']}\t{edge['target_name']}\n")
    print(f"Friendship edges saved to {user_edges_file}")
    
    return len(users), len(friendship_edges)

def extract_user_app_graph():
    """Extract User-App review graph with playtime weights."""
    print("\n=== Extracting User-App Review Graph ===")
    
    users, apps = load_combined_graph_data()
    
    # Read edges and filter for User-App reviews
    review_edges = []
    total_edges = 0
    
    print("Processing edges from final_recommender_graph_edges.txt...")
    
    weight_stats = {
        'total_weight': 0,
        'min_weight': float('inf'),
        'max_weight': 0,
        'weight_distribution': {}
    }
    
    sentiment_stats = {
        'total_sentiment': 0,
        'sentiment_count': 0,
        'missing_sentiment': 0,
        'sentiment_distribution': {}
    }
    
    with open("outputs/final_recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
        header = f.readline()  # Skip header
        
        batch_count = 0
        for line in f:
            total_edges += 1
            parts = line.strip().split('\t')
            
            if len(parts) >= 7:
                source_id = int(parts[0])
                target_id = int(parts[1])
                source_type = parts[2]
                target_type = parts[3]
                edge_type = parts[4]
                weight = float(parts[5])
                sentiment_score = parts[6] if parts[6] != "" else None
                
                # Only keep User-App reviews
                if edge_type == "review" and source_type == "User" and target_type == "App":
                    source_name = parts[7] if len(parts) > 7 else 'Unknown'
                    target_name = parts[8] if len(parts) > 8 else 'Unknown'
                    
                    review_edges.append({
                        'user_id': source_id,
                        'app_id': target_id,
                        'weight': weight,
                        'sentiment_score': sentiment_score,
                        'user_name': source_name,
                        'app_name': target_name
                    })
                    
                    # Update weight statistics
                    weight_stats['total_weight'] += weight
                    weight_stats['min_weight'] = min(weight_stats['min_weight'], weight)
                    weight_stats['max_weight'] = max(weight_stats['max_weight'], weight)
                    
                    weight_bucket = int(weight)
                    weight_stats['weight_distribution'][weight_bucket] = weight_stats['weight_distribution'].get(weight_bucket, 0) + 1
                    
                    # Update sentiment statistics
                    if sentiment_score is not None and sentiment_score != "None":
                        sentiment_score_float = float(sentiment_score)
                        sentiment_stats['total_sentiment'] += sentiment_score_float
                        sentiment_stats['sentiment_count'] += 1
                        sentiment_stats['sentiment_distribution'][int(sentiment_score_float)] = sentiment_stats['sentiment_distribution'].get(int(sentiment_score_float), 0) + 1
                    else:
                        sentiment_stats['missing_sentiment'] += 1
            
            # Progress tracking
            batch_count += 1
            if batch_count % 500000 == 0:
                print(f"  Processed {batch_count:,} edges, found {len(review_edges):,} reviews")
    
    print(f"Extraction complete:")
    print(f"  Total edges processed: {total_edges:,}")
    print(f"  Review edges found: {len(review_edges):,}")
    
    # Print statistics
    if review_edges:
        avg_weight = weight_stats['total_weight'] / len(review_edges)
        print(f"\nWeight statistics:")
        print(f"  Average weight: {avg_weight:.3f}")
        print(f"  Min weight: {weight_stats['min_weight']:.3f}")
        print(f"  Max weight: {weight_stats['max_weight']:.3f}")
        
        print(f"  Weight distribution:")
        for weight_bucket in sorted(weight_stats['weight_distribution'].keys()):
            count = weight_stats['weight_distribution'][weight_bucket]
            percentage = (count / len(review_edges)) * 100
            print(f"    Weight {weight_bucket}-{weight_bucket+1}: {count} edges ({percentage:.1f}%)")
        
        if sentiment_stats['sentiment_count'] > 0:
            avg_sentiment = sentiment_stats['total_sentiment'] / sentiment_stats['sentiment_count']
            print(f"\nSentiment statistics:")
            print(f"  Average sentiment: {avg_sentiment:.2f}")
            print(f"  Reviews with sentiment: {sentiment_stats['sentiment_count']}")
            print(f"  Reviews without sentiment: {sentiment_stats['missing_sentiment']}")
            
            print(f"  Sentiment distribution:")
            for sentiment in sorted(sentiment_stats['sentiment_distribution'].keys()):
                count = sentiment_stats['sentiment_distribution'][sentiment]
                percentage = (count / len(review_edges)) * 100
                print(f"    Score {sentiment}: {count} reviews ({percentage:.1f}%)")
    
    # Save User-App graph
    print("\nSaving User-App review graph...")
    
    # Save all nodes (users + apps)
    all_nodes = {**users, **apps}
    user_app_nodes_file = "outputs/user_app_graph_nodes.json"
    with open(user_app_nodes_file, 'w', encoding='utf-8') as f:
        json.dump(all_nodes, f, indent=2, ensure_ascii=False)
    print(f"User and App nodes saved to {user_app_nodes_file}")
    
    # Save weighted edges with sentiment scores
    user_app_edges_file = "outputs/user_app_graph_edges.txt"
    with open(user_app_edges_file, 'w', encoding='utf-8') as f:
        f.write("user_id\tapp_id\tweight\tsentiment_score\tuser_name\tapp_name\n")
        for edge in review_edges:
            sentiment = edge['sentiment_score'] if edge['sentiment_score'] is not None else ""
            f.write(f"{edge['user_id']}\t{edge['app_id']}\t{edge['weight']:.6f}\t{sentiment}\t{edge['user_name']}\t{edge['app_name']}\n")
    print(f"Weighted review edges saved to {user_app_edges_file}")
    
    return len(users), len(apps), len(review_edges)

def main():
    """Main function to extract both subgraphs."""
    print("=== Extracting Subgraphs from Combined Graph ===")
    print("This will create separate User-User and User-App graph files")
    
    start_time = time.time()
    
    try:
        # Extract User-User friendship graph
        user_count, friendship_count = extract_user_user_graph()
        
        # Extract User-App review graph  
        user_count_2, app_count, review_count = extract_user_app_graph()
        
        # Summary
        print("\n" + "="*60)
        print("SUBGRAPH EXTRACTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nUser-User Friendship Graph:")
        print(f"  Nodes: {user_count:,} users")
        print(f"  Edges: {friendship_count:,} friendships (unweighted)")
        print(f"  Files:")
        print(f"    - outputs/user_user_graph_nodes.json")
        print(f"    - outputs/user_user_graph_edges.txt")
        
        print(f"\nUser-App Review Graph:")
        print(f"  Nodes: {user_count_2 + app_count:,} total ({user_count_2:,} users + {app_count} apps)")
        print(f"  Edges: {review_count:,} reviews (weighted by playtime, with sentiment scores)")
        print(f"  Files:")
        print(f"    - outputs/user_app_graph_nodes.json")
        print(f"    - outputs/user_app_graph_edges.txt")
        
        total_time = time.time() - start_time
        print(f"\nTotal extraction time: {total_time:.2f} seconds")
        
        print(f"\n✓ Both subgraphs are ready for independent analysis!")
        print(f"✓ User-User graph contains friendship relationships only")
        print(f"✓ User-App graph contains weighted review relationships with sentiment scores")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Please make sure the combined graph files exist in outputs/")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 