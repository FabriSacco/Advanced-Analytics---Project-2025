import json
import networkit as nk
from collections import defaultdict
import time
import math

def load_friends_data(json_file_path):
    """Load and parse friends data."""
    print("Loading friends data...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    users = []
    friendships = []
    
    for item in data:
        if item.get('type') == 'node':
            users.append(item)
        elif item.get('type') == 'relationship':
            friendships.append(item)
    
    print(f"Friends data: {len(users)} users, {len(friendships)} friendships")
    return users, friendships

def load_review_data(json_file_path):
    """Load and parse review data."""
    print("Loading review data...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
    
    print(f"Review data: {len(users)} users, {len(apps)} apps, {len(reviews)} reviews")
    return users, apps, reviews

def load_game_categories():
    """Load game categories from the JSON file."""
    try:
        with open("outputs/game_categories.json", 'r', encoding='utf-8') as f:
            category_data = json.load(f)
        return category_data.get("game_to_category_mapping", {})
    except FileNotFoundError:
        print("Warning: outputs/game_categories.json not found. Apps will not have category attributes.")
        return {}
    except Exception as e:
        print(f"Warning: Error loading game categories: {e}")
        return {}

def load_sentiment_scores(scores_file_path):
    """Load sentiment scores from the JSON file."""
    print("Loading sentiment scores...")
    try:
        with open(scores_file_path, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        
        # Convert to dictionary for faster lookup: edge_id -> sentiment_score
        scores_dict = {}
        for item in scores_data:
            edge_id = item.get('edge_id')
            sentiment_score = item.get('sentiment_score')
            if edge_id is not None and sentiment_score is not None:
                scores_dict[edge_id] = sentiment_score
        
        print(f"Loaded sentiment scores for {len(scores_dict)} edges")
        return scores_dict
    except FileNotFoundError:
        print(f"Warning: Sentiment scores file not found at {scores_file_path}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading sentiment scores: {e}")
        return {}

def construct_combined_graph(friends_json_path, reviews_json_path, sentiment_scores_path=None):
    """
    Construct a combined graph with Users and Apps as nodes, containing both:
    - User-User edges (friendships)
    - User-App edges (reviews) with weights based on playtime only
    
    Args:
        friends_json_path (str): Path to friends.json
        reviews_json_path (str): Path to user_app_review.json
        sentiment_scores_path (str, optional): Path to sentiment scores JSON file
    
    Returns:
        tuple: (graph, user_attributes, app_attributes, user_id_mapping, app_id_mapping, edge_sentiment_scores)
    """
    start_time = time.time()
    
    # Load data from both sources
    friends_users, friendships = load_friends_data(friends_json_path)
    review_users, apps, reviews = load_review_data(reviews_json_path)
    
    # Load game categories
    print("Loading game categories...")
    game_to_category = load_game_categories()
    
    # Load sentiment scores
    sentiment_scores = {}
    if sentiment_scores_path:
        sentiment_scores = load_sentiment_scores(sentiment_scores_path)
    
    # Merge user data (friends users + review users), handling duplicates
    print("Merging user data...")
    all_users = {}  # Use dict to handle duplicates by ID
    
    # Add friends users
    for user in friends_users:
        user_id = user['id']
        all_users[user_id] = {
            'id': user_id,
            'properties': user.get('properties', {}),
            'source': 'friends'
        }
    
    # Add review users (merge with existing if duplicate)
    for user in review_users:
        user_id = user['id']
        if user_id in all_users:
            # Merge properties, prioritizing non-empty values
            existing_props = all_users[user_id]['properties']
            new_props = user.get('properties', {})
            for key, value in new_props.items():
                if key not in existing_props or not existing_props[key] or existing_props[key] == 'Unknown':
                    existing_props[key] = value
            all_users[user_id]['source'] = 'both'
        else:
            all_users[user_id] = {
                'id': user_id,
                'properties': user.get('properties', {}),
                'source': 'reviews'
            }
    
    # Convert back to list for consistent indexing
    merged_users = list(all_users.values())
    print(f"Total unique users after merging: {len(merged_users)}")
    
    # Create node mappings
    # Users: IDs 0 to len(merged_users)-1
    # Apps: IDs len(merged_users) to len(merged_users)+len(apps)-1
    
    user_original_to_nk_id = {}
    app_original_to_nk_id = {}
    user_attributes = {}
    app_attributes = {}
    
    # Map users
    for i, user in enumerate(merged_users):
        original_id = user['id']
        nk_id = i
        user_original_to_nk_id[original_id] = nk_id
        
        props = user['properties']
        user_attributes[nk_id] = {
            # Labels
            'type': 'User',
            'original_id': original_id,
            'personaname': props.get('personaname', 'Unknown'),
            # Attributes  
            'loccountrycode': props.get('loccountrycode', 'Unknown')
        }
    
    # Map apps with categories
    apps_with_categories = 0
    apps_without_categories = 0
    
    for i, app in enumerate(apps):
        original_id = app['id']
        nk_id = len(merged_users) + i
        app_original_to_nk_id[original_id] = nk_id
        
        props = app.get('properties', {})
        app_name = props.get('name', 'Unknown')
        
        # Get category for this app
        category = game_to_category.get(app_name, 'Uncategorized')
        if category != 'Uncategorized':
            apps_with_categories += 1
        else:
            apps_without_categories += 1
        
        app_attributes[nk_id] = {
            # Labels
            'type': 'App',
            'original_id': original_id,
            'name': app_name,
            # Attributes
            'app_type': 1 if props.get('type', 'game') == 'game' else 0,  # Binarized: 1 for game, 0 for dlc
            'category': category
        }
    
    print(f"App categorization: {apps_with_categories} apps categorized, {apps_without_categories} uncategorized")
    
    # Create NetworKit graph
    total_nodes = len(merged_users) + len(apps)
    G = nk.Graph(total_nodes, weighted=True, directed=False)
    
    print(f"Weighted graph initialized with {total_nodes} nodes:")
    print(f"  - {len(merged_users)} user nodes (IDs 0-{len(merged_users)-1})")
    print(f"  - {len(apps)} app nodes (IDs {len(merged_users)}-{total_nodes-1})")
    
    # Add User-User edges (friendships) - UNWEIGHTED (weight = 1.0)
    print("Adding friendship edges (unweighted)...")
    friendship_edges_added = 0
    friendship_edges_skipped = 0
    
    for friendship in friendships:
        user1_id = friendship.get('start')
        user2_id = friendship.get('end')
        
        # Check if both users exist in our mapping
        if (user1_id in user_original_to_nk_id and 
            user2_id in user_original_to_nk_id):
            
            nk_user1 = user_original_to_nk_id[user1_id]
            nk_user2 = user_original_to_nk_id[user2_id]
            
            # Add edge if it doesn't already exist
            if not G.hasEdge(nk_user1, nk_user2):
                G.addEdge(nk_user1, nk_user2, 1.0)  # Friendship weight = 1.0
                friendship_edges_added += 1
        else:
            friendship_edges_skipped += 1
    
    print(f"Added {friendship_edges_added} friendship edges (weight=1.0)")
    if friendship_edges_skipped > 0:
        print(f"Skipped {friendship_edges_skipped} friendships due to missing users")
    
    # ========== MODIFICATION: Add WEIGHTED User-App edges (reviews) ==========
    print("Adding weighted review edges based on playtime and sentiment...")
    review_edges_added = 0
    review_edges_skipped = 0
    
    # Data structure to store edge sentiment scores for recommender system
    edge_sentiment_scores = {}  # (user_nk_id, app_nk_id) -> sentiment_score
    
    playtime_stats = {
        'total_playtime': 0,
        'min_playtime': float('inf'),
        'max_playtime': 0,
        'zero_playtime': 0,
        'missing_playtime': 0,
        'weight_distribution': defaultdict(int)
    }
    
    sentiment_stats = {
        'total_sentiment_scores': 0,
        'sentiment_scores_used': 0,
        'missing_sentiment': 0,
        'sentiment_distribution': defaultdict(int)
    }
    
    def calculate_combined_weight(playtime_hours, sentiment_score, edge_id):
        """
        Calculate edge weight based on playtime hours only.
        Sentiment score is preserved separately for recommender system use.
        """
        # Calculate playtime component only (0.1 to 10.0)
        if playtime_hours is None or playtime_hours <= 0:
            playtime_weight = 0.1  # Minimum weight for reviews with no/zero playtime
        else:
            playtime_weight = min(math.log(1 + playtime_hours), 10.0)
        
        # Store sentiment score separately (no weight calculation here)
        if sentiment_score is not None:
            sentiment_stats['sentiment_scores_used'] += 1
            sentiment_stats['total_sentiment_scores'] += sentiment_score
        else:
            sentiment_stats['missing_sentiment'] += 1
        
        return playtime_weight, sentiment_score
    
    for edge_index, review in enumerate(reviews):
        user_id = review.get('start')
        app_id = review.get('end')
        
        # Check if both user and app exist in our mappings
        if (user_id in user_original_to_nk_id and 
            app_id in app_original_to_nk_id):
            
            nk_user = user_original_to_nk_id[user_id]
            nk_app = app_original_to_nk_id[app_id]
            
            # Extract playtime from review properties
            review_props = review.get('properties', {})
            playtime_hours = review_props.get('author_playtime_at_review')
            
            # Get sentiment score for this edge (using edge_index as edge_id)
            sentiment_score = sentiment_scores.get(edge_index)
            
            # Calculate combined weight
            weight, sentiment_score = calculate_combined_weight(
                playtime_hours, sentiment_score, edge_index
            )
            
            # Update statistics
            if playtime_hours is None:
                playtime_stats['missing_playtime'] += 1
            elif playtime_hours == 0:
                playtime_stats['zero_playtime'] += 1
            else:
                playtime_stats['total_playtime'] += playtime_hours
                playtime_stats['min_playtime'] = min(playtime_stats['min_playtime'], playtime_hours)
                playtime_stats['max_playtime'] = max(playtime_stats['max_playtime'], playtime_hours)
            
            # Track sentiment distribution
            if sentiment_score is not None:
                sentiment_stats['sentiment_distribution'][sentiment_score] += 1
            
            # Track weight distribution (based on playtime only)
            weight_bucket = int(weight)
            playtime_stats['weight_distribution'][weight_bucket] += 1
            
            # Add weighted edge if it doesn't already exist
            if not G.hasEdge(nk_user, nk_app):
                G.addEdge(nk_user, nk_app, weight)  # Weight based on playtime only
                review_edges_added += 1
            
            # Populate edge sentiment scores
            edge_sentiment_scores[(nk_user, nk_app)] = sentiment_score
        else:
            review_edges_skipped += 1
    
    # Print comprehensive statistics
    print(f"Added {review_edges_added} weighted review edges")
    print(f"\n=== Playtime Statistics ===")
    if playtime_stats['total_playtime'] > 0:
        valid_playtime_count = review_edges_added - playtime_stats['missing_playtime'] - playtime_stats['zero_playtime']
        if valid_playtime_count > 0:
            avg_playtime = playtime_stats['total_playtime'] / valid_playtime_count
            print(f"Playtime range: {playtime_stats['min_playtime']:.1f} - {playtime_stats['max_playtime']:.1f} hours")
            print(f"Average playtime: {avg_playtime:.1f} hours")
    
    print(f"Missing playtime: {playtime_stats['missing_playtime']} reviews")
    print(f"Zero playtime: {playtime_stats['zero_playtime']} reviews")
    
    print(f"\n=== Sentiment Statistics ===")
    print(f"Sentiment scores used: {sentiment_stats['sentiment_scores_used']} reviews")
    print(f"Missing sentiment: {sentiment_stats['missing_sentiment']} reviews")
    if sentiment_stats['sentiment_scores_used'] > 0:
        avg_sentiment = sentiment_stats['total_sentiment_scores'] / sentiment_stats['sentiment_scores_used']
        print(f"Average sentiment score: {avg_sentiment:.2f}")
    
    print(f"Sentiment score distribution:")
    for score in sorted(sentiment_stats['sentiment_distribution'].keys()):
        count = sentiment_stats['sentiment_distribution'][score]
        percentage = (count / review_edges_added) * 100
        print(f"  Score {score}: {count} reviews ({percentage:.1f}%)")
    
    print(f"\n=== Playtime Weight Distribution ===")
    for weight_bucket in sorted(playtime_stats['weight_distribution'].keys()):
        count = playtime_stats['weight_distribution'][weight_bucket]
        percentage = (count / review_edges_added) * 100
        print(f"  Weight {weight_bucket}-{weight_bucket+1}: {count} edges ({percentage:.1f}%)")

    print(f"\nFinal weighted combined graph:")
    print(f"  Total nodes: {G.numberOfNodes()}")
    print(f"  Total edges: {G.numberOfEdges()}")
    print(f"    - Friendship edges (weight=1.0): {friendship_edges_added}")
    print(f"    - Review edges (weighted by playtime only): {review_edges_added}")
    print(f"    - Sentiment scores preserved separately for recommender system")
    print(f"Data loading and weighted graph construction completed in {time.time() - start_time:.2f} seconds")
    
    return G, user_attributes, app_attributes, user_original_to_nk_id, app_original_to_nk_id, edge_sentiment_scores

def analyze_combined_graph(G, user_attributes, app_attributes, skip_expensive=False):
    """Analyze the combined graph with optional expensive operations."""
    print("\n=== Combined Graph Analysis ===")
    
    num_users = len(user_attributes)
    num_apps = len(app_attributes)
    
    print(f"Node composition:")
    print(f"  Users: {num_users}")
    print(f"  Apps: {num_apps}")
    print(f"  Total: {G.numberOfNodes()}")
    
    # Separate edges by type - optimized counting
    user_user_edges = 0
    user_app_edges = 0
    
    print("Counting edge types...")
    for u, v in G.iterEdges():
        if u < num_users and v < num_users:
            user_user_edges += 1
        else:
            user_app_edges += 1
    
    print(f"\nEdge composition:")
    print(f"  User-User (friendships): {user_user_edges}")
    print(f"  User-App (reviews): {user_app_edges}")
    print(f"  Total: {G.numberOfEdges()}")
    
    # App category analysis - fast
    category_counts = defaultdict(int)
    category_reviews = defaultdict(int)
    
    for app_id in range(num_users, num_users + num_apps):
        category = app_attributes[app_id].get('category', 'Uncategorized')
        review_count = G.degree(app_id)
        category_counts[category] += 1
        category_reviews[category] += review_count
    
    print(f"\nApp categories:")
    for category in sorted(category_counts.keys()):
        app_count = category_counts[category]
        total_reviews = category_reviews[category]
        avg_reviews = total_reviews / app_count if app_count > 0 else 0
        print(f"  {category}: {app_count} apps ({total_reviews} reviews, avg {avg_reviews:.1f})")
    
    if skip_expensive:
        print(f"\n⚠️  SKIPPING EXPENSIVE ANALYSIS (degree analysis & connectivity)")
        print(f"  This is normal for large graphs to speed up processing")
        print(f"  Detailed analysis will be performed on the filtered graph")
        return
    
    # EXPENSIVE OPERATIONS - only run on smaller graphs
    print(f"\nPerforming detailed analysis (this may take time for large graphs)...")
    
    # User degree analysis (combining both friendship and review connections)
    print("  Analyzing user connectivity...")
    user_degrees = []
    user_friendship_degrees = []
    user_review_degrees = []
    
    # Sample only first 1000 users for large graphs to estimate
    sample_size = min(1000, num_users)
    sample_users = list(range(0, num_users, max(1, num_users // sample_size)))[:sample_size]
    
    for user_id in sample_users:
        total_degree = G.degree(user_id)
        user_degrees.append(total_degree)
        
        # Count friendships vs reviews separately
        friendship_count = 0
        review_count = 0
        
        for neighbor in G.iterNeighbors(user_id):
            if neighbor < num_users:  # Another user (friendship)
                friendship_count += 1
            else:  # An app (review)
                review_count += 1
        
        user_friendship_degrees.append(friendship_count)
        user_review_degrees.append(review_count)
    
    print(f"\nUser connectivity statistics (sampled from {sample_size} users):")
    print(f"  Average total degree per user: {sum(user_degrees) / len(user_degrees):.2f}")
    print(f"  Average friendships per user: {sum(user_friendship_degrees) / len(user_friendship_degrees):.2f}")
    print(f"  Average reviews per user: {sum(user_review_degrees) / len(user_review_degrees):.2f}")
    
    # App degree analysis - this is fast since there are only ~80 apps
    app_degrees = []
    for app_id in range(num_users, num_users + num_apps):
        app_degrees.append(G.degree(app_id))
    
    if app_degrees:
        print(f"\nApp review statistics:")
        print(f"  Average reviews per app: {sum(app_degrees) / len(app_degrees):.2f}")
        print(f"  Max reviews for an app: {max(app_degrees)}")
        print(f"  Min reviews for an app: {min(app_degrees)}")
    
    # Skip connectivity analysis for very large graphs (it's very slow)
    if G.numberOfNodes() > 100000:
        print(f"\n⚠️  SKIPPING ConnectedComponents analysis for large graph ({G.numberOfNodes()} nodes)")
        print(f"  This analysis will be performed during largest component extraction")
    else:
        # Connectivity analysis
        print("  Analyzing connectivity...")
        cc = nk.components.ConnectedComponents(G)
        cc.run()
        print(f"\nConnectivity:")
        print(f"  Connected components: {cc.numberOfComponents()}")
        if cc.numberOfComponents() > 0:
            component_sizes_dict = cc.getComponentSizes()
            component_sizes = sorted(component_sizes_dict.values(), reverse=True)
            print(f"  Largest component size: {component_sizes[0]} nodes")

def analyze_edge_weights(G, user_attributes, app_attributes, skip_expensive=False):
    """Analyze the distribution of edge weights in the graph (playtime-based for reviews)."""
    print("\n=== Edge Weight Analysis ===")
    
    if skip_expensive and G.numberOfEdges() > 100000:
        print(f"⚠️  SKIPPING detailed edge weight analysis for large graph ({G.numberOfEdges()} edges)")
        print(f"  Basic edge counts already shown in graph analysis")
        print(f"  Detailed weight analysis will be performed on the filtered graph")
        return
    
    num_users = len(user_attributes)
    friendship_weights = []
    review_weights = []
    
    # For large graphs, sample edges to estimate distribution
    total_edges = G.numberOfEdges()
    if total_edges > 50000:
        print(f"Sampling edge weights from large graph ({total_edges} total edges)...")
        sample_size = min(10000, total_edges)
        edge_count = 0
        target_interval = max(1, total_edges // sample_size)
        
        for u, v in G.iterEdges():
            if edge_count % target_interval == 0:
                weight = G.weight(u, v)
                
                if u < num_users and v < num_users:
                    # User-User edge (friendship)
                    friendship_weights.append(weight)
                else:
                    # User-App edge (review)
                    review_weights.append(weight)
                
                if len(friendship_weights) + len(review_weights) >= sample_size:
                    break
            edge_count += 1
        
        print(f"Analyzed {len(friendship_weights) + len(review_weights)} sampled edges")
    else:
        print(f"Analyzing all edge weights...")
        for u, v in G.iterEdges():
            weight = G.weight(u, v)
            
            if u < num_users and v < num_users:
                # User-User edge (friendship)
                friendship_weights.append(weight)
            else:
                # User-App edge (review)
                review_weights.append(weight)
    
    print(f"Friendship edges (User-User):")
    if friendship_weights:
        print(f"  Count: {len(friendship_weights)}")
        print(f"  All weights: {set(friendship_weights)}")  # Should all be 1.0
    
    print(f"\nReview edges (User-App, weighted by playtime):")
    if review_weights:
        print(f"  Count: {len(review_weights)}")
        print(f"  Min weight: {min(review_weights):.3f}")
        print(f"  Max weight: {max(review_weights):.3f}")
        print(f"  Average weight: {sum(review_weights)/len(review_weights):.3f}")
        
        # Weight distribution
        weight_ranges = [(0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10)]
        for min_w, max_w in weight_ranges:
            count = sum(1 for w in review_weights if min_w <= w < max_w)
            percentage = (count / len(review_weights)) * 100
            print(f"  Weight {min_w}-{max_w}: {count} edges ({percentage:.1f}%)")

def save_combined_graph_data(G, user_attributes, app_attributes, output_prefix="combined"):
    """Save the combined graph data - OPTIMIZED VERSION."""
    print(f"\nSaving combined graph data...")
    
    num_users = len(user_attributes)
    total_edges = G.numberOfEdges()
    
    # Save edge list with edge types - OPTIMIZED for large graphs
    edge_list_file = f"{output_prefix}_edges.txt"
    print(f"Writing {total_edges:,} edges to {edge_list_file}...")
    
    with open(edge_list_file, 'w', encoding='utf-8') as f:
        f.write("source_id\ttarget_id\tsource_type\ttarget_type\tedge_type\tsource_name\ttarget_name\n")
        
        # Write in batches for better performance
        batch_size = 10000
        batch_lines = []
        edges_written = 0
        
        for u, v in G.iterEdges():
            if u < num_users and v < num_users:
                # User-User edge (friendship)
                edge_type = "friendship"
                source_type = "User"
                target_type = "User"
                source_name = user_attributes[u]['personaname']
                target_name = user_attributes[v]['personaname']
            else:
                # User-App edge (review)
                edge_type = "review"
                if u < num_users:  # u is user, v is app
                    user_id, app_id = u, v
                else:  # v is user, u is app
                    user_id, app_id = v, u
                
                source_type = "User"
                target_type = "App"
                source_name = user_attributes[user_id]['personaname']
                target_name = app_attributes[app_id]['name']
                u, v = user_id, app_id  # Ensure consistent ordering
            
            # Add to batch
            batch_lines.append(f"{u}\t{v}\t{source_type}\t{target_type}\t{edge_type}\t{source_name}\t{target_name}\n")
            edges_written += 1
            
            # Write batch when full
            if len(batch_lines) >= batch_size:
                f.writelines(batch_lines)
                batch_lines = []
                if edges_written % 100000 == 0:
                    print(f"  Written {edges_written:,}/{total_edges:,} edges ({edges_written/total_edges*100:.1f}%)")
        
        # Write remaining lines
        if batch_lines:
            f.writelines(batch_lines)
    
    print(f"Edge list saved to {edge_list_file}")
    
    # Save all node attributes
    all_attributes = {**user_attributes, **app_attributes}
    attributes_file = f"{output_prefix}_nodes.json"
    print(f"Saving {len(all_attributes):,} node attributes to {attributes_file}...")
    
    with open(attributes_file, 'w', encoding='utf-8') as f:
        json.dump(all_attributes, f, indent=2, ensure_ascii=False)
    print(f"Node attributes saved to {attributes_file}")

def update_recommender_graph_nodes(user_attributes, app_attributes, output_file="outputs/recommender_graph_nodes.json"):
    """Update the recommender graph nodes file with sentiment-enhanced data."""
    print(f"\nUpdating recommender graph nodes file...")
    
    # --- Start Debug ---
    # (Previous debug prints can remain or be commented out if too verbose)
    # print("DEBUG: Sample of user_attributes before merging:")
    # ...
    # print("DEBUG: Sample of app_attributes before merging:")
    # ...
    # --- End Debug ---

    # Combine all attributes (users and apps)
    all_attributes = {**user_attributes, **app_attributes}
    
    # Save updated nodes file (TRYING WITHOUT INDENT FIRST)
    print(f"Attempting to save {output_file} WITHOUT indentation...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_attributes, f, ensure_ascii=False) # NO INDENT
        print(f"Successfully saved {output_file} without indentation.")
        # If successful, you might want to stop here or proceed to test if colab.py can read this.
    except Exception as e_no_indent:
        print(f"❌ FAILED to save {output_file} even WITHOUT indentation: {e_no_indent}")
        print(f"This suggests a deeper issue with the data structure itself.")
        # Optionally, re-try with indent to get the original error for comparison if needed for other debugging
        # print(f"Trying again WITH indent to see original error location...")
        # with open(output_file, 'w', encoding='utf-8') as f:
        #     json.dump(all_attributes, f, indent=2, ensure_ascii=False)
        raise # Re-raise the error if saving without indent fails
    
    # If saving without indent worked, we can optionally try with indent
    # to confirm if indent was the sole issue, but the primary goal is a usable file.
    # For now, let's assume if no-indent works, that's the priority.

    # print(f"Updated recommender graph nodes saved to {output_file}") # Original print
    print(f"Total nodes: {len(all_attributes)} (Users: {len(user_attributes)}, Apps: {len(app_attributes)})")

def update_recommender_graph_edges(G, user_attributes, app_attributes, edge_sentiment_scores, output_file="outputs/recommender_graph_edges.txt"):
    """Update the recommender graph edges file with playtime weights and sentiment scores - OPTIMIZED VERSION."""
    print(f"\nUpdating recommender graph edges file...")
    
    num_users = len(user_attributes)
    total_edges = G.numberOfEdges()
    print(f"Writing {total_edges:,} edges with weights and sentiment scores to {output_file}...")
    
    # Save edge list with weights and sentiment scores - OPTIMIZED
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("source_id\ttarget_id\tsource_type\ttarget_type\tedge_type\tweight\tsentiment_score\tsource_name\ttarget_name\n")
        
        # Write in batches for better performance with large graphs
        batch_size = 10000
        batch_lines = []
        edges_written = 0
        
        for u, v in G.iterEdges():
            weight = G.weight(u, v)
            
            if u < num_users and v < num_users:
                # User-User edge (friendship)
                edge_type = "friendship"
                source_type = "User"
                target_type = "User"
                source_name = user_attributes[u]['personaname']
                target_name = user_attributes[v]['personaname']
                sentiment_score = ""  # No sentiment for friendships
            else:
                # User-App edge (review)
                edge_type = "review"
                if u < num_users:  # u is user, v is app
                    user_id, app_id = u, v
                else:  # v is user, u is app
                    user_id, app_id = v, u
                
                source_type = "User"
                target_type = "App"
                source_name = user_attributes[user_id]['personaname']
                target_name = app_attributes[app_id]['name']
                u, v = user_id, app_id  # Ensure consistent ordering
                
                # Get sentiment score for this edge
                sentiment_score = edge_sentiment_scores.get((user_id, app_id), "")
                if sentiment_score == "":
                    sentiment_score = ""  # Keep empty for missing scores
            
            # Add to batch
            batch_lines.append(f"{u}\t{v}\t{source_type}\t{target_type}\t{edge_type}\t{weight:.6f}\t{sentiment_score}\t{source_name}\t{target_name}\n")
            edges_written += 1
            
            # Write batch when full and show progress
            if len(batch_lines) >= batch_size:
                f.writelines(batch_lines)
                batch_lines = []
                if edges_written % 250000 == 0:  # Progress every 250K edges
                    print(f"  Progress: {edges_written:,}/{total_edges:,} edges ({edges_written/total_edges*100:.1f}%)")
        
        # Write remaining lines
        if batch_lines:
            f.writelines(batch_lines)
    
    print(f"Updated recommender graph edges saved to {output_file}")
    
    # Print statistics about sentiment scores in output - OPTIMIZED
    total_review_edges = sum(1 for score in edge_sentiment_scores.keys())  # More efficient counting
    sentiment_edges_count = sum(1 for score in edge_sentiment_scores.values() if score is not None)
    print(f"Edge sentiment statistics:")
    print(f"  Total review edges: {total_review_edges}")
    print(f"  Review edges with sentiment scores: {sentiment_edges_count}")
    print(f"  Review edges without sentiment scores: {total_review_edges - sentiment_edges_count}")

def extract_largest_connected_component(G, user_attributes, app_attributes, user_original_to_nk_id, app_original_to_nk_id, edge_sentiment_scores):
    """
    Extract the largest connected component from the User-User subgraph and all associated data.
    
    This function specifically analyzes connectivity based on the User-User friendship network
    loaded from exports/friends.json to ensure we get a cohesive social network for recommendations.
    
    Returns:
        tuple: (filtered_graph, filtered_user_attrs, filtered_app_attrs, user_mapping, app_mapping, filtered_sentiment_scores)
    """
    print("\n=== Extracting Largest Connected Component ===")
    print("IMPORTANT: Connectivity analysis is based on User-User friendships from exports/friends.json")
    
    num_users = len(user_attributes)
    
    # Create User-User subgraph to find connected components
    # This contains ONLY friendship edges from exports/friends.json
    user_graph = nk.Graph(num_users, weighted=False, directed=False)
    
    friendship_edges_count = 0
    for u, v in G.iterEdges():
        if u < num_users and v < num_users:  # Both are users (friendship edge)
            if not user_graph.hasEdge(u, v):
                user_graph.addEdge(u, v)
                friendship_edges_count += 1
    
    print(f"User-User friendship subgraph extracted from exports/friends.json:")
    print(f"  Nodes: {user_graph.numberOfNodes()} users")
    print(f"  Edges: {user_graph.numberOfEdges()} friendships")
    print(f"  (Verified: {friendship_edges_count} friendship edges processed)")
    
    # Find connected components in the friendship network
    cc = nk.components.ConnectedComponents(user_graph)
    cc.run()
    
    # Get the largest component
    component_sizes_dict = cc.getComponentSizes()
    largest_component_id = max(component_sizes_dict.keys(), key=lambda x: component_sizes_dict[x])
    largest_component_size = component_sizes_dict[largest_component_id]
    
    print(f"\nConnectivity analysis results:")
    print(f"  Total friendship-based components: {cc.numberOfComponents()}")
    print(f"  Largest component size: {largest_component_size:,} users ({largest_component_size/num_users*100:.1f}% of all users)")
    
    # Additional statistics about component sizes
    component_sizes = sorted(component_sizes_dict.values(), reverse=True)
    if len(component_sizes) > 1:
        print(f"  Second largest component: {component_sizes[1]:,} users")
        print(f"  Components with 1 user (isolated): {sum(1 for size in component_sizes if size == 1)}")
        print(f"  Components with 2-10 users: {sum(1 for size in component_sizes if 2 <= size <= 10)}")
        print(f"  Components with >10 users: {sum(1 for size in component_sizes if size > 10)}")
    
    # Get users in the largest component
    users_in_largest = set()
    for user_id in range(num_users):
        if user_id in user_attributes and cc.componentOfNode(user_id) == largest_component_id:
            users_in_largest.add(user_id)
    
    print(f"\nUsers selected for final graph: {len(users_in_largest):,}")
    
    # Find apps reviewed by users in the largest component
    apps_in_largest = set()
    review_edges_in_largest = []
    
    for u, v in G.iterEdges():
        if u < num_users and v >= num_users:  # User-App edge
            user_id, app_id = u, v
        elif v < num_users and u >= num_users:  # App-User edge  
            user_id, app_id = v, u
        else:
            continue  # Not a User-App edge
        
        if user_id in users_in_largest:
            apps_in_largest.add(app_id)
            review_edges_in_largest.append((user_id, app_id))
    
    print(f"Apps reviewed by largest component users: {len(apps_in_largest)}")
    print(f"Review edges in largest component: {len(review_edges_in_largest)}")
    
    # Create new mappings for the filtered graph
    # Users: map from largest component users to new consecutive IDs 0, 1, 2, ...
    # Apps: map from apps in largest component to new consecutive IDs starting after users
    
    old_to_new_user = {}
    new_to_old_user = {}
    filtered_user_attrs = {}
    
    new_user_id = 0
    for old_user_id in sorted(users_in_largest):
        old_to_new_user[old_user_id] = new_user_id
        new_to_old_user[new_user_id] = old_user_id
        filtered_user_attrs[new_user_id] = user_attributes[old_user_id].copy()
        new_user_id += 1
    
    old_to_new_app = {}
    new_to_old_app = {}
    filtered_app_attrs = {}
    
    new_app_id = len(old_to_new_user)  # Start after users
    for old_app_id in sorted(apps_in_largest):
        old_to_new_app[old_app_id] = new_app_id
        new_to_old_app[new_app_id] = old_app_id
        filtered_app_attrs[new_app_id] = app_attributes[old_app_id].copy()
        new_app_id += 1
    
    # Create the filtered graph
    total_filtered_nodes = len(old_to_new_user) + len(old_to_new_app)
    filtered_G = nk.Graph(total_filtered_nodes, weighted=True, directed=False)
    
    print(f"\nCreating filtered graph with {total_filtered_nodes} nodes:")
    print(f"  - Users: {len(old_to_new_user)} (IDs 0-{len(old_to_new_user)-1})")
    print(f"  - Apps: {len(old_to_new_app)} (IDs {len(old_to_new_user)}-{total_filtered_nodes-1})")
    
    # Add friendship edges (only within the largest component)
    friendship_edges_added = 0
    for u, v in G.iterEdges():
        if (u < num_users and v < num_users and 
            u in users_in_largest and v in users_in_largest):
            
            new_u = old_to_new_user[u]
            new_v = old_to_new_user[v]
            weight = G.weight(u, v)
            
            if not filtered_G.hasEdge(new_u, new_v):
                filtered_G.addEdge(new_u, new_v, weight)
                friendship_edges_added += 1
    
    # Add review edges (only for users and apps in the largest component)
    review_edges_added = 0
    filtered_sentiment_scores = {}
    
    for old_user_id, old_app_id in review_edges_in_largest:
        new_user_id = old_to_new_user[old_user_id]
        new_app_id = old_to_new_app[old_app_id]
        weight = G.weight(old_user_id, old_app_id)
        
        if not filtered_G.hasEdge(new_user_id, new_app_id):
            filtered_G.addEdge(new_user_id, new_app_id, weight)
            review_edges_added += 1
            
            # Transfer sentiment score if available
            if (old_user_id, old_app_id) in edge_sentiment_scores:
                filtered_sentiment_scores[(new_user_id, new_app_id)] = edge_sentiment_scores[(old_user_id, old_app_id)]
    
    print(f"\nFinal graph construction:")
    print(f"  Added {friendship_edges_added} friendship edges from largest component")
    print(f"  Added {review_edges_added} review edges from largest component users")
    print(f"  Transferred {len(filtered_sentiment_scores)} sentiment scores")
    
    # Create new original ID mappings for the filtered graph - OPTIMIZED VERSION
    print(f"Creating optimized ID mappings...")
    filtered_user_original_to_nk_id = {}
    filtered_app_original_to_nk_id = {}
    
    # Create reverse mappings for O(1) lookup instead of O(N) nested loops
    user_nk_to_original = {nk_id: orig_id for orig_id, nk_id in user_original_to_nk_id.items()}
    app_nk_to_original = {nk_id: orig_id for orig_id, nk_id in app_original_to_nk_id.items()}
    
    # Map users - now O(N) instead of O(N²)
    for new_id, old_id in new_to_old_user.items():
        original_id = user_nk_to_original.get(old_id)
        if original_id is not None:
            filtered_user_original_to_nk_id[original_id] = new_id
    
    # Map apps - now O(N) instead of O(N²)
    for new_id, old_id in new_to_old_app.items():
        original_id = app_nk_to_original.get(old_id)
        if original_id is not None:
            filtered_app_original_to_nk_id[original_id] = new_id
    
    print(f"ID mappings created: {len(filtered_user_original_to_nk_id)} users, {len(filtered_app_original_to_nk_id)} apps")
    
    print(f"\nFinal filtered graph statistics:")
    print(f"  Total nodes: {filtered_G.numberOfNodes()}")
    print(f"  Total edges: {filtered_G.numberOfEdges()}")
    print(f"  Graph density: {filtered_G.numberOfEdges() / (filtered_G.numberOfNodes() * (filtered_G.numberOfNodes() - 1) / 2) * 100:.4f}%")
    print(f"  ✓ Connected via friendship network from exports/friends.json")
    
    return (filtered_G, filtered_user_attrs, filtered_app_attrs, 
            filtered_user_original_to_nk_id, filtered_app_original_to_nk_id, 
            filtered_sentiment_scores)

def main():
    """Main function to construct and analyze the combined graph."""
    friends_file = "exports/friends.json"
    reviews_file = "exports/user_app_review.json"
    sentiment_scores_file = "outputs/steam_review_sentiment_scores.json"
    
    print("Starting combined graph construction...")
    print("This will merge friendship and review data into a single heterogeneous graph.")
    print("Using complete user_app_review.json dataset with sentiment scores...")
    print("Creating FINAL graph focused on the largest connected component...")
    print(f"✓ Friendship data source: {friends_file}")
    print(f"✓ Review data source: {reviews_file}")
    print(f"✓ Sentiment scores source: {sentiment_scores_file}")
    print("✓ Connectivity analysis will be based on User-User friendships from exports/friends.json")
    
    try:
        # Construct full combined graph with sentiment scores
        G, user_attrs, app_attrs, user_mapping, app_mapping, edge_sentiment_scores = construct_combined_graph(
            friends_file, reviews_file, sentiment_scores_file
        )
        
        # Analyze full combined graph
        print("\n" + "="*60)
        print("FULL GRAPH ANALYSIS")
        print("="*60)
        analyze_combined_graph(G, user_attrs, app_attrs, skip_expensive=True)
        analyze_edge_weights(G, user_attrs, app_attrs, skip_expensive=True)
        
        # Extract the largest connected component
        print("\n" + "="*60)
        print("CREATING FINAL FILTERED GRAPH")
        print("="*60)
        
        (filtered_G, filtered_user_attrs, filtered_app_attrs, 
         filtered_user_mapping, filtered_app_mapping, 
         filtered_sentiment_scores) = extract_largest_connected_component(
            G, user_attrs, app_attrs, user_mapping, app_mapping, edge_sentiment_scores
        )
        
        # Analyze the filtered graph (detailed analysis on smaller graph)
        print("\n" + "="*60)
        print("FILTERED GRAPH ANALYSIS")
        print("="*60)
        analyze_combined_graph(filtered_G, filtered_user_attrs, filtered_app_attrs, skip_expensive=False)
        analyze_edge_weights(filtered_G, filtered_user_attrs, filtered_app_attrs, skip_expensive=False)
        
        # Save the filtered graph data
        print("\n" + "="*60)
        print("SAVING FINAL GRAPH DATA")
        print("="*60)
        
        # Skip the intermediate combined graph file (saves time) - only save essentials
        print("⚠️  Skipping intermediate edge file generation to save time")
        print("  (Only saving essential recommender graph files)")
        
        # Save the final filtered graph - SKIP THIS (not essential)
        # save_combined_graph_data(filtered_G, filtered_user_attrs, filtered_app_attrs, "final_largest_component")
        
        # Update recommender graph files with filtered data - ESSENTIAL
        update_recommender_graph_nodes(filtered_user_attrs, filtered_app_attrs, "outputs/final_recommender_graph_nodes.json")
        update_recommender_graph_edges(filtered_G, filtered_user_attrs, filtered_app_attrs, filtered_sentiment_scores, "outputs/final_recommender_graph_edges.txt")
        
        # Save mapping files for reference
        print("\nSaving ID mapping files...")
        
        # Save user mapping
        user_mapping_file = "outputs/final_user_id_mapping.json"
        with open(user_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_user_mapping, f, indent=2)
        print(f"User ID mapping saved to {user_mapping_file}")
        
        # Save app mapping  
        app_mapping_file = "outputs/final_app_id_mapping.json"
        with open(app_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_app_mapping, f, indent=2)
        print(f"App ID mapping saved to {app_mapping_file}")
        
        # Save summary statistics
        summary_stats = {
            "data_sources": {
                "friendships": friends_file,
                "reviews": reviews_file,
                "sentiment_scores": sentiment_scores_file
            },
            "methodology": {
                "connectivity_basis": "User-User friendship network from exports/friends.json",
                "component_selection": "Largest connected component",
                "edge_weights": "Playtime-based for reviews, unweighted for friendships",
                "sentiment_preservation": "Stored separately for recommender system"
            },
            "original_graph": {
                "total_nodes": G.numberOfNodes(),
                "total_edges": G.numberOfEdges(),
                "users": len(user_attrs),
                "apps": len(app_attrs)
            },
            "final_graph": {
                "total_nodes": filtered_G.numberOfNodes(),
                "total_edges": filtered_G.numberOfEdges(),
                "users": len(filtered_user_attrs),
                "apps": len(filtered_app_attrs),
                "coverage_users": len(filtered_user_attrs) / len(user_attrs) * 100,
                "coverage_apps": len(filtered_app_attrs) / len(app_attrs) * 100
            }
        }
        
        summary_file = "outputs/final_graph_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"Summary statistics saved to {summary_file}")
        
        print("\n" + "="*60)
        print("FINAL GRAPH CONSTRUCTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"✓ Connectivity analysis based on friendship network from {friends_file}")
        print(f"Final graph contains:")
        print(f"  - {len(filtered_user_attrs):,} users ({len(filtered_user_attrs)/len(user_attrs)*100:.1f}% of original)")
        print(f"  - {len(filtered_app_attrs)} apps ({len(filtered_app_attrs)/len(app_attrs)*100:.1f}% of original)")
        print(f"  - {filtered_G.numberOfEdges():,} edges ({filtered_G.numberOfEdges()/G.numberOfEdges()*100:.1f}% of original)")
        print(f"\nFiles created:")
        print(f"  - outputs/final_recommender_graph_nodes.json")
        print(f"  - outputs/final_recommender_graph_edges.txt")
        print(f"  - outputs/final_user_id_mapping.json")
        print(f"  - outputs/final_app_id_mapping.json")
        print(f"  - outputs/final_graph_summary.json")
        
        return (filtered_G, filtered_user_attrs, filtered_app_attrs, 
                filtered_user_mapping, filtered_app_mapping, filtered_sentiment_scores)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required file - {e}")
        print("Please make sure all files exist in their respective directories")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    (filtered_G, filtered_user_attrs, filtered_app_attrs, 
     filtered_user_mapping, filtered_app_mapping, filtered_sentiment_scores) = main() 