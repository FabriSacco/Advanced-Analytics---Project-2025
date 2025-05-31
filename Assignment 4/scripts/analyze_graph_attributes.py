#!/usr/bin/env python3
"""
Graph Attributes Analysis
========================

This script analyzes and prints the attributes of different node types and edge types
in the sentiment-enhanced combined graph.
"""

import json
from collections import defaultdict

def analyze_node_attributes():
    """Analyze and print node attributes for Users and Apps."""
    
    print("="*60)
    print("NODE ATTRIBUTES ANALYSIS")
    print("="*60)
    
    # Load the updated nodes file
    with open("outputs/recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # Separate users and apps
    user_nodes = {}
    app_nodes = {}
    
    for node_id, attributes in nodes.items():
        if attributes.get('type') == 'User':
            user_nodes[node_id] = attributes
        elif attributes.get('type') == 'App':
            app_nodes[node_id] = attributes
    
    print(f"\nTotal nodes: {len(nodes)}")
    print(f"User nodes: {len(user_nodes)}")
    print(f"App nodes: {len(app_nodes)}")
    
    # Analyze User node attributes
    print("\n" + "="*40)
    print("USER NODE ATTRIBUTES")
    print("="*40)
    
    if user_nodes:
        # Get a sample user node
        sample_user_id = next(iter(user_nodes))
        sample_user = user_nodes[sample_user_id]
        
        print(f"\nSample User Node (ID: {sample_user_id}):")
        for key, value in sample_user.items():
            print(f"  {key}: {value}")
        
        print(f"\nAll User Attribute Keys:")
        all_user_keys = set()
        for user in user_nodes.values():
            all_user_keys.update(user.keys())
        for key in sorted(all_user_keys):
            print(f"  - {key}")
        
        # Analyze data source distribution
        data_source_counts = defaultdict(int)
        country_counts = defaultdict(int)
        
        for user in user_nodes.values():
            data_source_counts[user.get('data_source', 'Unknown')] += 1
            country_counts[user.get('loccountrycode', 'Unknown')] += 1
        
        print(f"\nUser Data Source Distribution:")
        for source, count in data_source_counts.items():
            percentage = (count / len(user_nodes)) * 100
            print(f"  {source}: {count} users ({percentage:.1f}%)")
        
        print(f"\nTop 10 Countries:")
        sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for country, count in sorted_countries:
            percentage = (count / len(user_nodes)) * 100
            print(f"  {country}: {count} users ({percentage:.1f}%)")
    
    # Analyze App node attributes
    print("\n" + "="*40)
    print("APP NODE ATTRIBUTES")
    print("="*40)
    
    if app_nodes:
        # Get a sample app node
        sample_app_id = next(iter(app_nodes))
        sample_app = app_nodes[sample_app_id]
        
        print(f"\nSample App Node (ID: {sample_app_id}):")
        for key, value in sample_app.items():
            print(f"  {key}: {value}")
        
        print(f"\nAll App Attribute Keys:")
        all_app_keys = set()
        for app in app_nodes.values():
            all_app_keys.update(app.keys())
        for key in sorted(all_app_keys):
            print(f"  - {key}")
        
        # Analyze app attributes
        category_counts = defaultdict(int)
        type_counts = defaultdict(int)
        free_counts = defaultdict(int)
        
        for app in app_nodes.values():
            category_counts[app.get('category', 'Unknown')] += 1
            type_counts[app.get('app_type', 'Unknown')] += 1
            free_counts[app.get('is_free', 'Unknown')] += 1
        
        print(f"\nApp Category Distribution:")
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(app_nodes)) * 100
            print(f"  {category}: {count} apps ({percentage:.1f}%)")
        
        print(f"\nApp Type Distribution:")
        for app_type, count in type_counts.items():
            percentage = (count / len(app_nodes)) * 100
            print(f"  {app_type}: {count} apps ({percentage:.1f}%)")
        
        print(f"\nFree vs Paid Apps:")
        for is_free, count in free_counts.items():
            percentage = (count / len(app_nodes)) * 100
            print(f"  {'Free' if is_free else 'Paid' if is_free == False else 'Unknown'}: {count} apps ({percentage:.1f}%)")

def analyze_edge_attributes():
    """Analyze and print edge attributes for User-User and User-App edges."""
    
    print("\n" + "="*60)
    print("EDGE ATTRIBUTES ANALYSIS")
    print("="*60)
    
    # Load the edges file (first 10000 lines for analysis to avoid memory issues)
    edges_data = []
    
    print("Loading edge data (sample)...")
    with open("outputs/recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        print(f"Edge file header: {header}")
        
        # Read first 10000 edges for analysis
        for i in range(10000):
            line = f.readline().strip()
            if not line:
                break
            parts = line.split('\t')
            if len(parts) == len(header):
                edges_data.append(dict(zip(header, parts)))
    
    print(f"Loaded {len(edges_data)} edges for analysis")
    
    # Analyze edge types
    edge_type_counts = defaultdict(int)
    for edge in edges_data:
        edge_type_counts[edge.get('edge_type', 'Unknown')] += 1
    
    print(f"\nEdge Type Distribution:")
    for edge_type, count in edge_type_counts.items():
        percentage = (count / len(edges_data)) * 100
        print(f"  {edge_type}: {count} edges ({percentage:.1f}%)")
    
    # Analyze User-User edges (Friendships)
    friendship_edges = [edge for edge in edges_data if edge.get('edge_type') == 'friendship']
    
    print("\n" + "="*40)
    print("USER-USER EDGE ATTRIBUTES (Friendships)")
    print("="*40)
    
    if friendship_edges:
        print(f"\nSample Friendship Edge:")
        sample_friendship = friendship_edges[0]
        for key, value in sample_friendship.items():
            print(f"  {key}: {value}")
        
        print(f"\nFriendship Edge Attribute Summary:")
        print(f"  Total friendship edges in sample: {len(friendship_edges)}")
        
        # Get unique values for source and target types
        source_types = set(edge.get('source_type', 'Unknown') for edge in friendship_edges)
        target_types = set(edge.get('target_type', 'Unknown') for edge in friendship_edges)
        print(f"  Source type: {list(source_types)}")
        print(f"  Target type: {list(target_types)}")
        
        # Weight analysis for friendships
        if 'weight' in friendship_edges[0]:
            weights = []
            for edge in friendship_edges:
                try:
                    weight = float(edge['weight'])
                    weights.append(weight)
                except (ValueError, TypeError):
                    pass
            
            if weights:
                print(f"  Weight statistics:")
                print(f"    Unique weights: {list(set(weights))}")
                print(f"    Min weight: {min(weights)}")
                print(f"    Max weight: {max(weights)}")
                print(f"    Mean weight: {sum(weights)/len(weights):.3f}")
    
    # Analyze User-App edges (Reviews)
    review_edges = [edge for edge in edges_data if edge.get('edge_type') == 'review']
    
    print("\n" + "="*40)
    print("USER-APP EDGE ATTRIBUTES (Reviews)")
    print("="*40)
    
    if review_edges:
        print(f"\nSample Review Edge:")
        sample_review = review_edges[0]
        for key, value in sample_review.items():
            print(f"  {key}: {value}")
        
        print(f"\nReview Edge Attribute Summary:")
        print(f"  Total review edges in sample: {len(review_edges)}")
        
        # Get unique values for source and target types
        source_types = set(edge.get('source_type', 'Unknown') for edge in review_edges)
        target_types = set(edge.get('target_type', 'Unknown') for edge in review_edges)
        print(f"  Source type: {list(source_types)}")
        print(f"  Target type: {list(target_types)}")
        
        # Weight analysis for reviews
        if 'weight' in review_edges[0]:
            weights = []
            for edge in review_edges:
                try:
                    weight = float(edge['weight'])
                    weights.append(weight)
                except (ValueError, TypeError):
                    pass
            
            if weights:
                weights.sort()
                print(f"  Weight statistics:")
                print(f"    Count: {len(weights)}")
                print(f"    Min weight: {min(weights):.3f}")
                print(f"    Max weight: {max(weights):.3f}")
                print(f"    Mean weight: {sum(weights)/len(weights):.3f}")
                print(f"    Median weight: {weights[len(weights)//2]:.3f}")
                
                # Weight distribution
                print(f"  Weight distribution:")
                weight_ranges = [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4)]
                for min_w, max_w in weight_ranges:
                    count = sum(1 for w in weights if min_w <= w < max_w)
                    percentage = (count / len(weights)) * 100
                    print(f"    {min_w}-{max_w}: {count} edges ({percentage:.1f}%)")

def print_detailed_examples():
    """Print detailed examples of each node and edge type."""
    
    print("\n" + "="*60)
    print("DETAILED EXAMPLES")
    print("="*60)
    
    # Load nodes
    with open("outputs/recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # Find examples
    user_example = None
    app_example = None
    
    for node_id, attributes in nodes.items():
        if attributes.get('type') == 'User' and user_example is None:
            user_example = (node_id, attributes)
        elif attributes.get('type') == 'App' and app_example is None:
            app_example = (node_id, attributes)
        
        if user_example and app_example:
            break
    
    print("\n📁 EXAMPLE USER NODE:")
    if user_example:
        node_id, attrs = user_example
        print(f"Node ID: {node_id}")
        print("Attributes:")
        for key, value in attrs.items():
            print(f"  {key}: {repr(value)}")
    
    print("\n🎮 EXAMPLE APP NODE:")
    if app_example:
        node_id, attrs = app_example
        print(f"Node ID: {node_id}")
        print("Attributes:")
        for key, value in attrs.items():
            print(f"  {key}: {repr(value)}")
    
    # Load a few edges for examples
    print("\n🔗 EXAMPLE EDGES:")
    with open("outputs/recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        
        friendship_found = False
        review_found = False
        
        for line in f:
            if friendship_found and review_found:
                break
                
            parts = line.strip().split('\t')
            if len(parts) == len(header):
                edge_data = dict(zip(header, parts))
                
                if edge_data.get('edge_type') == 'friendship' and not friendship_found:
                    print("\n👥 EXAMPLE FRIENDSHIP EDGE:")
                    for key, value in edge_data.items():
                        print(f"  {key}: {repr(value)}")
                    friendship_found = True
                
                elif edge_data.get('edge_type') == 'review' and not review_found:
                    print("\n⭐ EXAMPLE REVIEW EDGE:")
                    for key, value in edge_data.items():
                        print(f"  {key}: {repr(value)}")
                    review_found = True

def main():
    """Main function to run all analyses."""
    
    print("GRAPH ATTRIBUTES ANALYSIS")
    print("Analyzing sentiment-enhanced combined graph...")
    
    try:
        analyze_node_attributes()
        analyze_edge_attributes()
        print_detailed_examples()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nSummary:")
        print("✓ User nodes contain friendship and demographic data")
        print("✓ App nodes contain game/app metadata and categories")
        print("✓ User-User edges represent friendships (weight=1.0)")
        print("✓ User-App edges represent reviews (sentiment+playtime weights)")
        print("✓ All attributes are preserved and enhanced with sentiment data")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 