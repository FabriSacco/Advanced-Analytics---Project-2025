"""
Test script to isolate data loading issues
"""
import json
import pandas as pd
from tqdm import tqdm
import os

def test_json_loading():
    """Test JSON file loading"""
    print("🔍 Testing JSON file loading...")
    
    try:
        with open("outputs/final_recommender_graph_nodes.json", 'r') as f:
            data = json.load(f)
        
        print(f"✅ JSON loaded successfully!")
        print(f"📊 Total nodes: {len(data):,}")
        
        # Count users and apps
        users = 0
        apps = 0
        for node_id, node_data in data.items():
            if node_data['type'] == 'User':
                users += 1
            else:
                apps += 1
        
        print(f"👥 Users: {users:,}")
        print(f"🎮 Apps: {apps:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False

def test_edge_loading():
    """Test edge file loading"""
    print("\n🔍 Testing edge file loading...")
    
    try:
        friendships = 0
        reviews = 0
        
        with open("outputs/final_recommender_graph_edges.txt", 'r') as f:
            header = f.readline().strip()
            print(f"📄 Header: {header}")
            
            # Process first 1000 lines to test
            for i, line in enumerate(f):
                if i >= 1000:  # Only test first 1000 lines
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    edge_type = parts[4]
                    if edge_type == 'friendship':
                        friendships += 1
                    elif edge_type == 'review':
                        reviews += 1
        
        print(f"✅ Edge parsing test successful!")
        print(f"🤝 Friendships (first 1000): {friendships}")
        print(f"📝 Reviews (first 1000): {reviews}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading edges: {e}")
        return False

def test_full_edge_loading():
    """Test full edge file loading with progress"""
    print("\n🔍 Testing FULL edge file loading...")
    
    try:
        friendships = 0
        reviews = 0
        errors = 0
        
        # Count total lines first
        print("📊 Counting total lines...")
        with open("outputs/final_recommender_graph_edges.txt", 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
        
        print(f"📊 Total data lines: {total_lines:,}")
        
        with open("outputs/final_recommender_graph_edges.txt", 'r') as f:
            header = f.readline().strip()
            print(f"📄 Header: {header}")
            
            # Process all lines with progress bar
            for line in tqdm(f, total=total_lines, desc="Processing edges"):
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 6:
                        edge_type = parts[4]
                        if edge_type == 'friendship':
                            friendships += 1
                        elif edge_type == 'review':
                            reviews += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Show first 5 errors
                        print(f"⚠️ Error on line: {e}")
        
        print(f"✅ Full edge parsing completed!")
        print(f"🤝 Total friendships: {friendships:,}")
        print(f"📝 Total reviews: {reviews:,}")
        print(f"❌ Parse errors: {errors:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in full edge loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing data loading components...")
    
    # Test 1: JSON loading
    json_ok = test_json_loading()
    
    # Test 2: Edge loading (sample)
    edge_sample_ok = test_edge_loading()
    
    # Test 3: Full edge loading (if sample worked)
    if edge_sample_ok:
        full_edge_ok = test_full_edge_loading()
    else:
        full_edge_ok = False
    
    print(f"\n📋 Test Results:")
    print(f"  JSON Loading: {'✅' if json_ok else '❌'}")
    print(f"  Edge Sample: {'✅' if edge_sample_ok else '❌'}")
    print(f"  Full Edges: {'✅' if full_edge_ok else '❌'}")
    
    if all([json_ok, edge_sample_ok, full_edge_ok]):
        print("\n🎉 All tests passed! Data loading should work.")
    else:
        print("\n⚠️ Some tests failed. Check the errors above.") 