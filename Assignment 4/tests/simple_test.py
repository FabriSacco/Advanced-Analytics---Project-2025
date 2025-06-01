import json
import sys
import traceback

def test_json():
    print("Testing JSON loading...")
    try:
        with open("outputs/final_recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Success! Loaded {len(data)} nodes")
        return True
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

def test_edges():
    print("Testing edge loading...")
    try:
        count = 0
        with open("outputs/final_recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
            header = f.readline()
            print(f"Header: {header.strip()}")
            
            for i, line in enumerate(f):
                if i > 100:  # Just test first 100 lines
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    count += 1
        
        print(f"Success! Processed {count} edges from first 100 lines")
        return True
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Running simple data tests...")
    
    json_ok = test_json()
    edge_ok = test_edges()
    
    print(f"\nResults:")
    print(f"JSON: {'OK' if json_ok else 'FAILED'}")
    print(f"Edges: {'OK' if edge_ok else 'FAILED'}") 