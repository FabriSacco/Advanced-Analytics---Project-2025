import json
import networkx as nx

class BasicDataLoader:
    def __init__(self, data_path="outputs/"):
        self.data_path = data_path
        self.users = {}
        self.apps = {}
        self.user_friendships = []
        self.user_app_reviews = []
        
    def load_data(self):
        """Load all graph data from files"""
        print("\n📂 Loading Steam graph data...")
        
        # Load nodes
        print("  Loading nodes...")
        with open(f"{self.data_path}final_recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
            all_nodes = json.load(f)
        
        # Separate users and apps
        for node_id, node_data in all_nodes.items():
            if node_data['type'] == 'User':
                self.users[node_id] = node_data
            else:
                self.apps[node_id] = node_data
        
        print(f"    Users: {len(self.users):,}")
        print(f"    Apps: {len(self.apps):,}")
        
        # Load edges
        print("  Loading edges...")
        
        # Debug: Show first few lines to understand format
        print("  Analyzing edge file format...")
        with open(f"{self.data_path}final_recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
            header = f.readline().strip()
            print(f"    Header: {header}")
            
            # Show first 3 data lines
            for i in range(3):
                line = f.readline().strip()
                if line:
                    parts = line.split('\t')
                    print(f"    Sample line {i+1}: {len(parts)} columns -> {parts[:6]}")
                    
        print("  Processing all edges...")
        edge_count = 0
        with open(f"{self.data_path}final_recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                edge_count += 1
                if edge_count % 100000 == 0:
                    print(f"    Processed {edge_count:,} lines...")
                    
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    try:
                        # Correct format: source_id, target_id, source_type, target_type, edge_type, weight, sentiment_score, source_name, target_name
                        source = parts[0]       # Source ID
                        target = parts[1]       # Target ID  
                        source_type = parts[2]  # Source type
                        target_type = parts[3]  # Target type
                        edge_type = parts[4]    # Edge type (friendship/review)
                        weight = float(parts[5]) # Weight
                        
                        if edge_type == 'friendship' and source_type == 'User' and target_type == 'User':
                            # Friendship edge between users
                            if source != target:  # Avoid self-loops
                                self.user_friendships.append((source, target, weight))
                                
                        elif edge_type == 'review' and source_type == 'User' and target_type == 'App':
                            # Review edge from user to app
                            sentiment = 0.0
                            if len(parts) > 6:  # Sentiment score in column 6
                                try:
                                    sentiment = float(parts[6])
                                except (ValueError, IndexError):
                                    sentiment = 0.0
                            
                            self.user_app_reviews.append((source, target, weight, sentiment))
                            
                    except (ValueError, IndexError) as e:
                        # Only show first few warnings to avoid spam
                        if len(self.user_friendships) + len(self.user_app_reviews) < 3:
                            print(f"    Warning: Skipping malformed line: {e}")
                        continue
        
        print(f"    Friendships: {len(self.user_friendships):,}")
        print(f"    Reviews: {len(self.user_app_reviews):,}")
        
        return self

# Test the data loading
print("Testing fixed data loading...")
loader = BasicDataLoader()
loader.load_data()

print(f"\n📊 Data Loading Results:")
print(f"  Friendships loaded: {len(loader.user_friendships):,}")
print(f"  Reviews loaded: {len(loader.user_app_reviews):,}")
print(f"  Total edges: {len(loader.user_friendships) + len(loader.user_app_reviews):,}")
print(f"  Users: {len(loader.users):,}")
print(f"  Apps: {len(loader.apps):,}")

# Test connectivity analysis
print(f"\n🔍 Testing Connectivity Analysis...")

# Build friendship graph
print("  Building friendship network...")
valid_user_ids = set(loader.users.keys())
filtered_friendships = [(u, v, w) for u, v, w in loader.user_friendships 
                       if u in valid_user_ids and v in valid_user_ids]

G = nx.Graph()
G.add_nodes_from(valid_user_ids)
friendship_edges = [(u, v) for u, v, w in filtered_friendships]
G.add_edges_from(friendship_edges)

print(f"    Network: {G.number_of_nodes():,} users, {G.number_of_edges():,} friendships")

# Check connectivity
print("  Analyzing connectivity...")
is_connected = nx.is_connected(G)
connected_components = list(nx.connected_components(G))

print(f"    Is fully connected: {is_connected}")
print(f"    Number of components: {len(connected_components)}")

if not is_connected:
    component_sizes = [len(comp) for comp in sorted(connected_components, key=len, reverse=True)]
    print(f"    Component sizes (top 10): {component_sizes[:10]}")
    largest_cc = max(connected_components, key=len)
    print(f"    Largest CC: {len(largest_cc):,} users ({len(largest_cc)/len(valid_user_ids)*100:.1f}%)")
else:
    print(f"    ✅ Entire network is connected!")

# Show sample edges
if loader.user_friendships:
    print(f"\n🤝 Sample friendships:")
    for i, (u, v, w) in enumerate(loader.user_friendships[:3]):
        print(f"  {i+1}. User {u} -> User {v} (weight: {w})")

if loader.user_app_reviews:
    print(f"\n🎮 Sample reviews:")
    for i, (u, a, w, s) in enumerate(loader.user_app_reviews[:3]):
        print(f"  {i+1}. User {u} -> App {a} (weight: {w}, sentiment: {s})") 