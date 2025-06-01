import json
import networkx as nx

class LCCTester:
    def __init__(self):
        self.users = {}
        self.apps = {}
        self.user_friendships = []
        self.user_app_reviews = []
        
    def load_sample_data(self):
        """Load a sample of the data for testing"""
        print("Loading sample data for LCC testing...")
        
        # Load nodes
        with open('outputs/final_recommender_graph_nodes.json', 'r', encoding='utf-8') as f:
            all_nodes = json.load(f)
        
        # Separate users and apps
        for node_id, node_data in all_nodes.items():
            if node_data['type'] == 'User':
                self.users[node_id] = node_data
            else:
                self.apps[node_id] = node_data
        
        print(f"  Users: {len(self.users):,}")
        print(f"  Apps: {len(self.apps):,}")
        
        # Load a sample of friendships (first 100k for testing)
        count = 0
        max_edges = 100000
        
        with open('outputs/final_recommender_graph_edges.txt', 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                if count >= max_edges:
                    break
                    
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    source = parts[0]
                    target = parts[1]
                    edge_type = parts[4]
                    weight = float(parts[5])
                    
                    if edge_type == 'friendship':
                        if source != target:
                            self.user_friendships.append((source, target, weight))
                    elif edge_type == 'review':
                        sentiment = 0.0
                        if len(parts) > 6:
                            try:
                                sentiment = float(parts[6])
                            except:
                                pass
                        self.user_app_reviews.append((source, target, weight, sentiment))
                
                count += 1
        
        print(f"  Sample friendships: {len(self.user_friendships):,}")
        print(f"  Sample reviews: {len(self.user_app_reviews):,}")
        
    def test_lcc_extraction(self):
        """Test the LCC extraction process"""
        print("\n🔗 Testing LCC extraction...")
        
        # Step 1: Filter friendships to valid users
        valid_user_ids = set(self.users.keys())
        filtered_friendships = [(u, v, w) for u, v, w in self.user_friendships 
                               if u in valid_user_ids and v in valid_user_ids]
        
        print(f"  Filtered friendships: {len(filtered_friendships):,}")
        
        # Step 2: Build graph
        G = nx.Graph()
        G.add_nodes_from(valid_user_ids)
        friendship_edges = [(u, v) for u, v, w in filtered_friendships]
        G.add_edges_from(friendship_edges)
        
        print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        
        # Step 3: Check connectivity
        is_connected = nx.is_connected(G)
        print(f"  Is connected: {is_connected}")
        
        if not is_connected:
            components = list(nx.connected_components(G))
            component_sizes = [len(comp) for comp in sorted(components, key=len, reverse=True)]
            print(f"  Components: {len(components)}")
            print(f"  Top 5 component sizes: {component_sizes[:5]}")
            
            # Extract LCC
            largest_cc = max(components, key=len)
            G_lcc = G.subgraph(largest_cc).copy()
            
            print(f"  Largest CC: {G_lcc.number_of_nodes():,} nodes, {G_lcc.number_of_edges():,} edges")
            print(f"  LCC is connected: {nx.is_connected(G_lcc)}")
            
            return G_lcc, largest_cc
        else:
            print("  ✅ Entire graph is connected!")
            return G, set(valid_user_ids)

# Run the test
tester = LCCTester()
tester.load_sample_data()
lcc_graph, lcc_users = tester.test_lcc_extraction()

print(f"\n📊 Final LCC Results:")
print(f"  LCC users: {len(lcc_users):,}")
print(f"  LCC is fully connected: {nx.is_connected(lcc_graph)}") 