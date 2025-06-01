import json
import networkx as nx

# Quick connectivity check
print('Loading nodes...')
with open('outputs/final_recommender_graph_nodes.json', 'r', encoding='utf-8') as f:
    all_nodes = json.load(f)

users = {nid: data for nid, data in all_nodes.items() if data['type'] == 'User'}
print(f'Users: {len(users):,}')

# Load friendship edges
print('Loading friendships...')
friendships = []
with open('outputs/final_recommender_graph_edges.txt', 'r', encoding='utf-8') as f:
    next(f)  # Skip header
    count = 0
    for line in f:
        count += 1
        if count % 500000 == 0:
            print(f'  Processed {count:,} lines...')
        parts = line.strip().split('\t')
        if len(parts) >= 6 and parts[4] == 'friendship':
            friendships.append((parts[0], parts[1]))

print(f'Friendships: {len(friendships):,}')

# Build graph
print('Building graph...')
G = nx.Graph()
G.add_nodes_from(users.keys())
G.add_edges_from(friendships)

print(f'Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges')
print(f'Is connected: {nx.is_connected(G)}')

if not nx.is_connected(G):
    components = list(nx.connected_components(G))
    sizes = [len(comp) for comp in sorted(components, key=len, reverse=True)]
    print(f'Components: {len(components)}')
    print(f'Top 5 component sizes: {sizes[:5]}')
    print(f'Largest CC: {sizes[0]:,} users ({sizes[0]/len(users)*100:.1f}%)')
else:
    print('✅ Network is fully connected!') 