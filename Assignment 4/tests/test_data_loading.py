from steam_recommender_node2vec_gnn import DataLoader

print("Testing fixed data loading...")
loader = DataLoader()
loader.load_data()

print(f"\n📊 Data Loading Results:")
print(f"  Friendships loaded: {len(loader.user_friendships):,}")
print(f"  Reviews loaded: {len(loader.user_app_reviews):,}")
print(f"  Total edges: {len(loader.user_friendships) + len(loader.user_app_reviews):,}")
print(f"  Users: {len(loader.users):,}")
print(f"  Apps: {len(loader.apps):,}")

# Show sample edges
if loader.user_friendships:
    print(f"\n🤝 Sample friendships:")
    for i, (u, v, w) in enumerate(loader.user_friendships[:3]):
        print(f"  {i+1}. User {u} -> User {v} (weight: {w})")

if loader.user_app_reviews:
    print(f"\n🎮 Sample reviews:")
    for i, (u, a, w, s) in enumerate(loader.user_app_reviews[:3]):
        print(f"  {i+1}. User {u} -> App {a} (weight: {w}, sentiment: {s})") 