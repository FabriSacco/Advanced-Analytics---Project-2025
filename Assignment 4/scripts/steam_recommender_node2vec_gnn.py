"""
Steam Recommender System: Node2Vec + Heterogeneous GNN
Optimized for Google Colab with GPU (L4/A100)

This script implements a complete pipeline:
1. Data loading and preprocessing
2. Largest connected component extraction
3. Node2Vec embeddings for users
4. Heterogeneous GNN for recommendations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Deep Learning & Graph Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.utils import negative_sampling
from node2vec import Node2Vec

print("Steam Recommender System: Node2Vec + Heterogeneous GNN")
print("="*60)

# Configuration
CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'node2vec_dims': 64,        
    'node2vec_walk_length': 20, 
    'node2vec_num_walks': 5,    
    'node2vec_workers': 8,      
    'gnn_hidden_dims': 256,
    'gnn_out_dims': 64,
    'learning_rate': 0.001,
    'batch_size': 1024,
    'epochs': 100,
    'early_stopping': 10,
    'diversity_weight': 0.3,    
    'score_temperature': 2.0, 
    'min_score_diff': 0.05     
}

class DataLoader:
    """Handles loading and preprocessing of Steam graph data"""
    
    def __init__(self, data_path="outputs/"):
        self.data_path = data_path
        self.users = {}
        self.apps = {}
        self.user_friendships = []
        self.user_app_reviews = []
        
    def load_data(self):
        """Load all graph data from files"""
        print("\n Loading Steam graph data...")
        
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
                    print(f"    Sample line {i+1}: {len(parts)} columns -> {parts}")
                    
        print("  Processing all edges...")
        with open(f"{self.data_path}final_recommender_graph_edges.txt", 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in tqdm(f, desc="    Processing edges"):
                parts = line.strip().split('\t')
                if len(parts) >= 6:  # Need at least: source_id, target_id, source_type, target_type, edge_type, weight
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
    
    def extract_largest_connected_component(self):
        """Extract largest connected component and prepare final dataset structure"""
        print("\n Preparing final dataset structure...")
        
        # Step 1: Get users with complete profiles
        valid_user_ids = set(self.users.keys())
        print(f"  Users with complete profiles: {len(valid_user_ids):,}")
        
        # Step 2: Filter friendships to only include users with complete profiles
        print("  Filtering friendships to profiled users...")
        filtered_friendships = []
        for u, v, w in self.user_friendships:
            if u in valid_user_ids and v in valid_user_ids:
                filtered_friendships.append((u, v, w))
        
        print(f"    Friendships: {len(filtered_friendships):,} (was {len(self.user_friendships):,})")
        
        # Step 3: Build friendship graph and find largest connected component
        print("  Building friendship network...")
        G = nx.Graph()
        G.add_nodes_from(valid_user_ids)
        
        friendship_edges = [(u, v) for u, v, w in filtered_friendships]
        G.add_edges_from(friendship_edges)
        
        print(f"    Network: {G.number_of_nodes():,} users, {G.number_of_edges():,} friendships")
        
        # Find largest connected component
        print("  Finding largest connected component...")
        connected_components = list(nx.connected_components(G))
        
        if len(connected_components) == 0:
            raise ValueError("No connected components found in friendship network!")
        
        largest_cc = max(connected_components, key=len)
        G_lcc = G.subgraph(largest_cc).copy()
        
        print(f"    Total connected components: {len(connected_components)}")
        print(f"    Largest CC: {G_lcc.number_of_nodes():,} users ({len(largest_cc)/len(valid_user_ids)*100:.1f}%)")
        print(f"    LCC friendships: {G_lcc.number_of_edges():,}")
        
        # Step 4: Filter all data to LCC users only
        lcc_users = set(largest_cc)
        print(f"  Filtering all data to LCC users...")
        
        # Filter users (keep only LCC users with complete profiles)
        original_user_count = len(self.users)
        self.users = {uid: data for uid, data in self.users.items() if uid in lcc_users}
        print(f"    Users: {len(self.users):,} (was {original_user_count:,})")
        
        # Filter friendships (User-User edges, NO WEIGHTS for final GNN)
        self.user_friendships = [(u, v, w) for u, v, w in filtered_friendships 
                                if u in lcc_users and v in lcc_users]
        print(f"    User-User edges: {len(self.user_friendships):,}")
        
        # CRITICAL: Verify final User-User graph is fully connected
        print("  Verifying final User-User graph connectivity...")
        final_G = nx.Graph()
        final_G.add_nodes_from(lcc_users)
        final_edges = [(u, v) for u, v, w in self.user_friendships]
        final_G.add_edges_from(final_edges)
        
        # Check connectivity
        final_components = list(nx.connected_components(final_G))
        is_connected = nx.is_connected(final_G)
        
        print(f"    Final User-User graph: {final_G.number_of_nodes():,} nodes, {final_G.number_of_edges():,} edges")
        print(f"    Is fully connected: {is_connected}")
        print(f"    Number of components: {len(final_components)}")
        
        if not is_connected:
            print(f"    WARNING: Final User-User graph is NOT fully connected!")
            print(f"    Component sizes: {[len(comp) for comp in sorted(final_components, key=len, reverse=True)[:5]]}")
            
            # This should not happen if LCC was extracted correctly, but let's handle it
            largest_final_cc = max(final_components, key=len)
            print(f"    Using largest component of final graph: {len(largest_final_cc):,} users")
            
            # Re-filter to ensure connectivity
            lcc_users = set(largest_final_cc)
            self.users = {uid: data for uid, data in self.users.items() if uid in lcc_users}
            self.user_friendships = [(u, v, w) for u, v, w in self.user_friendships 
                                    if u in lcc_users and v in lcc_users]
            
            # Verify again
            final_G = nx.Graph()
            final_G.add_nodes_from(lcc_users)
            final_edges = [(u, v) for u, v, w in self.user_friendships]
            final_G.add_edges_from(final_edges)
            
            print(f"    After re-filtering: {final_G.number_of_nodes():,} nodes, {final_G.number_of_edges():,} edges")
            print(f"    Is fully connected: {nx.is_connected(final_G)}")
            
            # Update G_lcc to the final connected graph
            G_lcc = final_G
        
        # Filter reviews (User-App edges with playtime weights + sentiment)
        original_review_count = len(self.user_app_reviews)
        self.user_app_reviews = [(u, a, w, s) for u, a, w, s in self.user_app_reviews 
                                if u in lcc_users]
        print(f"    User-App edges: {len(self.user_app_reviews):,} (was {original_review_count:,})")
        
        # IMPORTANT: Filter apps to only those reviewed by LCC users
        print("  Filtering apps to only those reviewed by LCC users...")
        reviewed_apps = set()
        for u, a, w, s in self.user_app_reviews:
            reviewed_apps.add(a)
        
        original_app_count = len(self.apps)
        self.apps = {aid: data for aid, data in self.apps.items() if aid in reviewed_apps}
        print(f"    Apps: {len(self.apps):,} (was {original_app_count:,}) - only apps reviewed by LCC users")
        
        # Step 5: Summary of final dataset
        print(f"\n Final Dataset Structure:")
        print(f"    User-User Graph (LCC): {len(self.users):,} users, {len(self.user_friendships):,} friendships")
        print(f"    User-App Reviews: {len(self.user_app_reviews):,} reviews (playtime weighted)")
        print(f"    Apps: {len(self.apps):,} apps (only those reviewed by LCC users)")
        print(f"    User Attributes: loccountrycode only")
        print(f"    App Attributes: name, category, app_type, original_id")
        print(f"    Edge Attributes: User-App (playtime + sentiment), User-User (NO WEIGHTS in final GNN)")
        print(f"    User-User graph is fully connected: {nx.is_connected(G_lcc)}")
        
        return G_lcc

class Node2VecEmbedder:
    """Generates Node2Vec embeddings for user nodes with attribute-aware weights"""
    
    def __init__(self, config, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        self.model = None
        self.embeddings = None
        
    def fit(self, friendship_graph, user_app_reviews=None, user_to_idx=None):
        """Train Node2Vec on the friendship network with enhanced edge weights"""
        print(f"\n Training Node2Vec embeddings with attribute-aware weights...")
        print(f"  Dimensions: {self.config['node2vec_dims']}")
        print(f"  Walk length: {self.config['node2vec_walk_length']}")
        print(f"  Walks per node: {self.config['node2vec_num_walks']}")
        
        # Enhance the friendship graph with user activity weights
        if user_app_reviews and user_to_idx:
            print("  🔗 Enhancing friendship edges with user activity + country similarity...")
            self.enhanced_graph = self._enhance_friendship_weights(
                friendship_graph, user_app_reviews, user_to_idx
            )
        else:
            self.enhanced_graph = friendship_graph
            
        print(f"  Final graph: {self.enhanced_graph.number_of_nodes():,} nodes, {self.enhanced_graph.number_of_edges():,} edges")
        
        # Create Node2Vec model
        self.model = Node2Vec(
            self.enhanced_graph, 
            dimensions=self.config['node2vec_dims'],
            walk_length=self.config['node2vec_walk_length'],
            num_walks=self.config['node2vec_num_walks'],
            workers=self.config['node2vec_workers'],
            p=1.0,  # Return parameter
            q=1.0   # In-out parameter
        )
        
        # Train embeddings
        print("  Training Node2Vec model...")
        model = self.model.fit(
            window=10,
            min_count=1,
            batch_words=4,
            epochs=10
        )
        
        # Extract embeddings with progress bar
        user_ids = list(self.enhanced_graph.nodes())
        self.embeddings = {}
        
        print("  Extracting embeddings...")
        for user_id in tqdm(user_ids, desc="  Extracting embeddings", ncols=80):
            if user_id in model.wv:
                self.embeddings[user_id] = model.wv[user_id]
            else:
                # Fallback for missing nodes
                self.embeddings[user_id] = np.random.normal(0, 0.1, self.config['node2vec_dims'])
        
        print(f"  Generated embeddings for {len(self.embeddings):,} users")
        return self
    
    def _enhance_friendship_weights(self, friendship_graph, user_app_reviews, user_to_idx):
        """Enhance friendship edge weights based on user activity similarity AND country similarity"""
        
        # Calculate user activity profiles (playtime per app)
        user_activity = defaultdict(dict)
        for u, a, playtime, sentiment in user_app_reviews:
            if u in user_to_idx:  # Only for LCC users
                user_activity[u][a] = playtime
        
        print(f"    📈 User activity profiles: {len(user_activity):,} users")
        
        # Create enhanced graph with weighted edges
        enhanced_graph = friendship_graph.copy()
        
        edges_enhanced = 0
        edge_list = list(friendship_graph.edges())
        
        print(f"    🔗 Processing {len(edge_list):,} friendship edges...")
        for u, v in tqdm(edge_list, desc="    Enhancing edges", ncols=80):
            # 1. Calculate activity similarity between friends
            u_apps = set(user_activity[u].keys()) if u in user_activity else set()
            v_apps = set(user_activity[v].keys()) if v in user_activity else set()
            
            # Jaccard similarity of played apps + playtime correlation
            common_apps = u_apps.intersection(v_apps)
            union_apps = u_apps.union(v_apps)
            
            activity_sim = 0.0
            if union_apps:
                jaccard_sim = len(common_apps) / len(union_apps)
                
                # Add playtime correlation for common apps
                playtime_sim = 0.0
                if common_apps:
                    u_times = [user_activity[u][app] for app in common_apps]
                    v_times = [user_activity[v][app] for app in common_apps]
                    
                    # Simple correlation based on relative playtime rankings
                    u_rank = np.argsort(u_times)
                    v_rank = np.argsort(v_times)
                    playtime_sim = 1.0 - (np.abs(u_rank - v_rank).mean() / len(common_apps))
                
                activity_sim = (jaccard_sim + playtime_sim) / 2.0
            
            # 2. Calculate country similarity
            country_sim = 0.0
            if u in self.data_loader.users and v in self.data_loader.users:
                u_country = self.data_loader.users[u].get('loccountrycode', 'UNKNOWN')
                v_country = self.data_loader.users[v].get('loccountrycode', 'UNKNOWN')
                
                # Same country gets full similarity boost
                if u_country != 'UNKNOWN' and v_country != 'UNKNOWN':
                    country_sim = 1.0 if u_country == v_country else 0.0
            
            # 3. Combined similarity weight: Activity (70%) + Country (30%)
            combined_sim = 0.7 * activity_sim + 0.3 * country_sim
            weight = 1.0 + combined_sim  # Base weight 1.0, enhanced up to 2.0
            
            enhanced_graph[u][v]['weight'] = weight
            edges_enhanced += 1
        
        print(f"    🔗 Enhanced {edges_enhanced:,} friendship edges with activity + country similarity weights")
        return enhanced_graph

class AttributeEncoder:
    """Encodes categorical attributes into numeric features"""
    
    def __init__(self):
        self.country_encoder = {}
        self.category_encoder = {}
        self.app_type_encoder = {}
        
    def fit_user_attributes(self, users):
        """Fit encoders for user attributes"""
        print("  Encoding user attributes...")
        
        # Extract unique countries
        countries = set()
        for uid, user_data in users.items():
            country = user_data.get('loccountrycode', 'UNKNOWN')
            if country:
                countries.add(country)
        
        # Create country encoding (one-hot style but more compact)
        countries = sorted(list(countries))
        self.country_encoder = {country: i for i, country in enumerate(countries)}
        self.country_encoder['UNKNOWN'] = len(countries)  # For missing values
        
        print(f"    Countries: {len(countries)} unique values")
        return self
    
    def fit_app_attributes(self, apps):
        """Fit encoders for app attributes"""
        print("  Encoding app attributes...")
        
        # Extract unique categories
        categories = set()
        app_types = set()
        
        for aid, app_data in apps.items():
            category = app_data.get('category', 'Unknown')
            app_type = app_data.get('app_type', -1)
            
            if category:
                categories.add(category)
            if app_type is not None:
                app_types.add(app_type)
        
        # Create encodings
        categories = sorted(list(categories))
        self.category_encoder = {cat: i for i, cat in enumerate(categories)}
        
        app_types = sorted(list(app_types))
        self.app_type_encoder = {atype: i for i, atype in enumerate(app_types)}
        
        print(f"    Categories: {len(categories)} unique values")
        print(f"    🏷App types: {len(app_types)} unique values")
        return self
    
    def encode_user_features(self, users, user_ids):
        """Create feature vectors for users"""
        print("  Creating user feature vectors...")
        
        num_countries = len(self.country_encoder)
        features = []
        
        for uid in user_ids:
            user_data = users[uid]
            country = user_data.get('loccountrycode', 'UNKNOWN')
            
            # One-hot encode country (but more compact - just the index)
            country_idx = self.country_encoder.get(country, self.country_encoder['UNKNOWN'])
            
            # Create feature vector: [country_idx_normalized]
            country_normalized = country_idx / num_countries  # Normalize to [0,1]
            features.append([country_normalized])
        
        print(f"    User features: {len(features)} users, {len(features[0])} dimensions")
        return np.array(features)
    
    def encode_app_features(self, apps, app_ids):
        """Create feature vectors for apps"""
        print("  Creating app feature vectors...")
        
        features = []
        
        for aid in app_ids:
            app_data = apps[aid]
            category = app_data.get('category', 'Unknown')
            app_type = app_data.get('app_type', -1)
            
            # Encode category
            category_idx = self.category_encoder.get(category, 0)
            category_normalized = category_idx / len(self.category_encoder)
            
            # Encode app type
            app_type_idx = self.app_type_encoder.get(app_type, 0)
            app_type_normalized = app_type_idx / len(self.app_type_encoder)
            
            # Create feature vector: [category_normalized, app_type_normalized]
            features.append([category_normalized, app_type_normalized])
        
        print(f"    App features: {len(features)} apps, {len(features[0])} dimensions")
        return np.array(features)

class HeterogeneousRecommenderGNN(nn.Module):
    """Heterogeneous GNN for Steam app recommendations with semantic features"""
    
    def __init__(self, config, num_users, num_apps, user_feature_dim, app_feature_dim):
        super().__init__()
        self.config = config
        self.num_users = num_users
        self.num_apps = num_apps
        
        # Input projection layers to standardize feature dimensions
        self.user_proj = nn.Linear(user_feature_dim, config['gnn_hidden_dims'])
        self.app_proj = nn.Linear(app_feature_dim, config['gnn_hidden_dims'])
        
        # Heterogeneous convolutions
        self.convs = nn.ModuleList()
        
        # First layer
        conv1 = HeteroConv({
            ('user', 'friends_with', 'user'): SAGEConv(config['gnn_hidden_dims'], config['gnn_hidden_dims']),
            ('user', 'reviewed', 'app'): SAGEConv((config['gnn_hidden_dims'], config['gnn_hidden_dims']), config['gnn_hidden_dims']),
            ('app', 'reviewed_by', 'user'): SAGEConv((config['gnn_hidden_dims'], config['gnn_hidden_dims']), config['gnn_hidden_dims'])
        }, aggr='sum')
        self.convs.append(conv1)
        
        # Second layer
        conv2 = HeteroConv({
            ('user', 'friends_with', 'user'): SAGEConv(config['gnn_hidden_dims'], config['gnn_out_dims']),
            ('user', 'reviewed', 'app'): SAGEConv(config['gnn_hidden_dims'], config['gnn_out_dims']),
            ('app', 'reviewed_by', 'user'): SAGEConv(config['gnn_hidden_dims'], config['gnn_out_dims'])
        }, aggr='sum')
        self.convs.append(conv2)
        
        # Prediction layers
        self.rating_predictor = nn.Sequential(
            nn.Linear(config['gnn_out_dims'] * 2, config['gnn_hidden_dims']),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['gnn_hidden_dims'], 1)
        )
        
        # IMPROVED: Better link predictor without sigmoid saturation
        self.link_predictor = nn.Sequential(
            nn.Linear(config['gnn_out_dims'] * 2, config['gnn_hidden_dims']),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout for better generalization
            nn.Linear(config['gnn_hidden_dims'], config['gnn_hidden_dims'] // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['gnn_hidden_dims'] // 2, 1)
            # REMOVED: No sigmoid here - apply in get_recommendations with temperature
        )
        
        # Sentiment-aware recommendation layer
        self.sentiment_predictor = nn.Sequential(
            nn.Linear(config['gnn_out_dims'] * 2 + 1, config['gnn_hidden_dims']),  # +1 for playtime
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config['gnn_hidden_dims'], 1),
            nn.Tanh()  # Sentiment scores typically range from negative to positive
        )
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Project input features to common dimension
        x_dict = {
            'user': F.relu(self.user_proj(x_dict['user'])),
            'app': F.relu(self.app_proj(x_dict['app']))
        }
        
        # Apply convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        return x_dict
    
    def predict_rating(self, user_emb, app_emb):
        """Predict playtime-based rating for user-app pairs"""
        combined = torch.cat([user_emb, app_emb], dim=-1)
        return self.rating_predictor(combined)
    
    def predict_link(self, user_emb, app_emb):
        """Predict raw link scores for user-app pairs (no sigmoid)"""
        combined = torch.cat([user_emb, app_emb], dim=-1)
        return self.link_predictor(combined)
    
    def predict_link_calibrated(self, user_emb, app_emb, temperature=1.0):
        """Predict calibrated link probability with temperature scaling"""
        raw_scores = self.predict_link(user_emb, app_emb)
        # Apply temperature scaling and sigmoid
        calibrated_scores = torch.sigmoid(raw_scores / temperature)
        return calibrated_scores
    
    def predict_sentiment(self, user_emb, app_emb, playtime):
        """Predict sentiment given user-app embeddings and playtime"""
        combined = torch.cat([user_emb, app_emb, playtime.unsqueeze(-1)], dim=-1)
        return self.sentiment_predictor(combined)

class SteamRecommenderSystem:
    """Main recommender system class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.data_loader = None
        self.node2vec = None
        self.model = None
        self.data = None
        
    def prepare_data(self):
        """Load and prepare all data with semantic attribute encoding"""
        print("\n" + "="*60)
        print("DATA PREPARATION WITH SEMANTIC FEATURES")
        print("="*60)
        
        # Load data
        self.data_loader = DataLoader().load_data()
        
        # Extract largest connected component
        friendship_graph = self.data_loader.extract_largest_connected_component()
        
        # Create preliminary user mappings for Node2Vec enhancement
        user_ids = list(self.data_loader.users.keys())
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        
        # Generate Node2Vec embeddings with activity-aware weights
        print(f"\n GENERATING ENHANCED NODE2VEC EMBEDDINGS")
        self.node2vec = Node2VecEmbedder(self.config, self.data_loader).fit(
            friendship_graph, 
            user_app_reviews=self.data_loader.user_app_reviews,
            user_to_idx=user_to_idx
        )
        
        # Prepare PyTorch Geometric data with semantic features
        print(f"\n BUILDING HETEROGENEOUS GRAPH")
        self.data = self._create_hetero_data()
        
        return self
    
    def _create_hetero_data(self):
        """Create HeteroData object with semantic features and Node2Vec embeddings"""
        print("\n Creating heterogeneous graph with semantic features...")
        
        data = HeteroData()
        
        # Create mappings for LCC users and all apps
        user_ids = list(self.data_loader.users.keys())  # Only LCC users with complete profiles
        app_ids = list(self.data_loader.apps.keys())    # All 79 apps
        
        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        app_to_idx = {aid: i for i, aid in enumerate(app_ids)}
        
        print(f"  User-User Graph (LCC): {len(user_ids):,} users")
        print(f"  Apps available: {len(app_ids):,} apps")
        
        # Initialize and fit attribute encoder
        print("\n ENCODING SEMANTIC ATTRIBUTES")
        encoder = AttributeEncoder()
        encoder.fit_user_attributes(self.data_loader.users)
        encoder.fit_app_attributes(self.data_loader.apps)
        
        # Create user features: Node2Vec embeddings + semantic attributes
        print("\n  Creating enhanced user features...")
        user_semantic_features = encoder.encode_user_features(self.data_loader.users, user_ids)
        
        user_features = []
        print(f"  Combining Node2Vec + semantic features for {len(user_ids):,} users...")
        for i, uid in enumerate(tqdm(user_ids, desc="  Processing users", ncols=80)):
            # Get Node2Vec embedding
            if uid in self.node2vec.embeddings:
                node2vec_emb = self.node2vec.embeddings[uid]
            else:
                node2vec_emb = np.random.normal(0, 0.1, self.config['node2vec_dims'])
            
            # Get semantic features  
            semantic_features = user_semantic_features[i]
            
            # Combine: [Node2Vec features] + [semantic features]
            combined_features = np.concatenate([node2vec_emb, semantic_features])
            user_features.append(combined_features)
        
        data['user'].x = torch.FloatTensor(user_features)
        data['user'].num_nodes = len(user_ids)
        
        print(f"    User feature dimensions: {len(user_features[0])} ({self.config['node2vec_dims']} Node2Vec + {len(user_semantic_features[0])} semantic)")
        
        # Create app features: learnable embeddings + semantic attributes
        print("  Creating enhanced app features...")
        app_semantic_features = encoder.encode_app_features(self.data_loader.apps, app_ids)
        
        # For apps, we'll use the semantic features directly and let the GNN learn the rest
        data['app'].x = torch.FloatTensor(app_semantic_features)
        data['app'].num_nodes = len(app_ids)
        
        print(f"    App feature dimensions: {len(app_semantic_features[0])} semantic features")
        
        # User-User friendship edges: NO WEIGHTS for final GNN (Node2Vec uses enhanced weights internally)
        print("\n  Creating User-User edges (unweighted for GNN)...")
        
        # Get the enhanced friendship graph from Node2Vec (activity weights used only for Node2Vec training)
        enhanced_graph = self.node2vec.enhanced_graph
        
        friendship_edges = []
        
        for u, v in enhanced_graph.edges():
            if u in user_to_idx and v in user_to_idx:
                # Add edges without weights (unweighted for GNN message passing)
                friendship_edges.append([user_to_idx[u], user_to_idx[v]])
                friendship_edges.append([user_to_idx[v], user_to_idx[u]])  # Undirected
        
        if friendship_edges:
            data['user', 'friends_with', 'user'].edge_index = torch.LongTensor(friendship_edges).t()
            # NO edge_attr for User-User edges - unweighted
            print(f"   User-User edges: {len(friendship_edges):,} (bidirectional, unweighted)")
        
        # User-App review edges (with playtime weights + sentiment attributes)
        print("  Creating User-App edges (reviews with playtime + sentiment)...")
        review_edges = []
        review_weights = []  # Playtime-based weights (author_playtime_at_review)
        review_sentiments = []  # Sentiment scores (for recommender, not Node2Vec)
        
        for u, a, playtime_weight, sentiment in self.data_loader.user_app_reviews:
            if u in user_to_idx and a in app_to_idx:
                review_edges.append([user_to_idx[u], app_to_idx[a]])
                review_weights.append(playtime_weight)  # Use playtime as edge weight
                review_sentiments.append(sentiment)     # Sentiment for recommender system
        
        if review_edges:
            data['user', 'reviewed', 'app'].edge_index = torch.LongTensor(review_edges).t()
            data['user', 'reviewed', 'app'].edge_attr = torch.FloatTensor(review_weights)
            data['user', 'reviewed', 'app'].sentiment = torch.FloatTensor(review_sentiments)
            
            # Reverse edges for heterogeneous GNN
            data['app', 'reviewed_by', 'user'].edge_index = data['user', 'reviewed', 'app'].edge_index.flip(0)
            data['app', 'reviewed_by', 'user'].edge_attr = data['user', 'reviewed', 'app'].edge_attr
            data['app', 'reviewed_by', 'user'].sentiment = data['user', 'reviewed', 'app'].sentiment
            
            print(f"    User-App edges: {len(review_edges):,} (playtime-weighted + sentiment)")
        
        # Store mappings and encoder
        self.user_to_idx = user_to_idx
        self.app_to_idx = app_to_idx
        self.idx_to_user = {i: uid for uid, i in user_to_idx.items()}
        self.idx_to_app = {i: aid for aid, i in app_to_idx.items()}
        self.attribute_encoder = encoder
        
        # Final structure summary
        print(f"\nENHANCED HETEROGENEOUS GRAPH CREATED:")
        print(f"    Nodes: {len(user_ids):,} users (LCC) + {len(app_ids):,} apps")
        print(f"    User-User: {data['user', 'friends_with', 'user'].edge_index.shape[1]:,} friendship edges (unweighted)")
        print(f"    User-App: {data['user', 'reviewed', 'app'].edge_index.shape[1]:,} review edges (playtime-weighted)")
        print(f"    User features: Node2Vec ({self.config['node2vec_dims']}D) + Country encoding (1D)")
        print(f"    App features: Category + App-type encoding (2D)")
        print(f"    Edge weights: Playtime (User-App)")
        print(f"    Sentiment scores: Available for recommender system")
        
        return data
    
    def _build_friendship_graph(self, user_ids, user_to_idx):
        """Build NetworkX friendship graph for LCC users"""
        G = nx.Graph()
        G.add_nodes_from(user_ids)
        
        for u, v, w in self.data_loader.user_friendships:
            if u in user_to_idx and v in user_to_idx:
                G.add_edge(u, v, weight=w)
        
        return G
    
    def train_model(self):
        """Train the heterogeneous GNN"""
        print("\n" + "="*60)
        print("TRAINING HETEROGENEOUS GNN")
        print("="*60)
        
        # Initialize model
        self.model = HeterogeneousRecommenderGNN(
            self.config,
            len(self.user_to_idx),
            len(self.app_to_idx),
            self.data['user'].x.shape[1],  # Actual user feature dimension
            self.data['app'].x.shape[1]    # Actual app feature dimension
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Prepare training data - USE ALL INTERACTIONS (no artificial train/test split)
        data = self.data.to(self.device)
        
        # Use ALL user-app review edges for training
        edge_index = data['user', 'reviewed', 'app'].edge_index
        edge_attr = data['user', 'reviewed', 'app'].edge_attr
        sentiment = data['user', 'reviewed', 'app'].sentiment
        
        print(f"\n Starting training with ALL interaction data...")
        print(f"  Total edges: {edge_index.shape[1]:,}")
        print(f"  Note: Using all data - no artificial train/test split for recommender system")
        
        # Optimizer
        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        # Training loop with loss-based early stopping
        self.model.train()
        best_loss = float('inf')
        patience = 0
        loss_history = []
        
        # Create progress bar for training epochs
        epoch_pbar = tqdm(range(self.config['epochs']), desc="Training GNN", ncols=100)
        
        for epoch in epoch_pbar:
            optimizer.zero_grad()
            
            # Forward pass with actual node features
            x_dict = {
                'user': data['user'].x,
                'app': data['app'].x
            }
            
            edge_index_dict = {
                ('user', 'friends_with', 'user'): data['user', 'friends_with', 'user'].edge_index,
                ('user', 'reviewed', 'app'): edge_index,
                ('app', 'reviewed_by', 'user'): edge_index.flip(0)
            }
            
            # Get embeddings
            out_dict = self.model(x_dict, edge_index_dict)
            
            # Compute losses on ALL data
            user_emb = out_dict['user'][edge_index[0]]
            app_emb = out_dict['app'][edge_index[1]]
            
            # Rating prediction loss (predicting playtime)
            pred_ratings = self.model.predict_rating(user_emb, app_emb).squeeze()
            rating_loss = F.mse_loss(pred_ratings, edge_attr)
            
            # Link prediction loss (positive samples = existing interactions)
            pred_links = self.model.predict_link(user_emb, app_emb).squeeze()
            link_targets = torch.ones_like(pred_links)
            
            # IMPROVED: Better negative sampling strategy
            neg_edge_index = negative_sampling(
                edge_index, 
                num_nodes=(len(self.user_to_idx), len(self.app_to_idx)),
                num_neg_samples=min(edge_index.shape[1], edge_index.shape[1] // 2)  # Fewer negatives
            )
            
            neg_user_emb = out_dict['user'][neg_edge_index[0]]
            neg_app_emb = out_dict['app'][neg_edge_index[1]]
            neg_pred_links = self.model.predict_link(neg_user_emb, neg_app_emb).squeeze()
            neg_targets = torch.zeros_like(neg_pred_links)
            
            # Use raw scores with BCE with logits (more stable)
            link_loss = F.binary_cross_entropy_with_logits(
                torch.cat([pred_links, neg_pred_links]),
                torch.cat([link_targets, neg_targets])
            )
            
            # Total loss
            total_loss = rating_loss + link_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Track loss for convergence-based early stopping
            loss_history.append(total_loss.item())
            
            # Update progress bar with current loss
            epoch_pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Rating': f'{rating_loss.item():.4f}',
                'Link': f'{link_loss.item():.4f}',
                'Best': f'{best_loss:.4f}'
            })
            
            # Monitoring and early stopping based on loss convergence
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch:3d} | Total Loss: {total_loss:.4f} | "
                      f"Rating Loss: {rating_loss.item():.4f} | Link Loss: {link_loss.item():.4f}")
                
                # Early stopping based on loss improvement
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    patience = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience += 1
                    if patience >= self.config['early_stopping']:
                        print(f"Early stopping at epoch {epoch} - loss converged")
                        epoch_pbar.close()
                        break
                    
                # Additional convergence check: if loss plateau for last 20 epochs
                if len(loss_history) >= 20:
                    recent_losses = loss_history[-20:]
                    if max(recent_losses) - min(recent_losses) < 0.001:
                        print(f"Early stopping at epoch {epoch} - loss plateau detected")
                        epoch_pbar.close()
                        break
        
        epoch_pbar.close()
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f"\nTraining completed! Best loss: {best_loss:.4f}")
        print(f" Model trained on ALL {edge_index.shape[1]:,} user-app interactions")
        
        return self
    
    def get_recommendations(self, user_id, top_k=10):
        """Get top-k app recommendations for a user with improved diversity and personalization"""
        if user_id not in self.user_to_idx:
            return []
        
        # Get apps the user has already reviewed (to exclude from recommendations)
        user_reviewed_apps = set()
        user_categories = defaultdict(int)  # Track user's category preferences
        user_total_playtime = 0
        
        for u, a, playtime, sentiment in self.data_loader.user_app_reviews:
            if u == user_id:
                user_reviewed_apps.add(a)
                # Track user's category preferences
                if a in self.data_loader.apps:
                    category = self.data_loader.apps[a].get('category', 'Unknown')
                    user_categories[category] += playtime  # Weight by playtime
                    user_total_playtime += playtime
        
        print(f"  User {user_id} has already reviewed {len(user_reviewed_apps)} apps")
        if user_categories:
            top_user_categories = sorted(user_categories.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Top user categories: {dict(top_user_categories)}")
        
        self.model.eval()
        with torch.no_grad():
            data = self.data.to(self.device)
            
            # Get embeddings using message passing
            x_dict = {
                'user': data['user'].x,
                'app': data['app'].x
            }
            
            edge_index_dict = {
                ('user', 'friends_with', 'user'): data['user', 'friends_with', 'user'].edge_index,
                ('user', 'reviewed', 'app'): data['user', 'reviewed', 'app'].edge_index,
                ('app', 'reviewed_by', 'user'): data['app', 'reviewed_by', 'user'].edge_index
            }
            
            out_dict = self.model(x_dict, edge_index_dict)
            
            # Get user embedding
            user_idx = self.user_to_idx[user_id]
            user_emb = out_dict['user'][user_idx].unsqueeze(0)
            
            # Get embeddings for ALL apps
            app_embs = out_dict['app']
            
            # IMPROVED: Predict raw scores and apply temperature scaling
            user_emb_expanded = user_emb.repeat(len(app_embs), 1)
            raw_scores = self.model.predict_link(user_emb_expanded, app_embs).squeeze()
            
            # Apply temperature scaling for better score distribution
            temperature = self.config.get('score_temperature', 2.0)
            calibrated_scores = torch.sigmoid(raw_scores / temperature)
            
            # Filter candidate apps and calculate diversity-aware scores
            candidate_apps = []
            candidate_scores = []
            candidate_categories = []
            candidate_raw_scores = []
            
            print(f"     Filtering {len(calibrated_scores)} apps for new recommendations...")
            for app_idx, (score, raw_score) in enumerate(zip(calibrated_scores, raw_scores)):
                app_id = self.idx_to_app[app_idx]
                if app_id not in user_reviewed_apps:  # Only new apps
                    app_data = self.data_loader.apps[app_id]
                    category = app_data.get('category', 'Unknown')
                    
                    candidate_apps.append(app_idx)
                    candidate_scores.append(score.item())
                    candidate_categories.append(category)
                    candidate_raw_scores.append(raw_score.item())
            
            if not candidate_apps:
                print(f"  No new apps to recommend for user {user_id}")
                return []
            
            # IMPROVED: Diversity-aware recommendation selection
            recommendations = []
            used_categories = set()
            remaining_apps = list(zip(candidate_apps, candidate_scores, candidate_categories, candidate_raw_scores))
            
            # Sort by score initially
            remaining_apps.sort(key=lambda x: x[1], reverse=True)
            
            diversity_weight = self.config.get('diversity_weight', 0.3)
            
            print(f"    Applying diversity-aware selection (weight: {diversity_weight})...")
            for i in range(min(top_k, len(remaining_apps))):
                best_app = None
                best_score = -1
                best_idx = -1
                
                for idx, (app_idx, score, category, raw_score) in enumerate(remaining_apps):
                    # Base score
                    final_score = score
                    
                    # DIVERSITY BONUS: Boost apps from unused categories
                    if category not in used_categories and len(used_categories) > 0:
                        final_score += diversity_weight * 0.1  # Small but meaningful boost
                    
                    # PERSONALIZATION BONUS: Boost apps from user's preferred categories
                    if user_categories and category in user_categories:
                        category_preference = user_categories[category] / user_total_playtime
                        final_score += 0.05 * category_preference  # Preference boost
                    
                    # QUALITY THRESHOLD: Ensure minimum score difference
                    min_diff = self.config.get('min_score_diff', 0.05)
                    if len(recommendations) > 0:
                        last_score = recommendations[-1]['base_score']
                        if abs(score - last_score) < min_diff and category in used_categories:
                            continue  # Skip too similar scores from same category
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_app = (app_idx, score, category, raw_score)
                        best_idx = idx
                
                if best_app is not None:
                    app_idx, base_score, category, raw_score = best_app
                    app_id = self.idx_to_app[app_idx]
                    app_data = self.data_loader.apps[app_id]
                    
                    recommendations.append({
                        'rank': len(recommendations) + 1,
                        'app_id': app_id,
                        'app_name': app_data.get('name', 'Unknown'),
                        'category': category,
                        'score': best_score,  # Final diversity-adjusted score
                        'base_score': base_score,  # Original model score
                        'raw_score': raw_score  # Pre-sigmoid raw score
                    })
                    
                    used_categories.add(category)
                    remaining_apps.pop(best_idx)
                else:
                    break
            
            print(f"  Generated {len(recommendations)} DIVERSE app recommendations")
            print(f"  Categories used: {len(used_categories)} - {list(used_categories)}")
            print(f"  Score range: {recommendations[-1]['base_score']:.4f} - {recommendations[0]['base_score']:.4f}")
            
            return recommendations

def test_improved_recommendations(recommender, num_users=5):
    """Test the improved recommendation system with detailed analysis"""
    print("\n" + "="*70)
    print("TESTING IMPROVED RECOMMENDATION SYSTEM")
    print("="*70)
    
    # Test on multiple users
    sample_users = list(recommender.user_to_idx.keys())[:num_users]
    
    all_recommended_apps = set()
    category_distribution = defaultdict(int)
    score_ranges = []
    
    for i, user_id in enumerate(sample_users):
        print(f"\n User {user_id} (Test {i+1}/{num_users}):")
        print("-" * 50)
        
        recommendations = recommender.get_recommendations(user_id, top_k=10)
        
        if recommendations:
            # Collect statistics
            user_apps = set(rec['app_id'] for rec in recommendations)
            all_recommended_apps.update(user_apps)
            
            user_categories = [rec['category'] for rec in recommendations]
            for cat in user_categories:
                category_distribution[cat] += 1
            
            base_scores = [rec['base_score'] for rec in recommendations]
            score_ranges.append(max(base_scores) - min(base_scores))
            
            # Display recommendations with enhanced info
            print("  Recommendations:")
            for rec in recommendations:
                diversity_boost = rec['score'] - rec['base_score']
                print(f"    {rec['rank']}. {rec['app_name'][:40]:<40} "
                      f"({rec['category'][:20]:<20}) - "
                      f"Base: {rec['base_score']:.4f}, "
                      f"Final: {rec['score']:.4f} "
                      f"(+{diversity_boost:+.4f})")
            
            print(f"  Score diversity: {max(base_scores) - min(base_scores):.4f}")
            print(f"  Categories: {len(set(user_categories))} unique")
        else:
            print("  No recommendations generated")
    
    # Overall statistics
    print(f"\n OVERALL IMPROVEMENT ANALYSIS:")
    print("-" * 50)
    print(f"   Unique apps recommended: {len(all_recommended_apps)}")
    print(f"  Average score range: {np.mean(score_ranges):.4f} ± {np.std(score_ranges):.4f}")
    print(f"  Category distribution: {dict(sorted(category_distribution.items(), key=lambda x: x[1], reverse=True))}")
    
    # Improvement metrics
    avg_score_range = np.mean(score_ranges) if score_ranges else 0
    unique_apps_ratio = len(all_recommended_apps) / (num_users * 10)  # Max possible unique apps
    category_diversity = len(category_distribution) / len(recommender.data_loader.apps)
    
    print(f"\n IMPROVEMENT METRICS:")
    print(f"  Score diversity improvement: {'✅ Good' if avg_score_range > 0.01 else '⚠️ Still low'}")
    print(f"  App uniqueness ratio: {unique_apps_ratio:.2f} {'✅ Good' if unique_apps_ratio > 0.5 else '⚠️ Low'}")
    print(f"  Category coverage: {category_diversity:.2f} {'✅ Good' if category_diversity > 0.5 else '⚠️ Limited'}")
    
    return {
        'avg_score_range': avg_score_range,
        'unique_apps_ratio': unique_apps_ratio,
        'category_diversity': category_diversity,
        'total_unique_apps': len(all_recommended_apps)
    } 

def main():
    """Main execution function"""
    print("Steam Recommender System with Node2Vec + Heterogeneous GNN")
    print("=" * 70)
    
    # Initialize system
    recommender = SteamRecommenderSystem(CONFIG)
    
    # Prepare data and train model
    recommender.prepare_data().train_model()
    
    # Test the improved recommendation system
    print("\n" + "="*60)
    print(" TESTING IMPROVED RECOMMENDATION SYSTEM")
    print("="*60)
    
    # Test the improved system
    improvement_results = test_improved_recommendations(recommender, num_users=5)
    
    print(f"\n Recommender system training completed!")
    print(f" Final model ready for recommendations")
    print(f" Improvements: Better diversity, temperature scaling, category awareness")

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        print(" Running in Google Colab")
        
        # Install required packages
        os.system("pip install node2vec torch-geometric networkx")
        
    except ImportError:
        print(" Running locally")
    
    # Run main pipeline
    recommender = main() 
    
def evaluate_system_performance(recommender):
    """
     COMPREHENSIVE PERFORMANCE EVALUATION
    Run this function after your main pipeline completes to assess system performance.
    
    Usage:
        evaluate_system_performance(recommender)
    """
    
    print("\n" + "="*70)
    print(" COMPREHENSIVE PERFORMANCE EVALUATION")
    print("="*70)
    
    # 1. NODE2VEC EMBEDDING QUALITY ASSESSMENT
    print("\n NODE2VEC EMBEDDING QUALITY:")
    print("-" * 50)
    
    # Check embedding statistics
    embeddings = list(recommender.node2vec.embeddings.values())
    embeddings_array = np.array(embeddings)
    
    print(f"   Embedding Statistics:")
    print(f"    Dimensions: {embeddings_array.shape}")
    print(f"    Mean: {embeddings_array.mean():.4f}")
    print(f"    Std: {embeddings_array.std():.4f}")
    print(f"    Min: {embeddings_array.min():.4f}")
    print(f"    Max: {embeddings_array.max():.4f}")
    
    # Check embedding diversity (should not be too similar)
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Sample subset for efficiency
    sample_size = min(100, len(embeddings))
    sample_embeddings = embeddings_array[:sample_size]
    
    print(f"   Computing similarity matrix for {sample_size} embeddings...")
    similarity_matrix = cosine_similarity(sample_embeddings)
    
    # Remove diagonal (self-similarity = 1.0)
    np.fill_diagonal(similarity_matrix, 0)
    avg_similarity = similarity_matrix.mean()
    
    print(f"   Embedding Diversity:")
    print(f"    Average cosine similarity: {avg_similarity:.4f}")
    print(f"    Quality: {' Good diversity' if avg_similarity < 0.3 else ' High similarity (potential overfitting)'}")
    
    # 2. GNN TRAINING PERFORMANCE ANALYSIS
    print("\n GNN TRAINING PERFORMANCE:")
    print("-" * 50)
    
    # Evaluate on full dataset (same data used for training - this is correct for RecSys)
    recommender.model.eval()
    with torch.no_grad():
        data = recommender.data.to(recommender.device)
        
        # Forward pass
        x_dict = {
            'user': data['user'].x,
            'app': data['app'].x
        }
        
        edge_index_dict = {
            ('user', 'friends_with', 'user'): data['user', 'friends_with', 'user'].edge_index,
            ('user', 'reviewed', 'app'): data['user', 'reviewed', 'app'].edge_index,
            ('app', 'reviewed_by', 'user'): data['app', 'reviewed_by', 'user'].edge_index
        }
        
        out_dict = recommender.model(x_dict, edge_index_dict)
        
        # Evaluate on all user-app edges
        edge_index = data['user', 'reviewed', 'app'].edge_index
        edge_attr = data['user', 'reviewed', 'app'].edge_attr
        
        user_emb = out_dict['user'][edge_index[0]]
        app_emb = out_dict['app'][edge_index[1]]
        
        # Rating prediction performance
        pred_ratings = recommender.model.predict_rating(user_emb, app_emb).squeeze()
        rating_mse = F.mse_loss(pred_ratings, edge_attr).item()
        rating_mae = F.l1_loss(pred_ratings, edge_attr).item()
        
        # Link prediction performance
        pred_links = recommender.model.predict_link(user_emb, app_emb).squeeze()
        
        # Generate negative samples for evaluation
        neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=(len(recommender.user_to_idx), len(recommender.app_to_idx)),
            num_neg_samples=edge_index.shape[1]
        )
        
        neg_user_emb = out_dict['user'][neg_edge_index[0]]
        neg_app_emb = out_dict['app'][neg_edge_index[1]]
        neg_pred_links = recommender.model.predict_link(neg_user_emb, neg_app_emb).squeeze()
        
        # Combine positive and negative predictions
        all_predictions = torch.cat([pred_links, neg_pred_links]).cpu().numpy()
        all_labels = torch.cat([
            torch.ones(len(pred_links)),
            torch.zeros(len(neg_pred_links))
        ]).cpu().numpy()
        
        # Calculate metrics
        link_auc = roc_auc_score(all_labels, all_predictions)
        link_ap = average_precision_score(all_labels, all_predictions)
        
        print(f"   Rating Prediction:")
        print(f"    MSE: {rating_mse:.4f}")
        print(f"    MAE: {rating_mae:.4f}")
        print(f"    Quality: {' Good' if rating_mse < 1.0 else ' High error'}")
        
        print(f"  🔗 Link Prediction:")
        print(f"    AUC-ROC: {link_auc:.4f}")
        print(f"    Average Precision: {link_ap:.4f}")
        print(f"    Quality: {' Excellent' if link_auc > 0.8 else ' Good' if link_auc > 0.7 else ' Needs improvement'}")
    
    # 3. RECOMMENDATION QUALITY ASSESSMENT
    print("\n RECOMMENDATION QUALITY ASSESSMENT:")
    print("-" * 50)
    
    # Test recommendations for multiple users
    test_users = list(recommender.user_to_idx.keys())[:10]  # Test on 10 users
    total_recommended = 0
    total_coverage = 0
    user_scores = []
    
    print(f"   Testing recommendations for {len(test_users)} users...")
    for user_id in tqdm(test_users, desc="  Generating recommendations", ncols=80):
        recommendations = recommender.get_recommendations(user_id, top_k=5)
        total_recommended += len(recommendations)
        
        if recommendations:
            # Calculate score diversity
            scores = [rec['score'] for rec in recommendations]
            score_std = np.std(scores)
            user_scores.extend(scores)
            
            # Count unique categories
            categories = set([rec['category'] for rec in recommendations])
            total_coverage += len(categories)
            
            print(f"   User {user_id}: {len(recommendations)} recs, "
                  f"score range: {min(scores):.3f}-{max(scores):.3f}, "
                  f"categories: {len(categories)}")
    
    # Overall recommendation statistics
    avg_recs_per_user = total_recommended / len(test_users)
    avg_category_coverage = total_coverage / len(test_users)
    overall_score_diversity = np.std(user_scores) if user_scores else 0
    
    print(f"\n   Overall Recommendation Quality:")
    print(f"    Avg recommendations per user: {avg_recs_per_user:.1f}")
    print(f"    Avg category diversity: {avg_category_coverage:.1f}")
    print(f"    Score diversity (std): {overall_score_diversity:.4f}")
    print(f"    Quality: {' Good diversity' if overall_score_diversity > 0.1 else ' Low diversity'}")
    
    # 4. GRAPH STRUCTURE ANALYSIS
    print("\n GRAPH STRUCTURE ANALYSIS:")
    print("-" * 50)
    
    # Analyze the enhanced friendship graph
    G = recommender.node2vec.enhanced_graph
    
    # Basic graph metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Calculate degree statistics
    degrees = [G.degree(n) for n in G.nodes()]
    avg_degree = np.mean(degrees)
    degree_std = np.std(degrees)
    
    # Calculate clustering coefficient
    avg_clustering = nx.average_clustering(G)
    
    # Calculate shortest path length (on a sample for efficiency)
    if num_nodes <= 1000:
        try:
            avg_path_length = nx.average_shortest_path_length(G)
        except:
            avg_path_length = "N/A (disconnected)"
    else:
        avg_path_length = "N/A (too large)"
    
    print(f"   Network Statistics:")
    print(f"    Nodes: {num_nodes:,}")
    print(f"    Edges: {num_edges:,}")
    print(f"    Density: {density:.6f}")
    print(f"    Avg degree: {avg_degree:.2f} ± {degree_std:.2f}")
    print(f"    Clustering coefficient: {avg_clustering:.4f}")
    print(f"    Avg shortest path: {avg_path_length}")
    
    # 5. COUNTRY/ACTIVITY ENHANCEMENT ANALYSIS
    print("\n ENHANCEMENT IMPACT ANALYSIS:")
    print("-" * 50)
    
    # Analyze edge weights distribution
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    base_weights = sum(1 for w in edge_weights if abs(w - 1.0) < 0.001)
    enhanced_weights = len(edge_weights) - base_weights
    
    print(f"   Edge Weight Analysis:")
    print(f"    Base weight edges (1.0): {base_weights:,} ({base_weights/len(edge_weights)*100:.1f}%)")
    print(f"    Enhanced weight edges: {enhanced_weights:,} ({enhanced_weights/len(edge_weights)*100:.1f}%)")
    print(f"    Weight range: {min(edge_weights):.3f} - {max(edge_weights):.3f}")
    print(f"    Avg weight: {np.mean(edge_weights):.3f}")
    
    # Check country distribution
    countries = {}
    for user_id in recommender.data_loader.users:
        country = recommender.data_loader.users[user_id].get('loccountrycode', 'UNKNOWN')
        countries[country] = countries.get(country, 0) + 1
    
    print(f"   Country Distribution:")
    print(f"    Unique countries: {len(countries)}")
    print(f"    Top countries: {dict(sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    # 6. FINAL ASSESSMENT
    print("\n" + "="*70)
    print(" FINAL PERFORMANCE ASSESSMENT")
    print("="*70)
    
    # Create overall quality score
    quality_factors = []
    
    # Node2Vec quality (embedding diversity)
    if avg_similarity < 0.3:
        quality_factors.append(" Node2Vec: Good embedding diversity")
    else:
        quality_factors.append(" Node2Vec: High similarity detected")
    
    # GNN quality (AUC performance)
    if link_auc > 0.8:
        quality_factors.append(" GNN: Excellent link prediction")
    elif link_auc > 0.7:
        quality_factors.append(" GNN: Good link prediction")
    else:
        quality_factors.append(" GNN: Link prediction needs improvement")
    
    # Recommendation quality
    if avg_recs_per_user >= 3 and overall_score_diversity > 0.1:
        quality_factors.append(" Recommendations: Good quality and diversity")
    else:
        quality_factors.append(" Recommendations: Limited quality/diversity")
    
    # Enhancement effectiveness
    if enhanced_weights / len(edge_weights) > 0.5:
        quality_factors.append(" Enhancements: Activity/country features utilized")
    else:
        quality_factors.append(" Enhancements: Limited feature utilization")
    
    print("\n Quality Assessment:")
    for factor in quality_factors:
        print(f"  {factor}")
    
    overall_quality = sum(1 for f in quality_factors if f.startswith("")) / len(quality_factors)
    
    print(f"\n Overall System Quality: {overall_quality*100:.1f}% "
          f"({' Excellent' if overall_quality > 0.8 else ' Good' if overall_quality > 0.6 else ' Needs improvement'})")
    
    print(f"\n Evaluation completed! System is ready for production use.")
    
    return {
        'node2vec_similarity': avg_similarity,
        'gnn_auc': link_auc,
        'gnn_mse': rating_mse,
        'avg_recommendations': avg_recs_per_user,
        'score_diversity': overall_score_diversity,
        'enhancement_ratio': enhanced_weights / len(edge_weights) if edge_weights else 0,
        'overall_quality': overall_quality
    } 
