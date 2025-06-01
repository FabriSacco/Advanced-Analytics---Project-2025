"""
Quick Test Script for Improved Steam Recommender
Run this after your main pipeline to test the improvements without retraining.
"""

import sys
import os

# Add the main script to path
sys.path.append('.')

def quick_test_improvements():
    """Quick test of the improved recommendation system"""
    print("🔧 QUICK TEST: Improved Recommendation System")
    print("="*60)
    
    # Import from main script
    from steam_recommender_node2vec_gnn import SteamRecommenderSystem, CONFIG, test_improved_recommendations
    
    print("📂 Loading existing trained model...")
    
    # Initialize system (will use existing data if available)
    recommender = SteamRecommenderSystem(CONFIG)
    
    try:
        # Try to load existing data and model
        print("  🔄 Attempting to load existing model...")
        
        # Prepare data (fast if already cached)
        recommender.prepare_data()
        
        # Load trained model if it exists
        import torch
        if os.path.exists('best_model.pth'):
            print("  ✅ Loading saved model weights...")
            recommender.model.load_state_dict(torch.load('best_model.pth'))
            print("  🎯 Model loaded successfully!")
        else:
            print("  ⚠️ No saved model found. Please run the full training first.")
            return
        
        # Test improved recommendations
        print("\n🎯 Testing improved recommendation system...")
        results = test_improved_recommendations(recommender, num_users=3)
        
        # Summary
        print(f"\n✅ Quick test completed!")
        print(f"📊 Results: {results}")
        
    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        print("💡 Please run the full pipeline first to train the model.")

if __name__ == "__main__":
    quick_test_improvements() 