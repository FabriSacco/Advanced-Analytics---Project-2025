import json
from collections import defaultdict

def categorize_games():
    """Categorize the 80 games into balanced categories for recommender systems."""
    
    # Manual categorization based on game knowledge and genres
    # 6 categories to balance detail vs complexity
    game_categories = {
        "Action & Adventure": [
            "Marvel's Spider-Man 2",
            "Indiana Jones and the Great Circle", 
            "Tomb Raider IV-VI Remastered",
            "DYNASTY WARRIORS: ORIGINS",
            "NINJA GAIDEN 2 Black",
            "Like a Dragon: Pirate Yakuza in Hawaii",
            "WARRIORS: Abyss",
            "Grapples Galore",
            "ANTONBLAST",
            "Hyper Light Breaker",
            "SWORN",
            # Horror & Thriller merged here
            "The Thing: Remastered",
            "KLETKA",
            "Garten of Banban 0",
            "Urban Myth Dissolution Center"
        ],
        
        "RPG & JRPG": [
            "FINAL FANTASY VII REBIRTH",
            "Kingdom Come: Deliverance II",
            "Avowed",
            "Tales of Graces f Remastered",
            "The Legend of Heroes: Trails through Daybreak II",
            "Monster Hunter Wilds",
            "Path of Exile 2",
            "Heroes of Hammerwatch II",
            "Legacy: Steel & Sorcery",
            "ENDER MAGNOLIA: Bloom in the Mist"
        ],
        
        "Strategy & Tactics": [
            "Sid Meier's Civilization VII",
            "Company of Heroes 3: Fire & Steel",
            "Door Kickers 2: Task Force North",
            "Foundation",
            "Age of Darkness: Final Stand",
            "Mind Over Magic",
            "GIRLS' FRONTLINE 2: EXILIUM",
            "Orcs Must Die! Deathtrap",
            "Total War: WARHAMMER III - Golgfag – Omens of Destruction",
            "Total War: WARHAMMER III - Gorbad – Omens of Destruction"
        ],
        
        "Simulation & Management": [
            "My Summer Car",
            "Assetto Corsa EVO", 
            "Fast Food Simulator",
            "R.E.P.O.",
            "Euro Truck Simulator 2 - Greece",
            "Ranch Simulator: Southwest Ranch & Farm Expansion Pack",
            "Keep Driving",
            "Return to campus",
            "Space Engineers 2",
            "Aloft",
            # Sports & Racing merged here
            "PGA TOUR 2K25",
            "Tokyo Xtreme Racer",
            "Alpha League HD",
            "Hello Kitty Island Adventure"
        ],
        
        "Shooter & Combat": [
            "Marvel Rivals",
            "Delta Force",
            "Sniper Elite: Resistance",
            "TRIBE NINE",
            "SYNDUALITY Echo of Ada",
            "Murky Divers"
        ],
        
        "Indie & Casual": [
            "MiSide",
            "Ballionaire", 
            "Puck",
            "A Game About Digging A Hole",
            "Awaria",
            "Aurelia",
            "The Headliners",
            "Play Together",
            "Desktop Mate",
            "Incremental Epic Hero 2",
            "Rift of the NecroDancer"
        ]
    }
    
    return game_categories

def analyze_categorization():
    """Analyze the game categorization for balance and coverage."""
    categories = categorize_games()
    
    print("=== GAME CATEGORIZATION FOR RECOMMENDER SYSTEM ===\n")
    
    total_games = 0
    category_stats = []
    
    for category, games in categories.items():
        count = len(games)
        total_games += count
        category_stats.append((category, count))
        
        print(f"📁 {category} ({count} games):")
        for i, game in enumerate(sorted(games), 1):
            print(f"   {i:2d}. {game}")
        print()
    
    # Statistics
    print("=== CATEGORIZATION ANALYSIS ===")
    print(f"Total categories: {len(categories)}")
    print(f"Total games categorized: {total_games}")
    print(f"Expected total: 80")
    print(f"Missing games: {80 - total_games}")
    
    print(f"\nCategory size distribution:")
    category_stats.sort(key=lambda x: x[1], reverse=True)
    for category, count in category_stats:
        percentage = (count / total_games) * 100
        print(f"  {category}: {count} games ({percentage:.1f}%)")
    
    # Balance analysis
    sizes = [count for _, count in category_stats]
    avg_size = sum(sizes) / len(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    
    print(f"\nBalance metrics:")
    print(f"  Average category size: {avg_size:.1f} games")
    print(f"  Size range: {min_size} - {max_size} games")
    print(f"  Size variance: {max_size - min_size} games")
    
    # Recommendations for recommender systems
    print(f"\n=== RECOMMENDER SYSTEM BENEFITS ===")
    print(f"✅ {len(categories)} categories provide good granularity")
    print(f"✅ Categories range from {min_size}-{max_size} games (reasonable balance)")
    print(f"✅ Genre-based categories align with user preferences") 
    print(f"✅ Sufficient diversity for collaborative filtering")
    print(f"✅ Categories can be used as features for content-based filtering")
    
    return categories

def save_game_categories():
    """Save the categorization for use in recommender system."""
    categories = categorize_games()
    
    # Create a mapping from game name to category
    game_to_category = {}
    for category, games in categories.items():
        for game in games:
            game_to_category[game] = category
    
    # Save as JSON
    output_data = {
        "categories": categories,
        "game_to_category_mapping": game_to_category,
        "category_count": len(categories),
        "total_games_categorized": sum(len(games) for games in categories.values())
    }
    
    with open("game_categories.json", 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Game categories saved to 'game_categories.json'")
    print(f"This file can be used to add category features to your recommender system!")

if __name__ == "__main__":
    categories = analyze_categorization()
    save_game_categories() 