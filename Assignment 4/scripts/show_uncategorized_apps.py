#!/usr/bin/env python3
"""
Show Uncategorized Apps
======================

This script identifies and prints all apps that are marked as "Uncategorized"
in the graph nodes file.
"""

import json

def show_uncategorized_apps():
    """Load and display all uncategorized apps."""
    
    print("="*60)
    print("UNCATEGORIZED APPS ANALYSIS")
    print("="*60)
    
    # Load the nodes file
    with open("outputs/recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # Find uncategorized apps
    uncategorized_apps = []
    categorized_apps = []
    
    for node_id, attributes in nodes.items():
        if attributes.get('type') == 'App':
            app_name = attributes.get('name', 'Unknown')
            category = attributes.get('category', 'Unknown')
            app_type = attributes.get('app_type', 'Unknown')
            
            if category == 'Uncategorized':
                uncategorized_apps.append({
                    'node_id': node_id,
                    'name': app_name,
                    'app_type': app_type,
                    'original_id': attributes.get('original_id', 'Unknown')
                })
            else:
                categorized_apps.append({
                    'name': app_name,
                    'category': category
                })
    
    print(f"Total Apps: {len(uncategorized_apps) + len(categorized_apps)}")
    print(f"Categorized Apps: {len(categorized_apps)}")
    print(f"Uncategorized Apps: {len(uncategorized_apps)}")
    print(f"Categorization Rate: {len(categorized_apps)/(len(uncategorized_apps) + len(categorized_apps))*100:.1f}%")
    
    if uncategorized_apps:
        print(f"\n{'='*40}")
        print("UNCATEGORIZED APPS LIST")
        print("="*40)
        
        print(f"{'#':<3} {'Node ID':<10} {'Original ID':<12} {'App Type':<8} {'App Name'}")
        print("-" * 80)
        
        for i, app in enumerate(uncategorized_apps, 1):
            print(f"{i:<3} {app['node_id']:<10} {app['original_id']:<12} {app['app_type']:<8} {app['name']}")
        
        # Show app types distribution for uncategorized
        app_type_counts = {}
        for app in uncategorized_apps:
            app_type = app['app_type']
            app_type_counts[app_type] = app_type_counts.get(app_type, 0) + 1
        
        print(f"\n{'='*40}")
        print("UNCATEGORIZED APPS BY TYPE")
        print("="*40)
        for app_type, count in app_type_counts.items():
            percentage = (count / len(uncategorized_apps)) * 100
            print(f"{app_type}: {count} apps ({percentage:.1f}%)")
    
    # Also show a sample of categorized apps for comparison
    print(f"\n{'='*40}")
    print("SAMPLE CATEGORIZED APPS")
    print("="*40)
    
    # Group by category
    category_samples = {}
    for app in categorized_apps:
        category = app['category']
        if category not in category_samples:
            category_samples[category] = []
        category_samples[category].append(app['name'])
    
    for category, app_names in category_samples.items():
        print(f"\n{category} ({len(app_names)} apps):")
        # Show first 3 apps in each category
        for name in app_names[:3]:
            print(f"  - {name}")
        if len(app_names) > 3:
            print(f"  ... and {len(app_names) - 3} more")

def check_missing_from_categories_file():
    """Check which uncategorized apps are missing from the game_categories.json file."""
    
    print(f"\n{'='*60}")
    print("MISSING FROM CATEGORIES FILE ANALYSIS")
    print("="*60)
    
    # Load game categories
    try:
        with open("outputs/game_categories.json", 'r', encoding='utf-8') as f:
            categories_data = json.load(f)
        
        game_to_category = categories_data.get("game_to_category_mapping", {})
        print(f"Games in categories file: {len(game_to_category)}")
        
    except FileNotFoundError:
        print("Error: outputs/game_categories.json not found")
        return
    
    # Load nodes to get uncategorized apps
    with open("outputs/recommender_graph_nodes.json", 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    uncategorized_apps = []
    for node_id, attributes in nodes.items():
        if (attributes.get('type') == 'App' and 
            attributes.get('category') == 'Uncategorized'):
            uncategorized_apps.append(attributes.get('name', 'Unknown'))
    
    print(f"\nUncategorized apps missing from categories file:")
    print("-" * 50)
    
    for i, app_name in enumerate(uncategorized_apps, 1):
        if app_name in game_to_category:
            print(f"{i:2d}. {app_name} - ⚠️  EXISTS in categories file but marked uncategorized!")
        else:
            print(f"{i:2d}. {app_name} - ❌ Missing from categories file")
    
    # Check for case sensitivity issues
    print(f"\n{'='*40}")
    print("POTENTIAL CASE SENSITIVITY ISSUES")
    print("="*40)
    
    categories_lower = {name.lower(): name for name in game_to_category.keys()}
    
    for app_name in uncategorized_apps:
        if app_name.lower() in categories_lower:
            original_name = categories_lower[app_name.lower()]
            print(f"'{app_name}' vs '{original_name}' - Case mismatch!")

def main():
    """Main function to analyze uncategorized apps."""
    
    print("UNCATEGORIZED APPS ANALYSIS")
    print("Analyzing apps without category mappings...\n")
    
    try:
        show_uncategorized_apps()
        check_missing_from_categories_file()
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nNext steps:")
        print("1. Add missing apps to outputs/game_categories.json")
        print("2. Fix any case sensitivity issues")
        print("3. Re-run the graph construction script")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 