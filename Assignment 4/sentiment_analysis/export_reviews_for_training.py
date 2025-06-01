import json
from datetime import datetime

def export_reviews_for_training():
    """
    Export reviews from user_app_review.json in a clean format for training.
    Creates a JSON file with edge_id and review_text that can be used for training
    and then mapped back to the graph.
    """
    print("Loading review data...")
    with open('exports/user_app_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    training_data = []
    edge_counter = 0
    
    print("Processing reviews...")
    for item in data:
        if (item.get('type') == 'relationship' and 
            item.get('label') == 'REVIEWED' and
            'properties' in item):
            
            props = item['properties']
            
            # Check if review text exists and is not empty
            if ('review' in props and 
                props['review'] and 
                len(str(props['review']).strip()) > 0):
                
                review_text = str(props['review']).strip()
                
                # Create edge record
                edge_record = {
                    'edge_id': edge_counter,
                    'review_text': review_text,
                    'start_node': item.get('start'),  # User ID
                    'end_node': item.get('end'),      # App ID
                    'original_properties': {
                        'voted_up': props.get('voted_up'),
                        'author_playtime_at_review': props.get('author_playtime_at_review'),
                        'language': props.get('language'),
                        'weighted_vote_score': props.get('weighted_vote_score')
                    }
                }
                
                training_data.append(edge_record)
                edge_counter += 1
    
    # Save training data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_file = f'exports/reviews_for_training_{timestamp}.json'
    
    with open(training_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Create a simple version for easier Colab processing
    simple_training_data = []
    for record in training_data:
        simple_record = {
            'edge_id': record['edge_id'],
            'text': record['review_text'],
            'voted_up': record['original_properties']['voted_up']  # For reference
        }
        simple_training_data.append(simple_record)
    
    simple_file = f'exports/reviews_simple_{timestamp}.json'
    with open(simple_file, 'w', encoding='utf-8') as f:
        json.dump(simple_training_data, f, indent=2, ensure_ascii=False)
    
    # Save mapping file for importing scores back
    mapping_data = {
        'export_timestamp': timestamp,
        'total_reviews': len(training_data),
        'edge_mappings': []
    }
    
    for record in training_data:
        mapping_data['edge_mappings'].append({
            'edge_id': record['edge_id'],
            'start_node': record['start_node'],
            'end_node': record['end_node']
        })
    
    mapping_file = f'exports/edge_mapping_{timestamp}.json'
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== EXPORT COMPLETE ===")
    print(f"Total reviews exported: {len(training_data)}")
    print(f"\nFiles created:")
    print(f"1. {training_file}")
    print(f"   → Full data with all metadata for training")
    print(f"2. {simple_file}")
    print(f"   → Simple format (edge_id + text) for easy Colab processing")
    print(f"3. {mapping_file}")
    print(f"   → Mapping file to import scores back to graph")
    
    # Show sample data
    print(f"\nSample records:")
    for i, sample in enumerate(training_data[:3]):
        print(f"  {i+1}. Edge ID: {sample['edge_id']}")
        print(f"     User: {sample['start_node']} → App: {sample['end_node']}")
        print(f"     Text: {sample['review_text'][:80]}...")
        print(f"     Voted up: {sample['original_properties']['voted_up']}")
        print()
    
    return training_file, simple_file, mapping_file

if __name__ == "__main__":
    export_reviews_for_training() 