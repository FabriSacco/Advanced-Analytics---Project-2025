import json

def verify_app_type_binarization():
    """Verify that app_type has been properly binarized."""
    with open('outputs/recommender_graph_nodes.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    games = []
    dlc = []
    
    for node_id, node_data in data.items():
        if node_data.get('type') == 'App':
            app_name = node_data.get('name', 'Unknown')
            app_type = node_data.get('app_type')
            
            if app_type == 1:
                games.append(app_name)
            elif app_type == 0:
                dlc.append(app_name)
    
    print(f'APP TYPE BINARIZATION VERIFICATION')
    print(f'=' * 50)
    print(f'GAMES (app_type: 1) - Total: {len(games)}')
    for i, game in enumerate(games[:10]):
        print(f'  {i+1}. {game}')
    if len(games) > 10:
        print(f'  ... and {len(games)-10} more games')
    
    print(f'\nDLC (app_type: 0) - Total: {len(dlc)}')
    for i, item in enumerate(dlc):
        print(f'  {i+1}. {item}')
    
    print(f'\nSUMMARY:')
    print(f'- Games: {len(games)} apps (app_type = 1)')
    print(f'- DLC: {len(dlc)} apps (app_type = 0)')
    print(f'- Total: {len(games) + len(dlc)} apps')
    print(f'- Binarization successful: {"✓" if len(games) > 0 and len(dlc) > 0 else "⚠"}')

if __name__ == "__main__":
    verify_app_type_binarization() 