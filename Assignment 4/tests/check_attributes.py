import json

with open('outputs/final_recommender_graph_nodes.json', 'r', encoding='utf-8') as f:
    nodes = json.load(f)

# Sample user attributes
user_sample = None
app_sample = None
for nid, data in list(nodes.items())[:20]:
    if data['type'] == 'User' and user_sample is None:
        user_sample = data
        print('Sample User attributes:')
        for k, v in data.items():
            print(f'  {k}: {v}')
        break

# Find an app with all attributes
print('\nLooking for complete app samples...')
app_samples = []
for nid, data in list(nodes.items()):
    if data['type'] == 'App':
        app_samples.append(data)
        if len(app_samples) >= 5:
            break

for i, app in enumerate(app_samples):
    print(f'\nApp sample {i+1}:')
    for k, v in app.items():
        print(f'  {k}: {v}')

# Count unique values for categorical attributes
print('\n' + '='*50)
print('ATTRIBUTE ANALYSIS')
print('='*50)

# User attributes analysis
user_countries = set()
user_count = 0
for nid, data in nodes.items():
    if data['type'] == 'User':
        user_count += 1
        if 'loccountrycode' in data and data['loccountrycode']:
            user_countries.add(data['loccountrycode'])

print(f'\nUser countries: {len(user_countries)} unique values out of {user_count} users')
print(f'Sample countries: {list(user_countries)[:10]}')

# App attributes analysis
app_categories = set()
app_types = set()
is_free_values = set()
app_count = 0
category_count = 0
type_count = 0
free_count = 0

for nid, data in nodes.items():
    if data['type'] == 'App':
        app_count += 1
        if 'category' in data and data['category']:
            app_categories.add(data['category'])
            category_count += 1
        if 'app_type' in data and data['app_type'] is not None:
            app_types.add(data['app_type'])
            type_count += 1
        if 'is_free' in data and data['is_free'] is not None:
            is_free_values.add(data['is_free'])
            free_count += 1

print(f'\nApp categories: {len(app_categories)} unique values ({category_count}/{app_count} apps have category)')
print(f'Categories: {list(app_categories)}')

print(f'\nApp types: {len(app_types)} unique values ({type_count}/{app_count} apps have type)')
print(f'Types: {list(app_types)}')

print(f'\nIs free values: {len(is_free_values)} unique values ({free_count}/{app_count} apps have is_free)')
print(f'Values: {list(is_free_values)}')

# Check if there are other attributes
print(f'\nAll app attribute keys found:')
all_app_keys = set()
for nid, data in nodes.items():
    if data['type'] == 'App':
        all_app_keys.update(data.keys())
print(f'Keys: {sorted(all_app_keys)}') 