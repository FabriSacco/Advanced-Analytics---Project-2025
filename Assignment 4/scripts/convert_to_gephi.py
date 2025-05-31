import json
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time

def create_gephi_csv_files():
    """Create Gephi-compatible CSV files for both subgraphs."""
    print("=== Creating Gephi-Compatible CSV Files ===")
    
    # Convert User-User Friendship Graph
    print("\n1. Converting User-User Friendship Graph...")
    
    # Load user nodes
    with open("outputs/user_user_graph_nodes.json", 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    # Create nodes CSV for User-User graph
    user_nodes_csv = "outputs/gephi_user_user_nodes.csv"
    with open(user_nodes_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label', 'personaname', 'loccountrycode', 'original_id'])
        
        for user_id, user_data in users.items():
            writer.writerow([
                user_id,
                user_data.get('personaname', 'Unknown'),
                user_data.get('personaname', 'Unknown'),
                user_data.get('loccountrycode', 'Unknown'),
                user_data.get('original_id', user_id)
            ])
    
    print(f"  User nodes saved to {user_nodes_csv}")
    
    # Create edges CSV for User-User graph
    user_edges_csv = "outputs/gephi_user_user_edges.csv"
    with open(user_edges_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Type', 'Weight'])
        
        # Read friendship edges
        edge_count = 0
        with open("outputs/user_user_graph_edges.txt", 'r', encoding='utf-8') as edges_file:
            next(edges_file)  # Skip header
            for line in edges_file:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    source_id = parts[0]
                    target_id = parts[1]
                    writer.writerow([source_id, target_id, 'Undirected', 1.0])
                    edge_count += 1
                    
                    if edge_count % 500000 == 0:
                        print(f"    Processed {edge_count:,} friendship edges")
    
    print(f"  Friendship edges saved to {user_edges_csv}")
    print(f"  Total friendship edges: {edge_count:,}")
    
    # Convert User-App Review Graph
    print("\n2. Converting User-App Review Graph...")
    
    # Load all nodes (users + apps)
    with open("outputs/user_app_graph_nodes.json", 'r', encoding='utf-8') as f:
        all_nodes = json.load(f)
    
    # Create nodes CSV for User-App graph
    user_app_nodes_csv = "outputs/gephi_user_app_nodes.csv"
    with open(user_app_nodes_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label', 'Type', 'personaname', 'loccountrycode', 'app_name', 'category', 'app_type', 'original_id'])
        
        for node_id, node_data in all_nodes.items():
            if node_data['type'] == 'User':
                writer.writerow([
                    node_id,
                    node_data.get('personaname', 'Unknown'),
                    'User',
                    node_data.get('personaname', 'Unknown'),
                    node_data.get('loccountrycode', 'Unknown'),
                    '',  # app_name (empty for users)
                    '',  # category (empty for users)
                    '',  # app_type (empty for users)
                    node_data.get('original_id', node_id)
                ])
            else:  # App
                writer.writerow([
                    node_id,
                    node_data.get('name', 'Unknown'),
                    'App',
                    '',  # personaname (empty for apps)
                    '',  # loccountrycode (empty for apps)
                    node_data.get('name', 'Unknown'),
                    node_data.get('category', 'Uncategorized'),
                    node_data.get('app_type', 1),
                    node_data.get('original_id', node_id)
                ])
    
    print(f"  User+App nodes saved to {user_app_nodes_csv}")
    
    # Create edges CSV for User-App graph
    user_app_edges_csv = "outputs/gephi_user_app_edges.csv"
    with open(user_app_edges_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Source', 'Target', 'Type', 'Weight', 'sentiment_score'])
        
        # Read review edges
        edge_count = 0
        with open("outputs/user_app_graph_edges.txt", 'r', encoding='utf-8') as edges_file:
            next(edges_file)  # Skip header
            for line in edges_file:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    user_id = parts[0]
                    app_id = parts[1]
                    weight = parts[2]
                    sentiment = parts[3] if parts[3] != '' else ''
                    writer.writerow([user_id, app_id, 'Undirected', weight, sentiment])
                    edge_count += 1
    
    print(f"  Review edges saved to {user_app_edges_csv}")
    print(f"  Total review edges: {edge_count:,}")
    
    return {
        'user_user': {
            'nodes': user_nodes_csv,
            'edges': user_edges_csv,
            'node_count': len(users),
            'edge_count': edge_count
        },
        'user_app': {
            'nodes': user_app_nodes_csv,
            'edges': user_app_edges_csv,
            'node_count': len(all_nodes),
            'edge_count': edge_count
        }
    }

def create_gexf_file(graph_type, nodes_file, edges_file, output_file):
    """Create a GEXF file for direct import into Gephi."""
    print(f"\nCreating GEXF file for {graph_type} graph...")
    
    # Create root element
    gexf = ET.Element("gexf", version="1.2")
    gexf.set("xmlns", "http://www.gexf.net/1.2draft")
    gexf.set("xmlns:viz", "http://www.gexf.net/1.2draft/viz")
    
    # Add meta information
    meta = ET.SubElement(gexf, "meta")
    meta.set("lastmodifieddate", time.strftime("%Y-%m-%d"))
    creator = ET.SubElement(meta, "creator")
    creator.text = "Steam Graph Analysis"
    description = ET.SubElement(meta, "description")
    description.text = f"Steam {graph_type} Graph"
    
    # Create graph element
    graph = ET.SubElement(gexf, "graph")
    graph.set("mode", "static")
    graph.set("defaultedgetype", "undirected")
    
    # Define attributes for nodes
    attributes = ET.SubElement(graph, "attributes")
    attributes.set("class", "node")
    
    if graph_type == "User-User":
        # User attributes
        ET.SubElement(attributes, "attribute", id="0", title="personaname", type="string")
        ET.SubElement(attributes, "attribute", id="1", title="loccountrycode", type="string")
        ET.SubElement(attributes, "attribute", id="2", title="original_id", type="string")
    else:  # User-App
        # Mixed attributes
        ET.SubElement(attributes, "attribute", id="0", title="type", type="string")
        ET.SubElement(attributes, "attribute", id="1", title="personaname", type="string")
        ET.SubElement(attributes, "attribute", id="2", title="loccountrycode", type="string")
        ET.SubElement(attributes, "attribute", id="3", title="app_name", type="string")
        ET.SubElement(attributes, "attribute", id="4", title="category", type="string")
        ET.SubElement(attributes, "attribute", id="5", title="app_type", type="integer")
        ET.SubElement(attributes, "attribute", id="6", title="original_id", type="string")
    
    # Edge attributes
    edge_attributes = ET.SubElement(graph, "attributes")
    edge_attributes.set("class", "edge")
    if graph_type == "User-App":
        ET.SubElement(edge_attributes, "attribute", id="0", title="sentiment_score", type="float")
    
    # Add nodes
    nodes_elem = ET.SubElement(graph, "nodes")
    node_count = 0
    
    with open(nodes_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = ET.SubElement(nodes_elem, "node")
            node.set("id", row['Id'])
            node.set("label", row['Label'])
            
            # Add attribute values
            attvalues = ET.SubElement(node, "attvalues")
            
            if graph_type == "User-User":
                ET.SubElement(attvalues, "attvalue", attfor="0", value=row['personaname'])
                ET.SubElement(attvalues, "attvalue", attfor="1", value=row['loccountrycode'])
                ET.SubElement(attvalues, "attvalue", attfor="2", value=row['original_id'])
            else:  # User-App
                ET.SubElement(attvalues, "attvalue", attfor="0", value=row['Type'])
                ET.SubElement(attvalues, "attvalue", attfor="1", value=row['personaname'])
                ET.SubElement(attvalues, "attvalue", attfor="2", value=row['loccountrycode'])
                ET.SubElement(attvalues, "attvalue", attfor="3", value=row['app_name'])
                ET.SubElement(attvalues, "attvalue", attfor="4", value=row['category'])
                ET.SubElement(attvalues, "attvalue", attfor="5", value=row['app_type'])
                ET.SubElement(attvalues, "attvalue", attfor="6", value=row['original_id'])
            
            node_count += 1
            if node_count % 100000 == 0:
                print(f"  Added {node_count:,} nodes to GEXF")
    
    # Add edges
    edges_elem = ET.SubElement(graph, "edges")
    edge_count = 0
    
    with open(edges_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge = ET.SubElement(edges_elem, "edge")
            edge.set("id", str(edge_count))
            edge.set("source", row['Source'])
            edge.set("target", row['Target'])
            edge.set("weight", row['Weight'])
            
            # Add edge attributes for User-App graph
            if graph_type == "User-App" and 'sentiment_score' in row and row['sentiment_score']:
                attvalues = ET.SubElement(edge, "attvalues")
                ET.SubElement(attvalues, "attvalue", attfor="0", value=row['sentiment_score'])
            
            edge_count += 1
            if edge_count % 100000 == 0:
                print(f"  Added {edge_count:,} edges to GEXF")
    
    # Write GEXF file
    rough_string = ET.tostring(gexf, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    print(f"  GEXF saved to {output_file}")
    print(f"  Nodes: {node_count:,}, Edges: {edge_count:,}")

def main():
    """Main function to convert graphs to Gephi formats."""
    print("=== Converting Graphs to Gephi Formats ===")
    print("This will create both CSV and GEXF files for Gephi import")
    
    start_time = time.time()
    
    try:
        # Create CSV files
        csv_info = create_gephi_csv_files()
        
        # Create GEXF files (more compact and feature-rich)
        print("\n" + "="*50)
        print("CREATING GEXF FILES (RECOMMENDED FOR GEPHI)")
        print("="*50)
        
        # User-User GEXF
        create_gexf_file(
            "User-User",
            csv_info['user_user']['nodes'],
            csv_info['user_user']['edges'],
            "outputs/gephi_user_user_graph.gexf"
        )
        
        # User-App GEXF (smaller, more manageable)
        create_gexf_file(
            "User-App",
            csv_info['user_app']['nodes'],
            csv_info['user_app']['edges'],
            "outputs/gephi_user_app_graph.gexf"
        )
        
        # Summary
        print("\n" + "="*60)
        print("GEPHI CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nFiles created for Gephi:")
        print(f"\n1. User-User Friendship Graph:")
        print(f"   CSV: gephi_user_user_nodes.csv + gephi_user_user_edges.csv")
        print(f"   GEXF: gephi_user_user_graph.gexf (RECOMMENDED)")
        print(f"   Size: {csv_info['user_user']['node_count']:,} users, {csv_info['user_user']['edge_count']:,} friendships")
        
        print(f"\n2. User-App Review Graph:")
        print(f"   CSV: gephi_user_app_nodes.csv + gephi_user_app_edges.csv")
        print(f"   GEXF: gephi_user_app_graph.gexf (RECOMMENDED)")
        print(f"   Size: {csv_info['user_app']['node_count']:,} nodes, {csv_info['user_app']['edge_count']:,} reviews")
        
        total_time = time.time() - start_time
        print(f"\nTotal conversion time: {total_time:.2f} seconds")
        
        print(f"\n📋 GEPHI IMPORT INSTRUCTIONS:")
        print(f"   1. Open Gephi")
        print(f"   2. File → Open → Select .gexf file (RECOMMENDED)")
        print(f"   3. OR File → Import Spreadsheet → Select nodes CSV, then edges CSV")
        print(f"   4. Choose 'Undirected' graph type")
        print(f"   5. Use weight for edge thickness, sentiment for edge colors")
        print(f"   6. Use node type/category for node colors and sizes")
        
        print(f"\n⚠️  WARNING: User-User graph is very large (3.6M edges)")
        print(f"   Consider using Gephi's filtering tools or sampling for visualization")
        print(f"   User-App graph is more manageable for visualization (30K edges)")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        print("Please run extract_subgraphs.py first")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 