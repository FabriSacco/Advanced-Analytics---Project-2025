import os
import requests
import feedparser
import json
import time

# --------------------------------------
# CONFIGURATION
# --------------------------------------
save_dir = r"D:\Downloads\spark\spark\spark-3.5.5-bin-hadoop3\saved_data"  # Full path for saving data
batch_size = 100          # Number of articles per file
total_articles = 1000     # Total articles to fetch
label = "stat.TH"              # Label for file naming and filtering (e.g. "cs", "physics", "math", etc.) - change for each subcategory
query = f"cat:{label}"    # Use category-based query for accurate filtering
arxiv_api = "http://export.arxiv.org/api/query"
# --------------------------------------

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

def fetch_arxiv_articles(start_index, max_results, search_query):
    params = {
        "search_query": search_query,
        "start": start_index,
        "max_results": max_results
    }
    response = requests.get(arxiv_api, params=params)
    if response.status_code != 200:
        raise Exception(f"Arxiv API returned status {response.status_code}")
    feed = feedparser.parse(response.text)
    return feed.entries

def parse_entry(entry):
    """Parse a single Arxiv entry into a clean JSON-like dictionary."""
    categories = [tag['term'] for tag in entry.tags] if 'tags' in entry else []
    return {
        "aid": entry.id,
        "title": entry.title,
        "summary": entry.summary,
        "published": entry.published,
        "categories": ",".join(categories),
        "main_category": categories[0] if categories else None
    }

def save_batch(batch, batch_idx, label):
    """Save a batch of articles with a label in the filename."""
    filename = os.path.join(save_dir, f"{label}_batch_{batch_idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        for article in batch:
            f.write(json.dumps(article) + "\n")
    print(f"Saved batch {batch_idx} ({len(batch)} articles) to {filename}")

# Main loop
batch = []
batch_counter = 0
start_index = 0

print(f"\nStarting download for category: {label}...\n")

while start_index < total_articles:
    print(f"Fetching articles {start_index} to {start_index + batch_size}...")
    entries = fetch_arxiv_articles(start_index, batch_size, search_query=query)

    if not entries:
        print("No more entries returned. Stopping.")
        break

    for entry in entries:
        article = parse_entry(entry)

        # Filter to only include articles matching the label in main_category
        if not article["main_category"] or not article["main_category"].startswith(label):
            continue

        batch.append(article)

        if len(batch) >= batch_size:
            save_batch(batch, batch_counter, label)
            batch = []
            batch_counter += 1

    start_index += batch_size
    time.sleep(1)  # Be polite to the API

# Save any remaining articles
if batch:
    save_batch(batch, batch_counter, label)

print(f"\nFinished downloading {label} articles!\n")

