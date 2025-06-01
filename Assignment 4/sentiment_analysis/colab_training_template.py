# Google Colab Training Template for Steam Game Review Sentiment Analysis
# Upload this as a .ipynb file to Google Colab

"""
# Steam Game Review Sentiment Analysis Training

## Step 1: Setup Environment
"""

# Install required packages
# !pip install transformers torch datasets accelerate

import json
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

"""
## Step 2: Upload and Load Data

Upload these files to Colab:
1. reviews_simple_20250529_162404.json
2. edge_mapping_20250529_162404.json
"""

# Load the review data
with open('reviews_simple_20250529_162404.json', 'r', encoding='utf-8') as f:
    reviews_data = json.load(f)

print(f"Loaded {len(reviews_data)} reviews")

# Convert to DataFrame for easier handling
df = pd.DataFrame(reviews_data)
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check class distribution
print(f"\nClass distribution:")
print(df['voted_up'].value_counts())

"""
## Step 3: Prepare Data for Training
"""

# Convert voted_up to sentiment labels
# True = Positive (4-5 stars), False = Negative (1-2 stars)
# We'll create a 5-point scale:
def voted_up_to_sentiment(voted_up):
    """Convert voted_up boolean to 5-point sentiment scale"""
    if voted_up:
        return 4  # Positive (4-5 range)
    else:
        return 1  # Negative (1-2 range)

df['sentiment_label'] = df['voted_up'].apply(voted_up_to_sentiment)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['sentiment_label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment_label']
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

"""
## Step 4: Load and Setup Model
"""

# Choose model - update this based on your preference
MODEL_NAME = "LiYuan/amazon-review-sentiment-analysis"  # Best for reviews


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=5,  # 1-5 star rating
    problem_type="single_label_classification"
)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

# Create datasets
train_dataset = Dataset.from_dict({
    'text': train_texts,
    'labels': train_labels
})

val_dataset = Dataset.from_dict({
    'text': val_texts,
    'labels': val_labels
})

# Tokenize
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

"""
## Step 5: Training Configuration
"""

# Training arguments
training_args = TrainingArguments(
    output_dir='./steam-review-sentiment',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Adjust based on GPU memory
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    fp16=True,  # Mixed precision for L4 GPU
    dataloader_pin_memory=True,
    gradient_accumulation_steps=2,
)

# Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

"""
## Step 6: Train the Model
"""

# Start training
print("Starting training...")
trainer.train()

# Evaluate
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

"""
## Step 7: Make Predictions on Full Dataset
"""

# Load the full dataset for prediction
all_texts = df['text'].tolist()
all_edge_ids = df['edge_id'].tolist()

# Make predictions
def predict_sentiment_scores(texts, model, tokenizer, batch_size=32):
    model.eval()
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            truncation=True, 
            padding=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Convert to 1-5 scale scores
            probabilities = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1)
            
            # Convert to 1-5 scale (adjust as needed)
            scores = predicted_classes + 1  # Assuming 0-4 -> 1-5
            
            predictions.extend(scores.cpu().numpy().tolist())
    
    return predictions

print("Making predictions on full dataset...")
sentiment_scores = predict_sentiment_scores(all_texts, model, tokenizer)

# Create results dataset
results = []
for edge_id, score in zip(all_edge_ids, sentiment_scores):
    results.append({
        'edge_id': edge_id,
        'sentiment_score': score
    })

"""
## Step 8: Save Results
"""

# Save predictions
with open('steam_review_sentiment_scores.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved {len(results)} sentiment scores to steam_review_sentiment_scores.json")

# Download files
from google.colab import files
files.download('steam_review_sentiment_scores.json')
files.download('edge_mapping_20250529_162404.json')  # Download mapping file too

print("Files ready for download!")
print("Next step: Use import_scores_to_graph.py to add scores to your graph")

"""
## Summary

1. Trained sentiment analysis model on your Steam game reviews
2. Generated sentiment scores (1-5 scale) for all reviews
3. Created steam_review_sentiment_scores.json with edge_id -> score mapping
4. Use import_scores_to_graph.py locally to add scores to your NetworkX graph

Example usage back in your local environment:
```python
from import_scores_to_graph import import_scores_to_graph

import_scores_to_graph(
    'steam_review_sentiment_scores.json',
    'edge_mapping_20250529_162404.json'
)
```
""" 