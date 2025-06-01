import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import pandas as pd

def load_model():
    """Load the sentiment analysis model"""
    model_name = "tabularisai/robust-sentiment-analysis"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_sentiment_score(text, tokenizer, model):
    """
    Predict sentiment score for given text
    Returns score from 1-5 (1=Very Negative, 5=Very Positive)
    """
    if not text or not isinstance(text, str):
        return 3  # Neutral default for empty/invalid text
    
    # Prepare input
    inputs = tokenizer(text.lower(), return_tensors="pt", 
                      truncation=True, padding=True, max_length=512)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities and predicted class
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Convert 0-4 scale to 1-5 scale
    sentiment_score = predicted_class + 1
    confidence = probabilities[0][predicted_class].item()
    
    return sentiment_score, confidence

def score_reviews():
    """Score all reviews in the dataset"""
    print("Loading sentiment analysis model...")
    tokenizer, model = load_model()
    
    print("Loading review data...")
    with open('exports/user_app_review.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract reviews (relationships with review text)
    reviews = []
    for item in data:
        if (item.get('type') == 'relationship' and 
            'properties' in item and 
            isinstance(item['properties'], dict) and
            'review' in item['properties'] and 
            item['properties']['review']):
            reviews.append(item['properties'])
    
    print(f"Found {len(reviews)} reviews to score...")
    
    # Score reviews
    scored_reviews = []
    for review in tqdm(reviews, desc="Scoring reviews"):
        review_text = review['review']
        sentiment_score, confidence = predict_sentiment_score(review_text, tokenizer, model)
        
        # Add scores to review data
        review_with_score = review.copy()
        review_with_score['sentiment_score_1_5'] = sentiment_score
        review_with_score['sentiment_confidence'] = round(confidence, 4)
        
        # Create 1-10 scale by expanding 1-5 scale
        # 1-5 becomes: 1-2, 3-4, 5-6, 7-8, 9-10
        score_1_10 = (sentiment_score - 1) * 2 + 1
        # Add some variance based on confidence
        if confidence > 0.8:  # High confidence
            score_1_10 += 1 if sentiment_score >= 4 else 0
        review_with_score['sentiment_score_1_10'] = min(10, score_1_10)
        
        scored_reviews.append(review_with_score)
    
    # Save scored reviews
    with open('exports/scored_reviews.json', 'w', encoding='utf-8') as f:
        json.dump(scored_reviews, f, indent=2, ensure_ascii=False)
    
    # Create summary statistics
    scores_1_5 = [r['sentiment_score_1_5'] for r in scored_reviews]
    scores_1_10 = [r['sentiment_score_1_10'] for r in scored_reviews]
    voted_up = [r['voted_up'] for r in scored_reviews]
    
    print("\n=== SCORING RESULTS ===")
    print(f"Total reviews scored: {len(scored_reviews)}")
    print(f"\n1-5 Scale Distribution:")
    for i in range(1, 6):
        count = scores_1_5.count(i)
        percentage = (count / len(scores_1_5)) * 100
        print(f"  Score {i}: {count} reviews ({percentage:.1f}%)")
    
    print(f"\n1-10 Scale Distribution:")
    for i in range(1, 11):
        count = scores_1_10.count(i)
        percentage = (count / len(scores_1_10)) * 100
        print(f"  Score {i:2d}: {count} reviews ({percentage:.1f}%)")
    
    # Compare with existing voted_up data
    positive_reviews = sum(voted_up)
    negative_reviews = len(voted_up) - positive_reviews
    print(f"\nComparison with existing voted_up data:")
    print(f"  Positive (voted_up=True): {positive_reviews} ({positive_reviews/len(voted_up)*100:.1f}%)")
    print(f"  Negative (voted_up=False): {negative_reviews} ({negative_reviews/len(voted_up)*100:.1f}%)")
    
    # Correlation analysis
    high_sentiment = sum(1 for s in scores_1_5 if s >= 4)
    low_sentiment = sum(1 for s in scores_1_5 if s <= 2)
    print(f"\nSentiment model results:")
    print(f"  Positive sentiment (4-5): {high_sentiment} ({high_sentiment/len(scores_1_5)*100:.1f}%)")
    print(f"  Negative sentiment (1-2): {low_sentiment} ({low_sentiment/len(scores_1_5)*100:.1f}%)")
    
    print(f"\nScored reviews saved to: exports/scored_reviews.json")
    
    return scored_reviews

if __name__ == "__main__":
    scored_reviews = score_reviews() 