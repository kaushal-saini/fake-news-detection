"""
Prediction Module
=================

This module makes predictions on new, unseen news articles.

Functions:
- predict_news(): Predict if news is real or fake
- predict_batch_news(): Predict multiple articles
- get_prediction_details(): Get detailed prediction info
"""

import numpy as np
from pathlib import Path


def predict_news(news_text, model, vectorizer):
    """
    Predict if a single news article is real or fake.
    
    Parameters:
    -----------
    news_text : str
        News article text to classify
    model : LogisticRegression
        Trained model
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
        
    Returns:
    --------
    dict
        Prediction result with:
        - 'prediction': 0 (Fake) or 1 (Real)
        - 'label': "Fake" or "Real"
        - 'confidence': Confidence score (0-1)
        - 'probabilities': Probability for each class
        
    Example:
    --------
    >>> from src.prediction import predict_news
    >>> from src.model_training import load_model
    >>> from src.feature_extraction import load_vectorizer
    >>> 
    >>> model = load_model('models/fake_news_model.pkl')
    >>> vectorizer = load_vectorizer('models/vectorizer.pkl')
    >>> 
    >>> text = "Breaking news: Scientists discover new treatment for cancer"
    >>> result = predict_news(text, model, vectorizer)
    >>> print(result)
    """
    # Transform text using vectorizer
    X = vectorizer.transform([news_text])
    
    # Get prediction
    prediction = model.predict(X)[0]
    
    # Get probability
    probabilities = model.predict_proba(X)[0]
    confidence = max(probabilities)
    
    # Convert to label
    label = "Real" if prediction == 1 else "Fake"
    
    result = {
        'prediction': prediction,
        'label': label,
        'confidence': confidence,
        'probabilities': {
            'fake': probabilities[0],
            'real': probabilities[1]
        }
    }
    
    return result


def predict_batch_news(news_texts, model, vectorizer):
    """
    Predict for multiple news articles.
    
    Parameters:
    -----------
    news_texts : list
        List of news article texts
    model : LogisticRegression
        Trained model
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
        
    Returns:
    --------
    list
        List of prediction results
        
    Example:
    --------
    >>> texts = ["Article 1", "Article 2", "Article 3"]
    >>> results = predict_batch_news(texts, model, vectorizer)
    """
    results = []
    
    for text in news_texts:
        result = predict_news(text, model, vectorizer)
        results.append(result)
    
    return results


def get_prediction_details(prediction_result):
    """
    Format prediction result for display.
    
    Parameters:
    -----------
    prediction_result : dict
        Result from predict_news()
        
    Returns:
    --------
    str
        Formatted prediction details
        
    Example:
    --------
    >>> details = get_prediction_details(result)
    >>> print(details)
    """
    label = prediction_result['label']
    confidence = prediction_result['confidence']
    prob_fake = prediction_result['probabilities']['fake']
    prob_real = prediction_result['probabilities']['real']
    
    details = f"""
    Prediction: {label}
    Confidence: {confidence:.2%}
    
    Probability Breakdown:
    - Fake: {prob_fake:.2%}
    - Real: {prob_real:.2%}
    """
    
    return details


def display_prediction(news_text, prediction_result, max_length=100):
    """
    Display prediction in a nice format.
    
    Parameters:
    -----------
    news_text : str
        Original news text
    prediction_result : dict
        Result from predict_news()
    max_length : int
        Maximum characters of text to display
        
    Returns:
    --------
    None
        Prints formatted result
    """
    label = prediction_result['label']
    confidence = prediction_result['confidence']
    prob_fake = prediction_result['probabilities']['fake']
    prob_real = prediction_result['probabilities']['real']
    
    # Truncate text if too long
    display_text = news_text[:max_length] + "..." if len(news_text) > max_length else news_text
    
    print("\n" + "="*70)
    print("PREDICTION RESULT")
    print("="*70)
    print(f"\nNews Text:\n{display_text}")
    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nProbability Breakdown:")
    print(f"  Fake: {prob_fake:.2%} {'█' * int(prob_fake * 40)}")
    print(f"  Real: {prob_real:.2%} {'█' * int(prob_real * 40)}")
    print("="*70)


def predict_from_file(filepath, model, vectorizer):
    """
    Predict on news articles from a text file.
    
    File format: One article per line
    
    Parameters:
    -----------
    filepath : str
        Path to text file
    model : LogisticRegression
        Trained model
    vectorizer : TfidfVectorizer
        Fitted TF-IDF vectorizer
        
    Returns:
    --------
    list
        List of prediction results
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = f.readlines()
    
    results = predict_batch_news(articles, model, vectorizer)
    
    return results


if __name__ == "__main__":
    print("Prediction Module")
    print("See main.py for complete usage example")
