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
    Predict whether a news article is real or fake with confidence score.
    
    This function takes raw news text and produces a binary classification
    with a confidence percentage, useful for real-world deployment.
    
    Prediction Pipeline:
    ====================
    1. TEXT PREPROCESSING
       - Input: Raw news article text (any format)
       - Output: Cleaned text (normalized, no URLs, no stopwords)
       - Uses: Same preprocessing as training for consistency
    
    2. VECTORIZATION
       - Input: Cleaned text
       - Output: TF-IDF feature vector (5000 features)
       - Uses: Fitted vectorizer from training
    
    3. CLASSIFICATION
       - Input: Feature vector
       - Output: Logistic Regression probability (0-1)
       - 0.0 = Definitely Fake
       - 0.5 = Uncertain (borderline)
       - 1.0 = Definitely Real
    
    4. CONFIDENCE CALCULATION
       - Confidence = max(prob_fake, prob_real) × 100
       - Example: prob=[0.2, 0.8] → Real with 80% confidence
    
    Decision Logic:
    ================
    if probability > 0.5:
        → Predicted as REAL
    else:
        → Predicted as FAKE
    
    Confidence Interpretation:
    ============================
    90-100%: Very confident prediction, reliable
    75-89%:  Confident prediction, generally reliable
    60-74%:  Moderate confidence, use with caution
    50-59%:  Low confidence, borderline case
    
    Parameters:
    -----------
    news_text : str
        Raw news article text to classify
        - Can be any length (short headline to full article)
        - Any format (plain text OK)
        - Non-English text may have reduced accuracy
        
    model : LogisticRegression
        Pre-trained model (use load_model())
        - Must be trained on WELFake dataset
        - Specific to fake news classification
        
    vectorizer : TfidfVectorizer
        Pre-fitted vectorizer (use load_vectorizer())
        - Must be fitted with same training data as model
        - Ensures consistent feature representation
        
    Returns:
    --------
    dict
        Result dictionary containing:
        - 'prediction': 0 (Fake) or 1 (Real)
        - 'label': "Fake" or "Real" (human readable)
        - 'confidence': Confidence percentage (0-100)
        - 'probabilities': {'fake': float, 'real': float}
        
    Example:
    --------
    >>> from src.model_training import load_model
    >>> from src.feature_extraction import load_vectorizer
    >>> from src.prediction import predict_news
    >>>
    >>> model = load_model('models/fake_news_model.pkl')
    >>> vectorizer = load_vectorizer('models/vectorizer.pkl')
    >>>
    >>> # Example 1: Real news
    >>> text1 = "Scientists discover breakthrough in cancer treatment"
    >>> result1 = predict_news(text1, model, vectorizer)
    >>> # Output: {'prediction': 1, 'label': 'Real', 'confidence': 96, ...}
    >>>
    >>> # Example 2: Likely fake
    >>> text2 = "FAKE: Celebrity secretly joins alien colony"
    >>> result2 = predict_news(text2, model, vectorizer)
    >>> # Output: {'prediction': 0, 'label': 'Fake', 'confidence': 94, ...}
    
    Usage Tips:
    ===========
    1. Always use the same vectorizer and model together
    2. For batch predictions, use predict_batch_news() instead
    3. High confidence (>80%) predictions are more reliable
    4. Borderline cases (50-60%) require additional review
    5. Consider article context when reviewing low-confidence predictions
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
