"""
Model Training Module
======================

This module handles training the Logistic Regression model.

Logistic Regression:
- Binary classification algorithm
- Outputs probability between 0 and 1
- Fast and interpretable
- Works well with TF-IDF features

Functions:
- train_model(): Train Logistic Regression on training data
- save_model(): Save trained model to file
- load_model(): Load trained model from file
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model(X_train, y_train, random_state=42, max_iter=1000):
    """
    Train a Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like or sparse matrix
        Training features (TF-IDF vectors)
    y_train : array-like
        Training labels (0 for fake, 1 for real)
    random_state : int (default=42)
        Random seed for reproducibility
    max_iter : int (default=1000)
        Maximum number of iterations for convergence
        
    Returns:
    --------
    LogisticRegression
        Trained model
        
    Example:
    --------
    >>> model = train_model(X_train, y_train)
    
    Explanation:
    ---------
    Logistic Regression:
    - Uses sigmoid function to map features to probabilities
    - Probability close to 0 = Fake news
    - Probability close to 1 = Real news
    - Threshold 0.5 is typically used (prob > 0.5 = Real)
    """
    print("Training Logistic Regression Model...")
    print("This may take a moment depending on dataset size...")
    
    # Create and train model
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver='lbfgs',              # Solver algorithm
        verbose=1                     # Show training progress
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    print("\n✓ Model trained successfully!")
    print(f"  - Number of iterations: {model.n_iter_[0]}")
    print(f"  - Coefficients shape: {model.coef_.shape}")
    print(f"  - Classes: {model.classes_}")
    
    return model


def save_model(model, filepath):
    """
    Save trained model to file using pickle.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    filepath : str
        Path to save model
        
    Returns:
    --------
    None
        
    Example:
    --------
    >>> save_model(model, 'models/fake_news_model.pkl')
    """
    print(f"\nSaving model to {filepath}...")
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved successfully!")


def load_model(filepath):
    """
    Load trained model from file.
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
        
    Returns:
    --------
    LogisticRegression
        Loaded model
        
    Example:
    --------
    >>> model = load_model('models/fake_news_model.pkl')
    """
    print(f"Loading model from {filepath}...")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✓ Model loaded successfully!")
    
    return model


def predict_single(model, X_single):
    """
    Make a prediction for a single sample.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X_single : array-like or sparse matrix
        Single feature vector
        
    Returns:
    --------
    int
        Predicted label (0 or 1)
    """
    prediction = model.predict(X_single)
    return prediction[0]


def predict_batch(model, X):
    """
    Make predictions for multiple samples.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X : array-like or sparse matrix
        Feature vectors
        
    Returns:
    --------
    np.ndarray
        Predicted labels
    """
    predictions = model.predict(X)
    return predictions


def predict_proba(model, X):
    """
    Get prediction probabilities.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
    X : array-like or sparse matrix
        Feature vectors
        
    Returns:
    --------
    np.ndarray
        Probabilities for each class
        - Column 0: Probability of fake (class 0)
        - Column 1: Probability of real (class 1)
        
    Example:
    --------
    >>> probabilities = predict_proba(model, X_test)
    >>> print(probabilities[0])  # [0.2, 0.8] = 20% fake, 80% real
    """
    probabilities = model.predict_proba(X)
    return probabilities


def get_model_info(model):
    """
    Get information about the trained model.
    
    Parameters:
    -----------
    model : LogisticRegression
        Trained model
        
    Returns:
    --------
    dict
        Model information
    """
    info = {
        'n_features': model.n_features_in_,
        'n_classes': len(model.classes_),
        'classes': model.classes_,
        'coefficients_shape': model.coef_.shape,
        'intercept': model.intercept_[0],
        'solver': model.solver,
        'max_iterations': model.max_iter,
        'n_iterations': model.n_iter_[0] if hasattr(model, 'n_iter_') else None
    }
    
    return info


if __name__ == "__main__":
    # Example would require actual training data
    print("Model Training Module")
    print("See main.py for complete training pipeline")
