"""
Feature Extraction Module
==========================

This module handles feature extraction using TF-IDF Vectorizer.

TF-IDF converts text to numerical features by calculating:
- TF (Term Frequency): How often a word appears in a document
- IDF (Inverse Document Frequency): How rare a word is across all documents

Functions:
- create_tfidf_vectorizer(): Create and configure TF-IDF vectorizer
- fit_and_transform(): Fit vectorizer on training data and transform
- transform(): Transform data using fitted vectorizer
- get_feature_names(): Get feature names from vectorizer
"""

import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path


def create_tfidf_vectorizer(max_features=5000, min_df=5, max_df=0.7, ngram_range=(1, 2)):
    """
    Create and configure a TF-IDF Vectorizer.
    
    Parameters:
    -----------
    max_features : int (default=5000)
        Maximum number of features (words) to keep
        - Reduces dimensionality
        - Improves performance
        - Lower values = faster, less memory
        
    min_df : int (default=5)
        Minimum document frequency
        - Ignore words appearing in less than 5 documents
        - Removes rare, noisy words
        
    max_df : float (default=0.7)
        Maximum document frequency (as proportion)
        - Ignore words appearing in more than 70% of documents
        - Removes very common words that don't help classification
        
    ngram_range : tuple (default=(1, 2))
        Range of n-grams to consider
        - (1, 2): Consider both individual words (unigrams) and two-word phrases (bigrams)
        - (1, 1): Only individual words
        - (1, 3): Individual words, 2-word phrases, and 3-word phrases
        
    Returns:
    --------
    TfidfVectorizer
        Configured vectorizer object
        
    Example:
    --------
    >>> vectorizer = create_tfidf_vectorizer(max_features=3000)
    """
    print("Creating TF-IDF Vectorizer...")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,      # Keep top 5000 words
        min_df=min_df,                  # Word must appear in at least 5 documents
        max_df=max_df,                  # Word can appear in at most 70% of documents
        ngram_range=ngram_range,        # Use 1-word and 2-word combinations
        lowercase=True,                 # Convert to lowercase
        stop_words='english'            # Remove English stopwords
    )
    
    print(f"✓ Vectorizer created with max_features={max_features}")
    print(f"  - min_df={min_df}")
    print(f"  - max_df={max_df}")
    print(f"  - ngram_range={ngram_range}")
    
    return vectorizer


def fit_and_transform(vectorizer, texts):
    """
    Fit vectorizer on data and transform to TF-IDF features.
    
    This is used for TRAINING DATA.
    The vectorizer learns the vocabulary from training data.
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Unfitted vectorizer
    texts : array-like
        Training text data
        
    Returns:
    --------
    sparse matrix
        TF-IDF transformed features
        
    Example:
    --------
    >>> X_train = fit_and_transform(vectorizer, train_texts)
    """
    print("\nFitting TF-IDF Vectorizer on training data...")
    
    # Fit on training data and transform
    X_transformed = vectorizer.fit_transform(texts)
    
    print(f"✓ Vectorizer fitted successfully!")
    print(f"  - Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  - Features shape: {X_transformed.shape}")
    print(f"    * Samples: {X_transformed.shape[0]}")
    print(f"    * Features: {X_transformed.shape[1]}")
    
    return X_transformed


def transform_data(vectorizer, texts):
    """
    Transform data using FITTED vectorizer.
    
    This is used for TESTING/NEW DATA.
    Uses the vocabulary learned from training data.
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Already fitted vectorizer
    texts : array-like
        Text data to transform
        
    Returns:
    --------
    sparse matrix
        TF-IDF transformed features
        
    Example:
    --------
    >>> X_test = transform_data(vectorizer, test_texts)
    """
    print("\nTransforming test data using fitted vectorizer...")
    
    X_transformed = vectorizer.transform(texts)
    
    print(f"✓ Test data transformed successfully!")
    print(f"  - Samples: {X_transformed.shape[0]}")
    print(f"  - Features: {X_transformed.shape[1]}")
    
    return X_transformed


def get_feature_names(vectorizer, top_n=20):
    """
    Get the names of features (words) from the vectorizer.
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    top_n : int (default=20)
        Number of top features to display
        
    Returns:
    --------
    list
        List of feature names
        
    Example:
    --------
    >>> features = get_feature_names(vectorizer)
    """
    feature_names = vectorizer.get_feature_names_out()
    return feature_names


def convert_sparse_to_dense(X_sparse):
    """
    Convert sparse matrix to dense array.
    
    Warning: This uses more memory!
    Use only for small datasets.
    
    Parameters:
    -----------
    X_sparse : sparse matrix
        Sparse TF-IDF matrix
        
    Returns:
    --------
    np.ndarray
        Dense array
    """
    return X_sparse.toarray()


def save_vectorizer(vectorizer, filepath):
    """
    Save fitted vectorizer to file.
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    filepath : str
        Path to save vectorizer
        
    Returns:
    --------
    None
        
    Example:
    --------
    >>> save_vectorizer(vectorizer, 'models/vectorizer.pkl')
    """
    print(f"\nSaving vectorizer to {filepath}...")
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"✓ Vectorizer saved successfully!")


def load_vectorizer(filepath):
    """
    Load fitted vectorizer from file.
    
    Parameters:
    -----------
    filepath : str
        Path to saved vectorizer
        
    Returns:
    --------
    TfidfVectorizer
        Loaded vectorizer
        
    Example:
    --------
    >>> vectorizer = load_vectorizer('models/vectorizer.pkl')
    """
    print(f"Loading vectorizer from {filepath}...")
    
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f"✓ Vectorizer loaded successfully!")
    
    return vectorizer


def get_vectorizer_info(vectorizer):
    """
    Get information about the vectorizer.
    
    Parameters:
    -----------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
        
    Returns:
    --------
    dict
        Vectorizer information
    """
    info = {
        'vocab_size': len(vectorizer.get_feature_names_out()),
        'max_features': vectorizer.max_features,
        'min_df': vectorizer.min_df,
        'max_df': vectorizer.max_df,
        'ngram_range': vectorizer.ngram_range,
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    from preprocessing import clean_text
    
    # Sample texts
    sample_texts = [
        "This is a real news article about politics",
        "Breaking: Fake story about celebrities",
        "Scientific research shows important findings",
        "Hoax: Unbelievable news that is not true"
    ]
    
    # Create vectorizer
    vectorizer = create_tfidf_vectorizer(max_features=100)
    
    # Transform texts
    X = fit_and_transform(vectorizer, sample_texts)
    
    print(f"\nTF-IDF Matrix shape: {X.shape}")
    print(f"Sample features: {get_feature_names(vectorizer)[:10]}")
