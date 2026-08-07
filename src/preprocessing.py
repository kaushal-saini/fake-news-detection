"""
Data Preprocessing Module
==========================

This module handles all text preprocessing tasks including:
- Lowercase conversion
- Removing special characters and URLs
- Removing stopwords
- Lemmatization
- Removing extra whitespace

Functions:
- preprocess_text(): Clean a single text string
- preprocess_dataframe(): Clean entire dataframe
- clean_text(): Core text cleaning function
"""

import re
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


def remove_urls(text):
    """
    Remove URLs from text.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text without URLs
    """
    url_pattern = r'http\S+|www\S+'
    return re.sub(url_pattern, '', text)


def remove_special_characters(text):
    """
    Remove special characters and numbers, keep only letters and spaces.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with only letters and spaces
    """
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def remove_extra_whitespace(text):
    """
    Remove extra whitespace from text.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text with cleaned whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_stopwords_text(text):
    """
    Remove stopwords from text.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Text without stopwords
    """
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Join back
    return ' '.join(filtered_tokens)


def lemmatize_text(text):
    """
    Lemmatize text (convert words to base form).
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    str
        Lemmatized text
    """
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Lemmatize each token
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back
    return ' '.join(lemmatized_tokens)


def clean_text(text):
    """
    Complete text cleaning pipeline.
    
    Implements a 6-step text normalization process optimized for news classification.
    Each step is designed to reduce noise while preserving semantic meaning.
    
    Steps:
    1. Convert to lowercase - normalize case variations
    2. Remove URLs - eliminate web links that don't contribute to classification
    3. Remove special characters - keep only alphabetic content
    4. Remove extra whitespace - normalize spacing
    5. Remove stopwords - eliminate common, non-discriminative words
    6. Lemmatization - convert words to base form for better feature matching
    
    Parameters:
    -----------
    text : str
        Raw text to clean
        
    Returns:
    --------
    str
        Cleaned text ready for TF-IDF vectorization
        
    Example:
    --------
    >>> raw = "Check this URL: http://example.com!!! AMAZING news!!!"
    >>> cleaned = clean_text(raw)
    >>> print(cleaned)  # "check amazing news"
    """
    if not isinstance(text, str):
        return ""
    
    # Step 1: Lowercase
    text = text.lower()
    
    # Step 2: Remove URLs
    text = remove_urls(text)
    
    # Step 3: Remove special characters
    text = remove_special_characters(text)
    
    # Step 4: Remove extra whitespace
    text = remove_extra_whitespace(text)
    
    # Step 5: Remove stopwords
    text = remove_stopwords_text(text)
    
    # Step 6: Lemmatization
    text = lemmatize_text(text)
    
    return text


def preprocess_text(text):
    """
    Preprocess a single text string.
    
    Parameters:
    -----------
    text : str
        Raw text to preprocess
        
    Returns:
    --------
    str
        Preprocessed text
        
    Example:
    --------
    >>> cleaned = preprocess_text("Check this URL: http://example.com")
    """
    return clean_text(text)


def preprocess_dataframe(df, text_column='text', label_column='label'):
    """
    Preprocess entire dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    text_column : str
        Name of column containing text
    label_column : str
        Name of column containing labels
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
        
    Example:
    --------
    >>> df_cleaned = preprocess_dataframe(df, text_column='text', label_column='label')
    """
    print("Starting data preprocessing...")
    print(f"Total records: {len(df)}")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Remove duplicates
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates(subset=[text_column])
    removed_duplicates = initial_rows - len(df_processed)
    print(f"✓ Removed {removed_duplicates} duplicate rows")
    
    # Handle missing values
    missing_before = df_processed[text_column].isnull().sum()
    df_processed = df_processed.dropna(subset=[text_column])
    missing_removed = missing_before - df_processed[text_column].isnull().sum()
    print(f"✓ Removed {missing_removed} rows with missing text")
    
    # Clean text
    print("Cleaning text (this may take a while for large datasets)...")
    df_processed[text_column] = df_processed[text_column].apply(clean_text)
    print("✓ Text cleaning completed")
    
    # Remove rows with empty text after cleaning
    empty_before = len(df_processed)
    df_processed = df_processed[df_processed[text_column].str.len() > 0]
    empty_removed = empty_before - len(df_processed)
    print(f"✓ Removed {empty_removed} rows with empty text after cleaning")
    
    print(f"\nPreprocessing completed!")
    print(f"Original records: {initial_rows}")
    print(f"Final records: {len(df_processed)}")
    
    return df_processed


def get_text_statistics(df, text_column='text'):
    """
    Get statistics about the text data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe
    text_column : str
        Name of text column
        
    Returns:
    --------
    dict
        Text statistics
    """
    # Calculate word counts
    word_counts = df[text_column].str.split().str.len()
    
    stats = {
        'min_words': word_counts.min(),
        'max_words': word_counts.max(),
        'avg_words': word_counts.mean(),
        'median_words': word_counts.median(),
        'total_unique_words': len(set(' '.join(df[text_column]).split()))
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    sample_text = "Check this URL: http://example.com!!! This is SAMPLE text with #special characters."
    print("Original text:", sample_text)
    print("Cleaned text:", clean_text(sample_text))
