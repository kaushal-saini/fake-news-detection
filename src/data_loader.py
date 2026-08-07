"""
Data Loader Module
==================

This module loads and explores the fake news dataset.

Functions:
- load_data(): Load CSV file
- explore_data(): Display dataset statistics and samples
- check_missing_values(): Check for missing data
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath):
    """
    Load the fake news dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
        
    Example:
    --------
    >>> df = load_data('data/WELFake_Dataset.csv')
    """
    print(f"Loading data from {filepath}...")
    
    # Check if file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    # Load CSV file
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded successfully!")
    
    return df


def explore_data(df):
    """
    Explore and display dataset statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to explore
        
    Returns:
    --------
    None
    
    Displays:
    ---------
    - Dataset shape
    - Column names and types
    - First few rows
    - Basic statistics
    - Class distribution
    """
    print("\n" + "="*70)
    print("DATA EXPLORATION")
    print("="*70)
    
    # Dataset shape
    print(f"\nDataset Shape: {df.shape}")
    print(f"  - Rows: {df.shape[0]}")
    print(f"  - Columns: {df.shape[1]}")
    
    # Column information
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    
    # First few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Dataset statistics
    print(f"\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values!")
    else:
        print(missing[missing > 0])
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    # Class distribution
    if 'label' in df.columns:
        print(f"\nClass Distribution:")
        print(df['label'].value_counts())
        print(f"\nClass Proportions:")
        print(df['label'].value_counts(normalize=True))


def check_missing_values(df):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset to check
        
    Returns:
    --------
    pd.Series
        Missing value counts for each column
    """
    print("\nChecking Missing Values...")
    missing = df.isnull().sum()
    
    if missing.sum() == 0:
        print("✓ No missing values found!")
    else:
        print("Missing values by column:")
        print(missing[missing > 0])
    
    return missing


def get_column_info(df):
    """
    Get detailed information about columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataset
        
    Returns:
    --------
    dict
        Information about each column
    """
    info = {}
    
    for col in df.columns:
        info[col] = {
            'dtype': df[col].dtype,
            'missing': df[col].isnull().sum(),
            'unique': df[col].nunique(),
            'sample': df[col].iloc[0]
        }
    
    return info


if __name__ == "__main__":
    # Example usage
    df = load_data('data/WELFake_Dataset.csv')
    explore_data(df)
    check_missing_values(df)
