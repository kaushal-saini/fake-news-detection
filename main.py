"""
Fake News Detection - Main Execution Script
============================================

This is the main entry point for the Fake News Detection project.
It orchestrates the complete pipeline:

1. Load Dataset
2. Preprocess Data
3. Feature Extraction (TF-IDF)
4. Train Model (Logistic Regression)
5. Evaluate Model
6. Make Predictions

Usage:
------
python main.py

Requirements:
-----------
- WELFake_Dataset.csv must be in the data/ folder
- See README.md for setup instructions
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, explore_data
from preprocessing import preprocess_dataframe
from feature_extraction import (
    create_tfidf_vectorizer,
    fit_and_transform,
    transform_data,
    save_vectorizer,
    load_vectorizer
)
from model_training import train_model, save_model, load_model, predict_proba
from evaluation import evaluate_model, display_metrics, plot_confusion_matrix
from prediction import predict_news, display_prediction

from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path


def create_directories():
    """Create necessary directories if they don't exist."""
    print("\nSetting up directories...")
    
    dirs = ['data/processed', 'models']
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Directory '{directory}' ready")


def load_and_explore():
    """Step 1: Load and explore the dataset."""
    print("\n" + "="*70)
    print("STEP 1: LOAD AND EXPLORE DATA")
    print("="*70)
    
    try:
        df = load_data('data/WELFake_Dataset.csv')
        explore_data(df)
        return df
    except FileNotFoundError:
        print("\n❌ ERROR: Dataset not found!")
        print("Please download WELFake_Dataset.csv from:")
        print("https://www.kaggle.com/datasets/jainpooja/welfake-dataset")
        print("\nPlace it in the 'data/' folder and try again.")
        sys.exit(1)


def preprocess():
    """Step 2: Preprocess the data."""
    print("\n" + "="*70)
    print("STEP 2: PREPROCESS DATA")
    print("="*70)
    
    df = load_data('data/WELFake_Dataset.csv')
    df_processed = preprocess_dataframe(df, text_column='text', label_column='label')
    
    return df_processed


def split_data(df):
    """Step 3: Split data into training and testing sets."""
    print("\n" + "="*70)
    print("STEP 3: SPLIT DATA")
    print("="*70)
    
    print("\nSplitting data into training (80%) and testing (20%)...")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution
    )
    
    print(f"✓ Data split successfully!")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Testing samples: {len(X_test)}")
    print(f"  - Training set class distribution: {y_train.value_counts().to_dict()}")
    print(f"  - Testing set class distribution: {y_test.value_counts().to_dict()}")
    
    # Save split data
    print("\nSaving split data...")
    with open('data/processed/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('data/processed/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('data/processed/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('data/processed/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    print("✓ Split data saved to data/processed/")
    
    return X_train, X_test, y_train, y_test


def extract_features(X_train, X_test):
    """Step 4: Extract features using TF-IDF."""
    print("\n" + "="*70)
    print("STEP 4: FEATURE EXTRACTION (TF-IDF)")
    print("="*70)
    
    # Create vectorizer
    vectorizer = create_tfidf_vectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2)
    )
    
    # Fit on training data and transform
    X_train_tfidf = fit_and_transform(vectorizer, X_train)
    
    # Transform test data
    X_test_tfidf = transform_data(vectorizer, X_test)
    
    # Save vectorizer
    save_vectorizer(vectorizer, 'models/vectorizer.pkl')
    
    return X_train_tfidf, X_test_tfidf, vectorizer


def train():
    """Step 5: Train the Logistic Regression model."""
    print("\n" + "="*70)
    print("STEP 5: MODEL TRAINING")
    print("="*70)
    
    # Load data
    with open('data/processed/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('data/processed/y_train.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    # Load vectorizer
    vectorizer = load_vectorizer('models/vectorizer.pkl')
    
    # Extract features
    X_train_tfidf = vectorizer.transform(X_train)
    
    # Train model
    model = train_model(X_train_tfidf, y_train)
    
    # Save model
    save_model(model, 'models/fake_news_model.pkl')
    
    return model, vectorizer


def evaluate():
    """Step 6: Evaluate the model."""
    print("\n" + "="*70)
    print("STEP 6: MODEL EVALUATION")
    print("="*70)
    
    # Load model and data
    model = load_model('models/fake_news_model.pkl')
    vectorizer = load_vectorizer('models/vectorizer.pkl')
    
    with open('data/processed/X_test.pkl', 'rb') as f:
        X_test = pickle.load(f)
    with open('data/processed/y_test.pkl', 'rb') as f:
        y_test = pickle.load(f)
    
    # Extract features
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = predict_proba(model, X_test_tfidf)
    
    # Evaluate
    metrics = evaluate_model(y_test, y_pred)
    display_metrics(metrics)
    
    # Plot confusion matrix (optional - comment out if matplotlib issues)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        plot_confusion_matrix(metrics, save_path='results/confusion_matrix.png')
    except Exception as e:
        print(f"Note: Could not create plots: {e}")
    
    return metrics


def make_predictions():
    """Step 7: Make predictions on sample data."""
    print("\n" + "="*70)
    print("STEP 7: MAKE PREDICTIONS ON NEW DATA")
    print("="*70)
    
    model = load_model('models/fake_news_model.pkl')
    vectorizer = load_vectorizer('models/vectorizer.pkl')
    
    # Sample news articles
    sample_articles = [
        "Scientists discover breakthrough in cancer treatment using stem cells",
        "FAKE: Aliens found in Arctic cave claims mysterious source",
        "Government announces new climate action plan",
        "HOAX: Celebrity death rumor spreads on social media",
        "Breaking: New study shows benefits of daily exercise for heart health"
    ]
    
    print("\nMaking predictions on sample articles...\n")
    
    for i, article in enumerate(sample_articles, 1):
        result = predict_news(article, model, vectorizer)
        display_prediction(article, result, max_length=80)
        print()


def run_complete_pipeline():
    """
    Run the complete pipeline.
    """
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION - COMPLETE PIPELINE")
    print("="*70)
    
    # Create directories
    create_directories()
    
    # Check if models already exist
    if Path('models/fake_news_model.pkl').exists():
        print("\n✓ Found pre-trained model. Skipping training...")
        print("To retrain, delete 'models/fake_news_model.pkl' and run again.")
        
        # Evaluate existing model
        evaluate()
        
        # Make predictions
        make_predictions()
    else:
        # Full pipeline
        df = load_and_explore()
        df_processed = preprocess()
        X_train, X_test, y_train, y_test = split_data(df_processed)
        X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
        model = train()
        metrics = evaluate()
        make_predictions()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the model performance metrics above")
    print("2. Try the prediction examples")
    print("3. Modify and experiment with different parameters")
    print("4. Explore the code in the 'src/' folder")
    print("\nTo make predictions on your own data:")
    print("  - Edit make_predictions() function or")
    print("  - Use src/prediction.py predict_news() function")


if __name__ == "__main__":
    try:
        run_complete_pipeline()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
