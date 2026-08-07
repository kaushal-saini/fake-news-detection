"""
Model Evaluation Module
=======================

This module evaluates model performance using various metrics.

Metrics:
- Accuracy: Overall correctness
- Precision: How many predicted fakes are actually fake
- Recall: What portion of actual fakes we identified
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: True/False Positives/Negatives

Functions:
- evaluate_model(): Calculate all metrics
- display_metrics(): Display results in a readable format
- plot_confusion_matrix(): Visualize confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)


def evaluate_model(y_true, y_pred):
    """
    Evaluate model using all metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    dict
        Dictionary containing all metrics
        
    Metrics Explained:
    ------------------
    Accuracy: (TP + TN) / Total
      - Overall correctness of predictions
      - Simple but can be misleading with imbalanced data
      
    Precision: TP / (TP + FP)
      - Of the news we predicted as fake, how many are actually fake?
      - Important when false positives are costly
      - "How trustworthy are our fake predictions?"
      
    Recall: TP / (TP + FN)
      - Of all actual fake news, how many did we catch?
      - Important when false negatives are costly
      - "How many fakes do we find?"
      
    F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
      - Balanced combination of precision and recall
      - Good when you care about both false positives and false negatives
      
    Example:
    --------
    >>> metrics = evaluate_model(y_test, y_pred)
    >>> print(metrics['accuracy'])
    """
    print("\nEvaluating Model Performance...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Store in dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_report(y_true, y_pred, 
                                                       target_names=['Fake', 'Real'])
    }
    
    return metrics


def display_metrics(metrics):
    """
    Display evaluation metrics in a readable format.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary from evaluate_model()
        
    Returns:
    --------
    None
        Prints formatted results
        
    Example:
    --------
    >>> display_metrics(metrics)
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Display main metrics
    print("\nKey Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Fake  Real")
    print(f"Actual Fake  [ {cm[0,0]:5d} {cm[0,1]:5d} ]")
    print(f"       Real  [ {cm[1,0]:5d} {cm[1,1]:5d} ]")
    
    # Calculate derived metrics from confusion matrix
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nDetailed Analysis:")
    print(f"  True Negatives (TN):  {tn} - Correctly identified fake news")
    print(f"  False Positives (FP): {fp} - Real news incorrectly marked as fake")
    print(f"  False Negatives (FN): {fn} - Fake news missed (not detected)")
    print(f"  True Positives (TP):  {tp} - Correctly identified real news")
    
    print(f"\nSpecificity: {specificity:.4f} - How good at identifying real news")
    print(f"Sensitivity: {sensitivity:.4f} - How good at identifying fake news")
    
    # Classification Report
    print(f"\nClassification Report:")
    print(metrics['classification_report'])
    
    print("="*70)


def plot_confusion_matrix(metrics, save_path=None):
    """
    Plot confusion matrix as heatmap.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    save_path : str (optional)
        Path to save figure
        
    Returns:
    --------
    None
        Displays plot
    """
    cm = metrics['confusion_matrix']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'],
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Fake News Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_metrics_comparison(metrics, save_path=None):
    """
    Plot metrics comparison bar chart.
    
    Parameters:
    -----------
    metrics : dict
        Metrics dictionary
    save_path : str (optional)
        Path to save figure
        
    Returns:
    --------
    None
        Displays plot
    """
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score']
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylim([0, 1.1])
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics chart saved to {save_path}")
    
    plt.show()


def get_prediction_confidence(y_pred_proba):
    """
    Get prediction confidence scores.
    
    Parameters:
    -----------
    y_pred_proba : array-like
        Probability predictions from model.predict_proba()
        Shape: (n_samples, 2)
        - Column 0: Probability of class 0 (fake)
        - Column 1: Probability of class 1 (real)
        
    Returns:
    --------
    np.ndarray
        Maximum probability for each prediction
    """
    confidence = np.max(y_pred_proba, axis=1)
    return confidence


def print_sample_predictions(y_true, y_pred, y_pred_proba, n_samples=5):
    """
    Print sample predictions with confidence scores.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Probability predictions
    n_samples : int
        Number of samples to display
        
    Returns:
    --------
    None
        Prints results
    """
    print("\nSample Predictions (first 5):")
    print("-" * 80)
    
    for i in range(min(n_samples, len(y_true))):
        true_label = "Real" if y_true[i] == 1 else "Fake"
        pred_label = "Real" if y_pred[i] == 1 else "Fake"
        confidence = get_prediction_confidence(y_pred_proba[i:i+1])[0]
        correct = "✓" if y_true[i] == y_pred[i] else "✗"
        
        print(f"Sample {i+1}: {correct}")
        print(f"  True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.2%}")


if __name__ == "__main__":
    # Example usage
    from sklearn.metrics import confusion_matrix
    
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    y_pred = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    
    metrics = evaluate_model(y_true, y_pred)
    display_metrics(metrics)
