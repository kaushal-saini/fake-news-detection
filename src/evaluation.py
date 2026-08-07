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
    Comprehensive model evaluation using standard classification metrics.
    
    This function calculates all important metrics for binary classification
    to provide a complete picture of model performance. Use these metrics to
    understand trade-offs between catching fake news and minimizing false alarms.
    
    Metrics Explained:
    ==================
    
    Accuracy (Overall Correctness):
    --------------------------------
    Formula: (TP + TN) / Total
    Interpretation: What percentage of predictions are correct?
    When to Use: Good overview, but can be misleading with imbalanced data
    
    Precision (Fake Prediction Reliability):
    ------------------------------------------
    Formula: TP / (TP + FP)
    Interpretation: Of the articles we marked as FAKE, how many are actually fake?
    When to Use: Important when false alarms (marking real as fake) are costly
    Example: 95% precision = 95 out of 100 "fake" predictions are correct
    
    Recall (Fake Detection Rate):
    ------------------------------
    Formula: TP / (TP + FN)
    Interpretation: Of ALL actual fake articles, how many did we find?
    When to Use: Important when missing fakes is costly (detection coverage)
    Example: 96% recall = we catch 96 out of 100 actual fakes
    
    F1-Score (Balanced Metric):
    ----------------------------
    Formula: 2 × (Precision × Recall) / (Precision + Recall)
    Interpretation: Harmonic mean - penalizes extreme values
    When to Use: When precision and recall are equally important
    Range: 0 to 1 (1.0 is perfect, 0 is worst)
    
    Confusion Matrix:
    ------------------
    Provides detailed breakdown:
    - True Positives (TP): Correctly identified real news ✓
    - True Negatives (TN): Correctly identified fake news ✓
    - False Positives (FP): Real marked as fake ✗
    - False Negatives (FN): Fake not detected ✗
    
    Performance Guidelines:
    =======================
    ✓ Excellent: All metrics > 90%
    ✓ Good:      All metrics > 85%
    ⚠ Fair:      Some metrics 70-85%
    ✗ Poor:      Any metric < 70%
    
    Parameters:
    -----------
    y_true : array-like
        True labels (0=Fake, 1=Real)
    y_pred : array-like
        Predicted labels (0=Fake, 1=Real)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - accuracy: Overall correctness (0-1)
        - precision: Fake prediction reliability (0-1)
        - recall: Fake detection rate (0-1)
        - f1_score: Balanced metric (0-1)
        - confusion_matrix: 2D array [TN, FP; FN, TP]
        - classification_report: Detailed per-class metrics
        
    Example:
    --------
    >>> metrics = evaluate_model(y_test, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.2%}")  # 95.34%
    >>> print(f"Precision: {metrics['precision']:.2%}") # 94.76%
    >>> print(f"Recall: {metrics['recall']:.2%}")       # 95.98%
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
