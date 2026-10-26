"""
evaluation.py
=============
Model Evaluation module for Diabetes Risk Classification Using MLP.

This module provides academic evaluation metrics for multi-class classification:
1. Overall Accuracy.
2. Comprehensive Classification Report (Precision, Recall, F1-Score per class,
   plus Macro and Weighted Averages).
3. Confusion Matrix computation.
4. One-vs-Rest (OvR) ROC-AUC score calculation across all three classes:
   Class 0: Healthy, Class 1: Prediabetes, Class 2: Diabetes.
5. Formatted printing utilities to inspect model performance clearly.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from src.data_loader import get_class_names


def compute_roc_auc(y_test, y_pred_proba, n_classes=3):
    """
    Computes per-class and overall ROC-AUC scores using One-vs-Rest (OvR).
    
    In a 3-class classification problem (Healthy, Prediabetes, Diabetes):
    - Class i is treated as positive, and all other classes are treated as negative.
    - An individual ROC curve and AUC score is calculated for each class.
    - Macro-average and weighted-average AUCs are also determined.
    
    Args:
        y_test (array-like): True integer labels [0, 1, 2].
        y_pred_proba (array-like): Predicted probability matrix of shape (n_samples, n_classes).
        n_classes (int): Number of distinct target classes (default is 3).
        
    Returns:
        dict: Mapping of class index to individual AUC score, plus 'macro' and 'weighted' AUCs.
    """
    y_test_arr = np.array(y_test)
    y_proba_arr = np.array(y_pred_proba)
    
    # Binarize the labels for multi-class ROC computation (One-vs-Rest)
    y_test_bin = label_binarize(y_test_arr, classes=list(range(n_classes)))
    
    auc_scores = {}
    
    # Per-class ROC-AUC
    for i in range(n_classes):
        try:
            auc_scores[i] = roc_auc_score(y_test_bin[:, i], y_proba_arr[:, i])
        except ValueError:
            auc_scores[i] = float('nan')
            
    # Macro and Weighted Average ROC-AUC across all classes
    try:
        auc_scores['macro'] = roc_auc_score(
            y_test_arr, y_proba_arr, multi_class='ovr', average='macro'
        )
    except ValueError:
        auc_scores['macro'] = float('nan')
        
    try:
        auc_scores['weighted'] = roc_auc_score(
            y_test_arr, y_proba_arr, multi_class='ovr', average='weighted'
        )
    except ValueError:
        auc_scores['weighted'] = float('nan')
        
    return auc_scores


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluates the trained MLPClassifier on the unseen test dataset.
    
    Args:
        model: Fitted MLPClassifier instance.
        X_test (np.ndarray or pd.DataFrame): Scaled test features.
        y_test (np.ndarray or pd.Series): Ground-truth test target labels.
        class_names (dict or list, optional): Mapping of class indices to names.
        
    Returns:
        dict: Evaluation results containing:
            - 'accuracy': float, overall classification accuracy
            - 'classification_report': str, formatted scikit-learn report
            - 'confusion_matrix': np.ndarray, (n_classes, n_classes) confusion matrix
            - 'y_pred': np.ndarray, discrete class predictions
            - 'y_pred_proba': np.ndarray, continuous prediction probabilities
            - 'roc_auc': dict, per-class and aggregated ROC-AUC scores
            - 'class_names': dict, mapping used for reporting
    """
    if class_names is None:
        class_names = get_class_names()
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    print("\nRunning model inference on test dataset...")
    # Predict discrete class labels
    y_pred = model.predict(X_test)
    
    # Predict class posterior probabilities (needed for ROC-AUC)
    y_pred_proba = model.predict_proba(X_test)
    
    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Generate human-readable classification report
    sorted_labels = sorted(class_names.keys())
    target_names = [class_names[lbl] for lbl in sorted_labels]
    
    report_str = classification_report(
        y_test,
        y_pred,
        labels=sorted_labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted_labels)
    
    # Compute ROC-AUC scores
    n_classes = len(sorted_labels)
    roc_auc_dict = compute_roc_auc(y_test, y_pred_proba, n_classes=n_classes)
    
    results = {
        'accuracy': acc,
        'classification_report': report_str,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc_dict,
        'class_names': class_names
    }
    
    return results


def print_metrics(results):
    """
    Nicely formats and prints evaluation results to the console.
    
    Args:
        results (dict): Dictionary produced by evaluate_model().
    """
    class_names = results.get('class_names', get_class_names())
    acc = results['accuracy']
    cm = results['confusion_matrix']
    report = results['classification_report']
    roc_dict = results.get('roc_auc', {})
    
    print("\n" + "=" * 65)
    print("           MODEL PERFORMANCE EVALUATION METRICS           ")
    print("=" * 65)
    print(f"\n>> OVERALL ACCURACY: {acc * 100:.2f}% ({acc:.4f})\n")
    
    print(">> DETAILED CLASSIFICATION REPORT:")
    print("-" * 65)
    print(report)
    print("-" * 65)
    
    print("\n>> CONFUSION MATRIX (Rows: True Class | Columns: Predicted Class):")
    print("-" * 65)
    header = f"{'True \\ Pred':<15}" + "".join([f"{class_names.get(i, f'C{i}'):>15}" for i in range(len(cm))])
    print(header)
    print("-" * 65)
    for i, row in enumerate(cm):
        row_str = f"{class_names.get(i, f'C{i}'):<15}" + "".join([f"{val:>15,}" for val in row])
        print(row_str)
    print("-" * 65)
    
    if roc_dict:
        print("\n>> AREA UNDER THE ROC CURVE (One-vs-Rest ROC-AUC):")
        print("-" * 65)
        for i, class_name in class_names.items():
            auc_val = roc_dict.get(i, float('nan'))
            print(f"  Class {i} ({class_name:<12}) ROC-AUC: {auc_val:.4f}")
        print("-" * 65)
        if 'macro' in roc_dict:
            print(f"  Macro-Average ROC-AUC        : {roc_dict['macro']:.4f}")
        if 'weighted' in roc_dict:
            print(f"  Weighted-Average ROC-AUC     : {roc_dict['weighted']:.4f}")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    print("evaluation.py loaded successfully.")
