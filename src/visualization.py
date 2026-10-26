"""
visualization.py
================
Visualization module for Diabetes Risk Classification Using MLP.

This module provides academic-grade plotting functions:
1. Confusion Matrix Heatmap (counts and relative proportions).
2. Training Loss vs Epochs (Loss curve demonstrating batch gradient descent).
3. One-vs-Rest (OvR) ROC Curves with per-class Area Under the Curve (AUC).
4. Target Class Distribution Bar Chart (illustrating class imbalance).
5. Input Feature Importance Estimation (derived from MLP first-layer weight magnitudes).

All plotting functions automatically create target directories, save high-resolution
PNG figures (150 DPI), and display plots interactively.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless backend to prevent blocking GUI popups
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.data_loader import get_class_names, get_feature_names


def _ensure_dir(filepath):
    """Internal helper to ensure the target directory for a figure exists."""
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_confusion_matrix(cm, class_names=None, save_path='reports/figures/confusion_matrix.png'):
    """
    Plots and saves a styled Confusion Matrix heatmap using Seaborn.
    
    Args:
        cm (np.ndarray): 2D square confusion matrix array.
        class_names (dict or list, optional): Mapping or list of class names.
        save_path (str): Destination file path for saving the figure.
    """
    _ensure_dir(save_path)
    
    if class_names is None:
        class_names = get_class_names()
        
    if isinstance(class_names, dict):
        labels = [class_names[i] for i in sorted(class_names.keys())]
    else:
        labels = list(class_names)
        
    plt.figure(figsize=(8, 6))
    
    # Create annotations with count and row percentage
    cm_sum_per_row = cm.sum(axis=1, keepdims=True)
    # Avoid zero-division if a row is completely zero
    cm_perc = np.divide(cm.astype('float'), cm_sum_per_row, where=cm_sum_per_row != 0) * 100
    
    annot_labels = np.empty_like(cm, dtype=object)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            annot_labels[r, c] = f"{cm[r, c]:,}\n({cm_perc[r, c]:.1f}%)"
            
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt='',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    plt.title('Confusion Matrix - Diabetes Risk Classification (MLP)', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix plot saved to: {save_path}")
    plt.close()


def plot_learning_curve(loss_curve, save_path='reports/figures/learning_curve.png'):
    """
    Plots and saves the training loss trajectory over epochs for Full-Batch Gradient Descent.
    
    In Full-Batch GD, each epoch represents one full iteration over all training samples,
    resulting in a characteristic smooth, monotonically decreasing loss trajectory.
    
    Args:
        loss_curve (list or np.ndarray): Loss recorded at each epoch.
        save_path (str): Destination file path for saving the figure.
    """
    _ensure_dir(save_path)
    
    epochs = list(range(1, len(loss_curve) + 1))
    
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, loss_curve, color='#1f77b4', linewidth=2.5, label='Full-Batch GD Loss')
    
    # Highlight start and end points
    plt.scatter([epochs[0], epochs[-1]], [loss_curve[0], loss_curve[-1]],
                color='#d62728', zorder=5, s=60)
    plt.annotate(f"Start: {loss_curve[0]:.4f}", (epochs[0], loss_curve[0]),
                 textcoords="offset points", xytext=(10, 5), fontsize=10)
    plt.annotate(f"Final: {loss_curve[-1]:.4f}", (epochs[-1], loss_curve[-1]),
                 textcoords="offset points", xytext=(-70, 10), fontsize=10)
    
    plt.title('Training Loss Curve (Full-Batch Gradient Descent)', fontsize=14, pad=15)
    plt.xlabel('Epoch (Full-Batch Update Step)', fontsize=12)
    plt.ylabel('Cross-Entropy Loss (Log-Loss)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Learning curve plot saved to: {save_path}")
    plt.close()


def plot_roc_curves(y_test, y_pred_proba, class_names=None, save_path='reports/figures/roc_curves.png'):
    """
    Plots and saves multi-class One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) curves.
    
    Args:
        y_test (array-like): True class labels [0, 1, 2].
        y_pred_proba (array-like): Predicted probabilities of shape (n_samples, n_classes).
        class_names (dict or list, optional): Mapping or list of class names.
        save_path (str): Destination file path for saving the figure.
    """
    _ensure_dir(save_path)
    
    if class_names is None:
        class_names = get_class_names()
        
    y_test_arr = np.array(y_test)
    y_proba_arr = np.array(y_pred_proba)
    
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test_arr, classes=list(range(n_classes)))
    
    palette = ['#2ca02c', '#ff7f0e', '#d62728']  # Green (Healthy), Orange (Prediabetes), Red (Diabetes)
    
    plt.figure(figsize=(9, 7))
    
    for i in range(n_classes):
        name = class_names[i] if isinstance(class_names, dict) else class_names[i]
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_arr[:, i])
        roc_auc_val = auc(fpr, tpr)
        
        color = palette[i % len(palette)]
        plt.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.2,
            label=f"Class {i} ({name}) (AUC = {roc_auc_val:.3f})"
        )
        
    # Plot diagonal 50% random guessing baseline
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Baseline (AUC = 0.500)')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.title('Multi-Class One-vs-Rest ROC Curves (MLP Classifier)', fontsize=14, pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.legend(loc='lower right', fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"ROC curves plot saved to: {save_path}")
    plt.close()


def plot_class_distribution(y, class_names=None, save_path='reports/figures/class_distribution.png'):
    """
    Plots and saves a bar chart showing class frequency and percentage distribution.
    
    Args:
        y (pd.Series or np.ndarray): Target class labels.
        class_names (dict or list, optional): Mapping or list of class names.
        save_path (str): Destination file path for saving the figure.
    """
    _ensure_dir(save_path)
    
    if class_names is None:
        class_names = get_class_names()
        
    y_series = pd.Series(y)
    counts = y_series.value_counts().sort_index()
    total = len(y_series)
    
    labels = [class_names.get(i, f"Class {i}") for i in counts.index]
    colors = ['#4c72b0', '#dd8452', '#c44e52']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts.values, color=colors[:len(labels)], edgecolor='black', width=0.55)
    
    # Add count and percentage labels above each bar
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (total * 0.015),
            f"{int(height):,}\n({pct:.2f}%)",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
        
    plt.title('Target Class Distribution (Diabetes_012)', fontsize=14, pad=15)
    plt.xlabel('Health Status Category', fontsize=12)
    plt.ylabel('Number of Individuals', fontsize=12)
    plt.ylim(0, max(counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Class distribution plot saved to: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_names=None, save_path='reports/figures/feature_importance.png'):
    """
    Computes, plots, and saves feature importance derived from the first-layer
    weights of the Multilayer Perceptron.
    
    Methodology:
    Each input feature j connects to all neurons in hidden layer 1 via weights W_j.
    The importance metric is calculated as the mean absolute connection weight:
        Importance(j) = (1 / M_1) * sum_{k=1}^{M_1} |W_{j, k}^{(1)}|
        
    Args:
        model: Fitted MLPClassifier containing `coefs_`.
        feature_names (list of str, optional): Ordered names of the 21 features.
        save_path (str): Destination file path for saving the figure.
    """
    _ensure_dir(save_path)
    
    if not hasattr(model, 'coefs_') or len(model.coefs_) == 0:
        raise ValueError(
            "Model has no learned weights ('coefs_'). Train the model before "
            "plotting feature importance."
        )
        
    if feature_names is None:
        feature_names = get_feature_names()
        
    # model.coefs_[0] has shape (n_input_features, n_first_hidden_neurons)
    w_input = model.coefs_[0]
    importances = np.mean(np.abs(w_input), axis=1)
    
    # Create sorted DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(
        importance_df['Feature'],
        importance_df['Importance'],
        color='#3470a3',
        edgecolor='black',
        height=0.65
    )
    
    plt.title('Feature Importance via First-Layer Weight Magnitudes (MLP)', fontsize=14, pad=15)
    plt.xlabel('Mean Absolute Weight (|W|) to First Hidden Layer', fontsize=12)
    plt.ylabel('Clinical & Demographic Indicators', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


if __name__ == '__main__':
    print("visualization.py loaded successfully.")
