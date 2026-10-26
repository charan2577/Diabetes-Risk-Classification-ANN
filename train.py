"""
=============================================================================
DIABETES RISK CLASSIFICATION USING MULTI-LAYER PERCEPTRON (MLP)
Main Training Script: train.py
=============================================================================
Academic Project: CDC BRFSS 2015 Diabetes Health Indicators Classification
Model: Multi-Layer Perceptron (Artificial Neural Network)
Optimization: Full-Batch Gradient Descent (solver='sgd', batch_size=N_train)

This script executes the complete training and evaluation pipeline:
  1. Welcome Banner & System Information
  2. Dataset Ingestion (load_dataset)
  3. Class Distribution Inspection & Exploratory Visualization
  4. Data Splitting & Feature Scaling (preprocess_pipeline)
  5. Multi-Layer Perceptron Architecture Initialization (create_mlp_model)
  6. Model Training via Full-Batch Gradient Descent (train_model)
  7. Performance Evaluation on Test Set (evaluate_model)
  8. Comprehensive Metric Reporting (Accuracy, Precision, Recall, F1)
  9. Diagnostic Plot Generation (Confusion Matrix, Loss Curve, ROC, Distribution)
  10. Model & Scaler Serialization to disk (models/)
  11. Final Summary & Artifact Locations

Usage:
    python train.py
=============================================================================
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration: Ensure project root is present in Python module search path
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Module Imports from src/
# ---------------------------------------------------------------------------
try:
    from src.data_loader import load_dataset
except ImportError as e:
    raise ImportError(f"Failed to import from src.data_loader: {e}")

try:
    from src.preprocessing import preprocess_pipeline
except ImportError as e:
    raise ImportError(f"Failed to import from src.preprocessing: {e}")

try:
    from src.model import create_mlp_model, save_model
except ImportError:
    # Fallback if save_model is defined elsewhere or in src.model
    from src.model import create_mlp_model
    save_model = None

try:
    from src.train_utils import train_model
except ImportError as e:
    raise ImportError(f"Failed to import from src.train_utils: {e}")

try:
    from src.evaluation import evaluate_model, print_metrics
except ImportError as e:
    raise ImportError(f"Failed to import from src.evaluation: {e}")

try:
    from src.visualization import (
        plot_confusion_matrix,
        plot_learning_curve,
        plot_roc_curves,
        plot_class_distribution,
        plot_feature_importance,
    )
except ImportError:
    import src.visualization as viz
    plot_confusion_matrix = getattr(viz, 'plot_confusion_matrix', None)
    plot_learning_curve = getattr(viz, 'plot_learning_curve', getattr(viz, 'plot_loss_curve', None))
    plot_roc_curves = getattr(viz, 'plot_roc_curves', getattr(viz, 'plot_roc_curve', None))
    plot_class_distribution = getattr(viz, 'plot_class_distribution', None)
    plot_feature_importance = getattr(viz, 'plot_feature_importance', None)


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------
def print_banner():
    """Prints a friendly academic welcome banner describing the training process."""
    banner = """
================================================================================
          DIABETES RISK CLASSIFICATION USING NEURAL NETWORKS (MLP)
                  Full-Batch Gradient Descent Training
================================================================================
  Dataset : CDC Diabetes Health Indicators (BRFSS 2015)
  Records : 253,680 survey responses
  Features: 21 clinical, behavioral, and demographic indicators
  Classes : 3 Target Categories:
            - Class 0: Healthy / No Diabetes
            - Class 1: Prediabetes
            - Class 2: Diabetes
  Method  : Artificial Neural Network (MLPClassifier)
  Optimizer: Full-Batch SGD (batch_size = total training samples)
================================================================================
    """
    print(banner)


def check_target_distribution(y):
    """
    Calculates and prints the class distribution of the target variable.
    """
    print("\n[+] Analyzing Target Class Distribution ('Diabetes_012'):")
    class_labels = {
        0: "Class 0: Healthy / No Diabetes",
        1: "Class 1: Prediabetes",
        2: "Class 2: Diabetes"
    }
    
    y_series = pd.Series(y)
    counts = y_series.value_counts().sort_index()
    total = len(y_series)
    
    print("-" * 65)
    print(f"{'Class ID':<10} | {'Description':<32} | {'Count':<10} | {'Percentage'}")
    print("-" * 65)
    for class_id, count in counts.items():
        label = class_labels.get(int(class_id), f"Class {class_id}")
        pct = (count / total) * 100
        print(f"{int(class_id):<10} | {label:<32} | {count:<10,} | {pct:>6.2f}%")
    print("-" * 65)
    print(f"{'Total':<10} | {'All Survey Records':<32} | {total:<10,} | 100.00%")
    print("-" * 65)
    
    minority_ratio = (counts.get(1, 0) / total) * 100
    print(f"[*] Note: Prediabetes (Class 1) accounts for only ~{minority_ratio:.1f}% of data,")
    print("    highlighting significant class imbalance inherent in epidemiological health surveys.\n")


def safe_save_scaler(scaler, filepath):
    """Safely saves the fitted scaler to the models directory."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        from src.preprocessing import save_scaler
        save_scaler(scaler, filepath)
    except Exception:
        joblib.dump(scaler, filepath)
    print(f"[+] Scaler successfully saved to: {filepath}")


def safe_save_model(model, filepath):
    """Safely saves the trained MLP model to the models directory."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if save_model is not None:
        try:
            save_model(model, filepath)
            print(f"[+] Model successfully saved using src.model.save_model to: {filepath}")
            return
        except Exception:
            pass
    joblib.dump(model, filepath)
    print(f"[+] Model successfully serialized and saved to: {filepath}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main(args=None):
    """
    Main execution pipeline for training the MLP model with full-batch gradient descent.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train MLP Classifier with Full-Batch Gradient Descent")
    parser.add_argument(
        '--class-weight',
        type=str,
        default='balanced',
        choices=['balanced', 'none'],
        help="Class weighting strategy for cross-entropy loss: 'balanced' to counter class imbalance or 'none' for unweighted."
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=200,
        help="Maximum number of full-batch training iterations/epochs (default: 200)."
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help="Initial learning rate (alpha) for full-batch gradient descent (default: 0.01)."
    )
    parsed_args = parser.parse_args(args)
    class_weight_setting = 'balanced' if parsed_args.class_weight.lower() == 'balanced' else None
    
    # 1. Print Welcome Banner
    print_banner()
    
    # Ensure destination directories exist
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports", "figures"), exist_ok=True)
    
    # 2. Load Dataset using load_dataset()
    print("--------------------------------------------------------------------------------")
    print("STEP 1: Loading Dataset")
    print("--------------------------------------------------------------------------------")
    X, y = load_dataset()
    print(f"[+] Dataset successfully loaded into memory.")
    print(f"    - Dimensions: {X.shape[0]:,} rows x {X.shape[1]} features")
    
    # 3. Check Class Distribution
    print("\n--------------------------------------------------------------------------------")
    print("STEP 2: Inspecting Target Class Distribution")
    print("--------------------------------------------------------------------------------")
    check_target_distribution(y)
    
    # 4. Split and Scale Data using preprocess_pipeline()
    print("--------------------------------------------------------------------------------")
    print("STEP 3: Splitting and Preprocessing Features")
    print("--------------------------------------------------------------------------------")
    print("[*] Running preprocessing pipeline (Train/Test Split + Feature Scaling)...")
    preproc_result = preprocess_pipeline((X, y))
    
    X_train = preproc_result['X_train']
    X_test = preproc_result['X_test']
    y_train = np.array(preproc_result['y_train']).ravel()
    y_test = np.array(preproc_result['y_test']).ravel()
    scaler = preproc_result['scaler']
    feature_names = preproc_result['feature_names']
    
    num_train_samples = X_train.shape[0]
    num_test_samples = X_test.shape[0]
    num_features = X_train.shape[1]
    
    print(f"[+] Preprocessing complete:")
    print(f"    - Training samples (X_train): {num_train_samples:,} samples, {num_features} features")
    print(f"    - Testing samples  (X_test) : {num_test_samples:,} samples, {num_features} features")
    print(f"    - Target distribution in train: {np.bincount(y_train.astype(int))}")
    print(f"    - Target distribution in test : {np.bincount(y_test.astype(int))}")
    
    # 5. Create MLP Model using create_mlp_model()
    # In Full-Batch Gradient Descent, the batch size equals the entire training set size.
    print("\n--------------------------------------------------------------------------------")
    print("STEP 4: Initializing Multi-Layer Perceptron (Full-Batch Gradient Descent)")
    print("--------------------------------------------------------------------------------")
    print(f"[*] Setting batch_size = {num_train_samples:,} (full-batch gradient descent).")
    print("[*] Setting solver = 'sgd' (Standard Stochastic Gradient Descent configured with full batch).")
    print(f"[*] Setting class_weight = {class_weight_setting} (Cross-Entropy Class Balancing).")
    
    model = create_mlp_model(
        batch_size=num_train_samples,
        class_weight=class_weight_setting,
        max_iter=parsed_args.max_iter,
        learning_rate_init=parsed_args.learning_rate
    )
    
    # Verify model parameters
    effective_solver = getattr(model, 'solver', 'sgd')
    effective_batch_size = getattr(model, 'batch_size', num_train_samples)
    print(f"[+] MLP Model created successfully:")
    print(f"    - Architecture Hidden Layers : {getattr(model, 'hidden_layer_sizes', (128, 64, 32))}")
    print(f"    - Activation Function        : {getattr(model, 'activation', 'relu')}")
    print(f"    - Optimization Solver        : {effective_solver}")
    print(f"    - Batch Size                 : {effective_batch_size:,} (Full Dataset Batch)")
    print(f"    - Learning Rate              : {getattr(model, 'learning_rate_init', 0.01)}")
    print(f"    - Maximum Iterations (Epochs): {getattr(model, 'max_iter', 200)}")
    print(f"    - Class Weighting            : {getattr(model, 'class_weight', class_weight_setting)}")
    
    # 6. Train using train_model() — Full-Batch Gradient Descent
    print("\n--------------------------------------------------------------------------------")
    print("STEP 5: Training MLP Model via Full-Batch Gradient Descent")
    print("--------------------------------------------------------------------------------")
    print("[*] Commencing gradient descent optimization over all training records...")
    print("[*] Note: Full-batch GD computes gradients across all 200k+ records per iteration.")
    print("    Please allow a moment for epochs to converge...")
    
    training_start = time.time()
    train_output = train_model(model, X_train, y_train)
    if train_output is not None:
        model = train_output
    training_duration = time.time() - training_start
    
    print(f"[+] Training completed successfully in {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes).")
    if hasattr(model, 'n_iter_'):
        print(f"    - Total epochs executed : {model.n_iter_}")
    if hasattr(model, 'loss_'):
        print(f"    - Final training loss   : {model.loss_:.6f}")
        
    # 7. Evaluate using evaluate_model()
    print("\n--------------------------------------------------------------------------------")
    print("STEP 6: Evaluating Model Performance on Test Data")
    print("--------------------------------------------------------------------------------")
    print("[*] Computing predictions and evaluation metrics on unseen test partition...")
    metrics = evaluate_model(model, X_test, y_test)
    
    # 8. Print all metrics
    print("\n--------------------------------------------------------------------------------")
    print("STEP 7: Comprehensive Performance Report")
    print("--------------------------------------------------------------------------------")
    if 'print_metrics' in globals() and print_metrics is not None:
        print_metrics(metrics)
    elif isinstance(metrics, dict):
        for k, v in metrics.items():
            if k in ['confusion_matrix', 'classification_report']:
                continue
            if isinstance(v, (int, float, np.floating)):
                print(f"    - {k:<25}: {v:.4f}")
            else:
                print(f"    - {k:<25}: {v}")
                
        if 'classification_report' in metrics:
            print("\nDetailed Classification Report:")
            print(metrics['classification_report'])
        if 'confusion_matrix' in metrics:
            print("\nConfusion Matrix (Raw Counts):")
            print(metrics['confusion_matrix'])
        
    # 9. Generate all plots
    print("\n--------------------------------------------------------------------------------")
    print("STEP 8: Generating Diagnostic Visualizations")
    print("--------------------------------------------------------------------------------")
    
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    lc_path = os.path.join(figures_dir, "learning_curve.png")
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    dist_path = os.path.join(figures_dir, "class_distribution.png")
    fi_path = os.path.join(figures_dir, "feature_importance.png")
    
    # 9a. Class Distribution Plot
    print(f"[*] Generating target class distribution plot...")
    if plot_class_distribution is not None:
        try:
            plot_class_distribution(y, save_path=dist_path)
        except Exception as e:
            print(f"[-] Could not save class distribution plot: {e}")
                
    # 9b. Confusion Matrix Plot
    print(f"[*] Generating confusion matrix visualization...")
    if plot_confusion_matrix is not None:
        try:
            plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
        except Exception as e:
            print(f"[-] Could not save confusion matrix plot: {e}")
                
    # 9c. Learning / Loss Curve Plot
    print(f"[*] Generating optimization learning/loss curve plot...")
    if plot_learning_curve is not None:
        try:
            if hasattr(model, 'loss_curve_'):
                plot_learning_curve(model.loss_curve_, save_path=lc_path)
            else:
                print("[-] Model does not contain loss_curve_ attribute.")
        except Exception as e:
            print(f"[-] Could not save learning curve plot: {e}")
                
    # 9d. ROC Curves Plot
    print(f"[*] Generating multi-class ROC curves plot...")
    if plot_roc_curves is not None:
        try:
            plot_roc_curves(y_test, metrics['y_pred_proba'], save_path=roc_path)
        except Exception as e:
            print(f"[-] Could not save ROC curves plot: {e}")

    # 9e. Feature Importance Plot
    print(f"[*] Generating feature importance plot...")
    if plot_feature_importance is not None:
        try:
            plot_feature_importance(model, feature_names=feature_names, save_path=fi_path)
        except Exception as e:
            print(f"[-] Could not save feature importance plot: {e}")

    # 10. Save Model and Scaler
    print("\n--------------------------------------------------------------------------------")
    print("STEP 9: Persisting Model & Scaler Artifacts")
    print("--------------------------------------------------------------------------------")
    model_save_path = os.path.join(PROJECT_ROOT, "models", "mlp_model.pkl")
    safe_save_model(model, model_save_path)
    
    if scaler is not None:
        scaler_save_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
        safe_save_scaler(scaler, scaler_save_path)

    # 11. Print Summary of Results and Saved Files
    print("\n=================================================================================")
    print("                              TRAINING PIPELINE COMPLETE                          ")
    print("=================================================================================")
    print(f"  Training Time   : {training_duration:.2f} seconds ({training_duration / 60:.2f} minutes)")
    if hasattr(model, 'n_iter_'):
        print(f"  Epochs Trained  : {model.n_iter_}")
    if hasattr(model, 'loss_'):
        print(f"  Final Loss      : {model.loss_:.6f}")
    if isinstance(metrics, dict) and 'accuracy' in metrics:
        print(f"  Test Accuracy   : {metrics['accuracy'] * 100:.2f}%")
        
    print("\n  Generated Artifacts:")
    print(f"    - Model  : {model_save_path}")
    if scaler is not None:
        print(f"    - Scaler : {os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')}")
    print(f"    - Plots  : {figures_dir}/")
    print("               |-- class_distribution.png")
    print("               |-- confusion_matrix.png")
    print("               |-- learning_curve.png")
    print("               |-- roc_curves.png")
    print("               `-- feature_importance.png")
    print("\n  Next Steps:")
    print("    1. Evaluate saved model: python evaluate.py")
    print("    2. Launch Web Interface: streamlit run app.py")
    print("=================================================================================\n")


if __name__ == '__main__':
    main()
