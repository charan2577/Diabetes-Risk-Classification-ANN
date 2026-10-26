"""
=============================================================================
DIABETES RISK CLASSIFICATION USING MULTI-LAYER PERCEPTRON (MLP)
Model Evaluation Script: evaluate.py
=============================================================================
Academic Project: CDC BRFSS 2015 Diabetes Health Indicators Classification
Model: Multi-Layer Perceptron (Artificial Neural Network)

This script loads an existing trained model and evaluates it against test data:
  1. Loads the CDC Diabetes Health Indicators dataset (load_dataset)
  2. Preprocesses and scales features using the established pipeline (preprocess_pipeline)
  3. Loads the serialized MLP model artifact from models/mlp_model.pkl
  4. Evaluates predictions and calculates performance metrics (evaluate_model)
  5. Displays detailed metric tables (Accuracy, Precision, Recall, F1, Matrix)
  6. Generates and updates diagnostic visualization figures in plots/

Usage:
    python evaluate.py
=============================================================================
"""

import os
import sys
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
    raise ImportError(f"Failed to import load_dataset from src.data_loader: {e}")

try:
    from src.preprocessing import preprocess_pipeline
except ImportError as e:
    raise ImportError(f"Failed to import preprocess_pipeline from src.preprocessing: {e}")

try:
    from src.model import load_model
except ImportError:
    # Fallback to joblib if load_model is not explicitly defined in src.model
    load_model = None

try:
    from src.evaluation import evaluate_model, print_metrics
except ImportError as e:
    raise ImportError(f"Failed to import evaluate_model from src.evaluation: {e}")

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
    """Prints a banner for the standalone evaluation procedure."""
    banner = """
================================================================================
          DIABETES RISK CLASSIFICATION - MODEL EVALUATION SUITE
================================================================================
  Loading serialized model artifact and testing generalization on unseen data.
  Target Classes:
    - 0: Healthy / No Diabetes
    - 1: Prediabetes
    - 2: Diabetes
================================================================================
    """
    print(banner)


def safe_load_model(filepath):
    """
    Safely loads the serialized MLP model artifact from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the serialized model file (.pkl).
        
    Returns:
    --------
    model : MLPClassifier
        The loaded scikit-learn MLPClassifier instance.
    """
    if not os.path.exists(filepath):
        print(f"\n[!] ERROR: Model file not found at '{filepath}'.")
        print("    Please train and save the model first by executing:")
        print("        python train.py\n")
        sys.exit(1)
        
    if load_model is not None:
        try:
            model = load_model(filepath)
            print(f"[+] Model successfully loaded using src.model.load_model from: {filepath}")
            return model
        except Exception:
            pass
            
    model = joblib.load(filepath)
    print(f"[+] Model successfully loaded using joblib from: {filepath}")
    return model


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------
def main():
    """
    Main execution workflow for evaluating a saved MLP model.
    """
    print_banner()
    
    # 1. Load Dataset and Preprocess
    print("--------------------------------------------------------------------------------")
    print("STEP 1: Ingesting Data & Preparing Test Split")
    print("--------------------------------------------------------------------------------")
    print("[*] Loading CDC BRFSS 2015 dataset...")
    X, y = load_dataset()
    print(f"[+] Dataset loaded: {X.shape[0]:,} rows x {X.shape[1]} features")
    
    print("[*] Applying preprocessing pipeline...")
    preproc_result = preprocess_pipeline((X, y))
    
    X_test = preproc_result['X_test']
    y_test = np.array(preproc_result['y_test']).ravel()
    feature_names = preproc_result['feature_names']
    print(f"[+] Test set prepared with {X_test.shape[0]:,} samples and {X_test.shape[1]} features.")
    
    # 2. Load Saved Model from models/mlp_model.pkl
    print("\n--------------------------------------------------------------------------------")
    print("STEP 2: Loading Trained MLP Model Artifact")
    print("--------------------------------------------------------------------------------")
    model_path = os.path.join(PROJECT_ROOT, "models", "mlp_model.pkl")
    model = safe_load_model(model_path)
    
    # Display architecture information
    if hasattr(model, 'hidden_layer_sizes'):
        print(f"    - Network Architecture       : {model.hidden_layer_sizes}")
        print(f"    - Activation Function        : {getattr(model, 'activation', 'relu')}")
        print(f"    - Optimization Algorithm     : {getattr(model, 'solver', 'sgd')}")
        print(f"    - Batch Size                 : {getattr(model, 'batch_size', 'full-batch')}")
        if hasattr(model, 'n_iter_'):
            print(f"    - Completed Training Epochs  : {model.n_iter_}")
        if hasattr(model, 'loss_'):
            print(f"    - Final Training Loss        : {model.loss_:.6f}")

    # 3. Evaluate and Print All Metrics
    print("\n--------------------------------------------------------------------------------")
    print("STEP 3: Running Test Set Evaluation")
    print("--------------------------------------------------------------------------------")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n--------------------------------------------------------------------------------")
    print("STEP 4: Test Performance Results")
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
            print("\nDetailed Per-Class Classification Report:")
            print(metrics['classification_report'])
        if 'confusion_matrix' in metrics:
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
        
    # 4. Generate All Diagnostic Plots
    print("\n--------------------------------------------------------------------------------")
    print("STEP 5: Generating Diagnostic Evaluation Visualizations")
    print("--------------------------------------------------------------------------------")
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    lc_path = os.path.join(figures_dir, "learning_curve.png")
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    dist_path = os.path.join(figures_dir, "class_distribution.png")
    fi_path = os.path.join(figures_dir, "feature_importance.png")
    
    # 4a. Class Distribution Plot
    print(f"[*] Checking class distribution visualization...")
    if plot_class_distribution is not None:
        try:
            plot_class_distribution(y, save_path=dist_path)
        except Exception as e:
            print(f"[-] Could not save class distribution plot: {e}")
                
    # 4b. Confusion Matrix Plot
    print(f"[*] Generating confusion matrix heatmap...")
    if plot_confusion_matrix is not None:
        try:
            plot_confusion_matrix(metrics['confusion_matrix'], save_path=cm_path)
        except Exception as e:
            print(f"[-] Could not save confusion matrix plot: {e}")

    # 4c. Learning / Loss Curve Plot
    print(f"[*] Generating learning / loss curve visualization...")
    if plot_learning_curve is not None:
        try:
            if hasattr(model, 'loss_curve_'):
                plot_learning_curve(model.loss_curve_, save_path=lc_path)
            else:
                print("[-] Model does not have a recorded loss_curve_ attribute.")
        except Exception as e:
            print(f"[-] Could not save learning curve plot: {e}")

    # 4d. Multi-class ROC Curves Plot
    print(f"[*] Generating multi-class ROC curves...")
    if plot_roc_curves is not None:
        try:
            plot_roc_curves(y_test, metrics['y_pred_proba'], save_path=roc_path)
        except Exception as e:
            print(f"[-] Could not save ROC curves plot: {e}")

    # 4e. Feature Importance Plot
    print(f"[*] Generating feature importance plot...")
    if plot_feature_importance is not None:
        try:
            plot_feature_importance(model, feature_names=feature_names, save_path=fi_path)
        except Exception as e:
            print(f"[-] Could not save feature importance plot: {e}")

    # Summary
    print("\n=================================================================================")
    print("                             EVALUATION COMPLETED                                ")
    print("=================================================================================")
    if isinstance(metrics, dict) and 'accuracy' in metrics:
        print(f"  Test Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Plots Updated  : {figures_dir}/")
    print("  Interactive App: Run 'streamlit run app.py' to launch clinical predictor UI.")
    print("=================================================================================\n")


if __name__ == '__main__':
    main()
