"""
train_utils.py
==============
Training Utilities for Multilayer Perceptrons with Full-Batch Gradient Descent.

================================================================================
THEORETICAL FOUNDATION: GRADIENT DESCENT VARIANTS
================================================================================
In training Artificial Neural Networks, the objective function (loss function)
measures how far predictions are from ground truth labels across the training set:
    L(theta) = (1 / N) * sum_{i=1}^N loss(f(x_i; theta), y_i)

The weights (theta) are updated iteratively in the direction of steepest descent:
    theta_{t+1} = theta_t - alpha * grad(L(theta_t))

The size of the sample subset used to compute grad(L) distinguishes three main paradigms:

1. STOCHASTIC GRADIENT DESCENT (SGD - pure):
   - Batch size = 1
   - The gradient is computed from a single observation at a time.
   - Pros: Fast per update, low memory usage, noisy trajectory helps escape saddle points.
   - Cons: Highly erratic gradient updates; fails to leverage vectorized hardware; never
     smoothly settles into the exact minimum without aggressive learning rate decay.

2. MINI-BATCH GRADIENT DESCENT:
   - Batch size = k (typically 32, 64, 128, or 256)
   - The gradient is computed over a small partition of the dataset.
   - Pros: Standard compromise used in modern deep learning (Adam, RMSprop, mini-batch SGD).
     Enables parallelized GPU matrix operations while maintaining stochastic regularization.
   - Cons: Requires tuning of batch size; gradients are still approximations of the true surface.

3. FULL-BATCH GRADIENT DESCENT (Batch GD - used in this project):
   - Batch size = N (the ENTIRE training dataset, e.g., 202,944 samples)
   - The true gradient is computed by evaluating the loss and backpropagating across
     every single training sample before making ANY parameter update.
   - Pros: True, deterministic gradient vector; smooth and monotonic convergence along
     the actual loss landscape; zero variance in the gradient estimation.
   - Cons: Computationally demanding per epoch since all N forward and backward passes
     must complete for just one parameter update step.

In scikit-learn's `MLPClassifier`, setting `batch_size = X_train.shape[0]` explicitly
enforces Full-Batch Gradient Descent.
================================================================================
"""

import time
import numpy as np


def train_model(model, X_train, y_train, sample_weight=None):
    """
    Trains the given MLPClassifier using true Full-Batch Gradient Descent.
    
    To implement full-batch gradient descent in scikit-learn:
    - We set `model.batch_size = X_train.shape[0]` (the total number of training records).
    - When `batch_size` equals the training set size, each epoch comprises exactly
      ONE forward pass and ONE backward pass over the complete training set, yielding
      a single exact parameter update per epoch.
      
    Args:
        model: An instance of sklearn.neural_network.MLPClassifier.
        X_train (np.ndarray or pd.DataFrame): Scaled training features.
        y_train (np.ndarray or pd.Series): Training target labels.
        sample_weight (np.ndarray, optional): Per-sample weights for class-weighted cross entropy.
        
    Returns:
        MLPClassifier: The fitted neural network model.
    """
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
    
    # ENFORCE FULL-BATCH GRADIENT DESCENT:
    # Overriding batch_size to the exact count of training instances
    model.batch_size = int(n_samples)
    
    print("\n========================================================")
    print(" Starting Full-Batch Gradient Descent Training")
    print("========================================================")
    print(f" Training Samples (N) : {n_samples:,}")
    print(f" Features per Sample  : {n_features}")
    print(f" Batch Size Enforced  : {model.batch_size:,} (Full Dataset = 1 Batch)")
    print(f" Optimization Solver  : {model.solver}")
    print(f" Learning Rate (alpha): {model.learning_rate_init}")
    print(f" Max Iterations       : {model.max_iter} epochs")
    if hasattr(model, 'class_weight') and model.class_weight is not None:
        print(f" Class Weighting      : {model.class_weight}")
    elif sample_weight is not None:
        print(f" Sample Weights       : Provided (custom array)")
    else:
        print(f" Class Weighting      : None (Standard Unweighted)")
    print("========================================================\n")
    print("Beginning backpropagation iterations (one gradient step per epoch)...")
    
    start_time = time.time()
    
    # Fit the Multilayer Perceptron
    # With batch_size = n_samples, each iteration printed by verbose=True is 1 full-batch epoch
    model.fit(X_train, y_train, sample_weight=sample_weight)
    
    elapsed_time = time.time() - start_time
    
    print("\n========================================================")
    print(" Training Completed Successfully!")
    print("========================================================")
    print(f" Elapsed Training Time : {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} mins)")
    print(f" Total Epochs Run      : {model.n_iter_} / {model.max_iter}")
    
    if hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
        initial_loss = model.loss_curve_[0]
        final_loss = model.loss_curve_[-1]
        loss_reduction = initial_loss - final_loss
        pct_reduction = (loss_reduction / initial_loss) * 100 if initial_loss != 0 else 0
        
        print(f" Initial Training Loss : {initial_loss:.6f}")
        print(f" Final Training Loss   : {final_loss:.6f}")
        print(f" Absolute Loss Delta   : {loss_reduction:.6f} (-{pct_reduction:.2f}%)")
    print("========================================================\n")
    
    return model


def get_training_history(model):
    """
    Extracts the epoch-by-epoch loss trajectory from a fitted MLPClassifier.
    
    Args:
        model: Fitted MLPClassifier instance containing `loss_curve_`.
        
    Returns:
        dict: Dictionary containing:
            - 'loss': List or np.ndarray of loss values at each epoch.
            - 'epochs': List of integer epoch numbers [1, 2, ..., total_epochs].
            
    Raises:
        ValueError: If the model is unfitted or lacks loss_curve_.
    """
    if not hasattr(model, 'loss_curve_') or model.loss_curve_ is None:
        raise ValueError(
            "The model has not recorded a loss curve. Ensure it is trained with "
            "train_model() before calling get_training_history()."
        )
    
    loss_curve = list(model.loss_curve_)
    epochs = list(range(1, len(loss_curve) + 1))
    
    return {
        'loss': loss_curve,
        'epochs': epochs
    }


if __name__ == '__main__':
    print("train_utils.py loaded successfully.")
