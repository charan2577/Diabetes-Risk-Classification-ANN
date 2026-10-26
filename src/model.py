"""
model.py
========
Multilayer Perceptron (MLP) architecture definition and management module
for Diabetes Risk Classification.

Academic Project Objective:
Demonstrate the fundamental principles of Artificial Neural Networks (ANN)
trained with true Full-Batch Gradient Descent (Batch GD).

This module provides:
1. `create_mlp_model`: Configures an MLPClassifier using the 'sgd' solver.
2. `get_model_summary`: Prints layer architectures, parameter counts, and hyperparameter settings.
3. `save_model`: Persists trained weights and metadata using joblib.
4. `load_model`: Restores a pre-trained model for inference or evaluation.
"""

import os
import joblib
from sklearn.neural_network import MLPClassifier


class FullBatchMLPClassifier(MLPClassifier):
    """
    Multilayer Perceptron classifier specifically engineered to guarantee
    Full-Batch Gradient Descent (Batch GD).
    
    Academic Rationale:
    In standard scikit-learn MLPClassifier, `batch_size='auto'` defaults to
    min(200, n_samples). For a 202,944-sample dataset, that would perform
    1,015 mini-batch updates per epoch instead of the required single full-batch
    gradient update.
    
    FullBatchMLPClassifier guarantees that:
    1. `solver='sgd'` is strictly enforced (Adam is disallowed).
    2. A single batch containing 100% of the training dataset is formed at fit-time.
    3. Exactly ONE parameter update occurs per epoch.
    4. The gradient is the exact derivative of the empirical risk over all N samples.
    5. Zero mini-batch iterations occur.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="sgd",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.01,
        power_t=0.5,
        max_iter=200,
        shuffle=False,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        class_weight=None,
        **kwargs
    ):
        if solver != 'sgd':
            raise ValueError(
                f"FullBatchMLPClassifier strictly requires solver='sgd' (Batch Gradient Descent). "
                f"Disallowed optimizer: '{solver}'."
            )
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver='sgd',
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
            **kwargs
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        # Dynamically compute batch size from actual training partition
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        self.batch_size = int(n_samples)
        self.shuffle = False  # In full-batch GD, intra-batch shuffling is redundant
        self.early_stopping = False
        
        # Calculate sample weights strictly from training labels if class_weight is specified
        if sample_weight is None and self.class_weight is not None:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight(self.class_weight, y)
            
        return super().fit(X, y, sample_weight=sample_weight)


def create_mlp_model(
    hidden_layers=(128, 64, 32),
    hidden_layer_sizes=None,
    activation='relu',
    learning_rate_init=0.01,
    max_iter=200,
    batch_size=None,
    momentum=0.9,
    nesterovs_momentum=True,
    random_state=42,
    class_weight=None,
    **kwargs
):
    """
    Initializes a Multilayer Perceptron (FullBatchMLPClassifier) configured for
    Full-Batch Gradient Descent.
    
    Academic Note on Optimization Solver & Batch Size:
    - `solver='sgd'` is strictly enforced (classical gradient descent, NOT Adam).
    - `batch_size` dynamically scales to the entire training set size at fit-time.
    - `early_stopping=False` ensures all N training records contribute to gradients.
    - `class_weight` enables class-weighted cross-entropy (e.g. 'balanced') to counter
      severe class imbalance (e.g. 1.8% minority class) during full-batch updates.
      
    Args:
        hidden_layers (tuple): Tuple specifying neuron counts in each hidden layer.
                               Default is (128, 64, 32), a 3-hidden-layer network.
        hidden_layer_sizes (tuple, optional): Alias for hidden_layers (scikit-learn compatibility).
        activation (str): Activation function for hidden layers ('relu', 'tanh', 'logistic').
        learning_rate_init (float): Step size (alpha) for gradient descent weight updates.
        max_iter (int): Maximum number of full-batch epochs to execute.
        batch_size (int, optional): Initial batch size (auto-calculated from X.shape[0] at fit).
        momentum (float): Momentum factor for gradient descent (default 0.9). Set to 0.0 for pure vanilla GD.
        nesterovs_momentum (bool): Whether to use Nesterov momentum.
        random_state (int): Random seed for reproducible weight initialization.
        class_weight (str, dict, optional): Weighting strategy for cross-entropy loss ('balanced' or dict).
        **kwargs: Additional parameters passed to FullBatchMLPClassifier.
        
    Returns:
        FullBatchMLPClassifier: Configured scikit-learn MLPClassifier instance.
    """
    print("\nInitializing Multilayer Perceptron (FullBatchMLPClassifier)...")
    
    # Resolve layer sizes (support both hidden_layer_sizes and hidden_layers)
    actual_layers = hidden_layer_sizes if hidden_layer_sizes is not None else hidden_layers
    actual_batch_size = batch_size if batch_size is not None else 'auto'
    
    model = FullBatchMLPClassifier(
        hidden_layer_sizes=actual_layers,
        activation=activation,
        solver='sgd',
        learning_rate='constant',
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        batch_size=actual_batch_size,
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        shuffle=False,        # In full-batch GD, shuffling data within an epoch has no effect
        random_state=random_state,
        verbose=True,         # Prints loss progression per epoch to console for educational clarity
        early_stopping=False, # Ensure execution for the intended number of iterations
        class_weight=class_weight,
        **kwargs
    )
    
    print("==========================================")
    print(" MLP Model Configuration")
    print("==========================================")
    print(f"  Architecture      : Input(21) -> {' -> '.join(str(h) for h in actual_layers)} -> Output(3)")
    print(f"  Hidden Layers     : {actual_layers}")
    print(f"  Activation        : {activation}")
    print(f"  Optimizer / Solver: sgd (Standard Gradient Descent)")
    print(f"  Learning Rate     : {learning_rate_init} (constant)")
    print(f"  Max Iterations    : {max_iter} epochs")
    print(f"  Momentum          : {momentum} (nesterov={nesterovs_momentum})")
    print(f"  Batch Setting     : Full-Batch GD (enforced dynamically at fit-time)")
    print(f"  Class Weighting   : {class_weight if class_weight is not None else 'None (Unweighted)'}")
    print("==========================================\n")
    
    return model


def get_model_summary(model):
    """
    Inspects and prints detailed structural parameters and weight counts of the MLP model.
    Works for both unfitted and fitted MLPClassifier models.
    
    Args:
        model (MLPClassifier): The neural network model to inspect.
    """
    print("\n==========================================")
    print(" Multilayer Perceptron Summary")
    print("==========================================")
    print(f"  Configured Hidden Layers: {model.hidden_layer_sizes}")
    print(f"  Activation Function     : {model.activation}")
    print(f"  Optimization Solver     : {model.solver}")
    print(f"  Initial Learning Rate   : {model.learning_rate_init}")
    print(f"  Configured Batch Size   : {model.batch_size}")
    print(f"  Max Iterations (Epochs) : {model.max_iter}")
    
    # Check if model has been fitted by checking for the 'coefs_' attribute
    if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
        total_params = 0
        print("\n  --- Trained Layer-by-Layer Parameters ---")
        for i, (w, b) in enumerate(zip(model.coefs_, model.intercepts_)):
            layer_params = w.size + b.size
            total_params += layer_params
            layer_type = "Hidden Layer" if i < len(model.coefs_) - 1 else "Output Layer"
            print(f"    Layer {i+1} ({layer_type}):")
            print(f"      - Weights shape : {w.shape} ({w.size:,} parameters)")
            print(f"      - Biases shape  : {b.shape} ({b.size:,} parameters)")
            print(f"      - Subtotal      : {layer_params:,} parameters")
            
        print(f"\n  Total Trainable Parameters : {total_params:,}")
        print(f"  Completed Iterations (Epochs): {model.n_iter_}")
        if hasattr(model, 'loss_'):
            print(f"  Final Training Loss        : {model.loss_:.6f}")
    else:
        print("\n  [Status: Model is unfitted. Train model to view parameter weights.]")
    print("==========================================\n")


def save_model(model, filepath='models/mlp_model.pkl'):
    """
    Saves a trained MLPClassifier model to disk using joblib.
    
    Args:
        model (MLPClassifier): Fitted model instance to save.
        filepath (str): Relative destination path for the serialized file.
        
    Returns:
        str: Path where the model was saved.
    """
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)
        
    print(f"Saving model to '{filepath}'...")
    joblib.dump(model, filepath)
    print("Model successfully saved!")
    return filepath


def load_model(filepath='models/mlp_model.pkl'):
    """
    Loads a saved MLPClassifier model from disk using joblib.
    
    Args:
        filepath (str): Path to the saved model file.
        
    Returns:
        MLPClassifier: Restored model instance.
        
    Raises:
        FileNotFoundError: If the model file is not found at the given path.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No saved model found at '{filepath}'. Please train and save a model first."
        )
    print(f"Loading model from '{filepath}'...")
    model = joblib.load(filepath)
    print("Model loaded successfully!")
    return model


if __name__ == '__main__':
    print("Testing model creation...")
    sample_model = create_mlp_model()
    get_model_summary(sample_model)
