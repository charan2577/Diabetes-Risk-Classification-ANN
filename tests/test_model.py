"""
tests/test_model.py
Unit tests for the Multilayer Perceptron (MLP) model module in Diabetes Risk Classification.
All tests use synthetic/dummy data for rapid and isolated execution.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

# Ensure project root is available in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import create_mlp_model, save_model, load_model


def test_create_mlp_model():
    """
    Test model instantiation.
    Verifies that create_mlp_model returns an MLPClassifier configured with the 'sgd' solver.
    """
    model = create_mlp_model()
    assert isinstance(model, MLPClassifier), f"Expected MLPClassifier instance, got {type(model)}"
    assert model.solver == "sgd", f"Expected solver 'sgd' for full-batch gradient descent, got '{model.solver}'"


def test_model_hidden_layers():
    """
    Test custom hidden layer architecture specification.
    Verifies that custom hidden_layer_sizes tuples are correctly applied to the model.
    """
    custom_layers = (64, 32)
    model = create_mlp_model(hidden_layer_sizes=custom_layers)
    assert model.hidden_layer_sizes == custom_layers, (
        f"Expected hidden_layer_sizes {custom_layers}, got {model.hidden_layer_sizes}"
    )

    # Also verify default architecture (128, 64, 32)
    default_model = create_mlp_model()
    assert default_model.hidden_layer_sizes == (128, 64, 32), (
        f"Expected default hidden_layer_sizes (128, 64, 32), got {default_model.hidden_layer_sizes}"
    )


def test_save_and_load_model(tmp_path):
    """
    Test model serialization and deserialization using tmp_path fixture.
    Verifies that a saved model can be loaded back and produces identical predictions.
    """
    np.random.seed(42)
    X_dummy = np.random.randn(30, 21)
    y_dummy = np.array([0, 1, 2] * 10)

    # Train a lightweight model on dummy data
    model = create_mlp_model(hidden_layer_sizes=(16, 8), max_iter=20, random_state=42)
    model.fit(X_dummy, y_dummy)

    # Save to temporary path
    save_path = tmp_path / "test_mlp_model.joblib"
    save_model(model, str(save_path))

    assert save_path.exists(), f"Model file was not created at {save_path}"

    # Load model and verify integrity
    loaded_model = load_model(str(save_path))
    assert isinstance(loaded_model, MLPClassifier), "Loaded object is not an MLPClassifier instance"

    original_preds = model.predict(X_dummy)
    loaded_preds = loaded_model.predict(X_dummy)
    np.testing.assert_array_equal(
        original_preds,
        loaded_preds,
        err_msg="Loaded model predictions do not match original model predictions",
    )


def test_model_can_fit():
    """
    Test model fitting on synthetic 21-feature multi-class dataset.
    Verifies that the MLPClassifier converges, assigns output dimensions, and records loss_curve_.
    """
    np.random.seed(42)
    # 100 samples, 21 features, 3 classes (0, 1, 2)
    X_dummy = np.random.randn(100, 21)
    y_dummy = np.random.choice([0, 1, 2], size=100)

    model = create_mlp_model(hidden_layer_sizes=(32, 16), max_iter=20, random_state=42)
    model.fit(X_dummy, y_dummy)

    # Verify that the model was fitted and loss trajectory was tracked
    assert hasattr(model, "loss_curve_"), "Fitted model is missing loss_curve_ attribute"
    assert len(model.loss_curve_) > 0, "Loss curve must contain recorded loss values per iteration"
    assert hasattr(model, "classes_"), "Fitted model is missing classes_ attribute"
    assert len(model.classes_) == 3, f"Expected 3 target classes, found {len(model.classes_)}"
    assert model.n_outputs_ == 3, f"Expected n_outputs_ to be 3, found {model.n_outputs_}"
    assert model.batch_size == 100, f"Expected batch_size dynamically set to 100, got {model.batch_size}"


def test_model_class_weight():
    """
    Test model fitting with class_weight='balanced' on an imbalanced dataset.
    Verifies that class-weighted cross-entropy executes properly under full-batch gradient descent.
    """
    np.random.seed(42)
    # 100 samples with strong imbalance: 80 class 0, 5 class 1, 15 class 2
    X_dummy = np.random.randn(100, 21)
    y_dummy = np.array([0] * 80 + [1] * 5 + [2] * 15)

    model = create_mlp_model(
        hidden_layer_sizes=(16, 8),
        max_iter=15,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_dummy, y_dummy)

    assert model.class_weight == "balanced"
    assert hasattr(model, "loss_curve_")
    assert len(model.loss_curve_) > 0
    assert model.batch_size == 100
    assert len(model.classes_) == 3

