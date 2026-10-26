"""
tests/test_preprocessing.py
Unit tests for data preprocessing module in Diabetes Risk Classification.
All tests use synthetic/dummy data for fast, reproducible execution without
requiring the full external CDC dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is available in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    scale_features,
    split_data,
    check_class_distribution,
)


def test_scale_features():
    """
    Test feature scaling using StandardScaler.
    Verifies that scaled training features have mean approximately 0 and standard deviation approximately 1.
    """
    np.random.seed(42)
    # Generate synthetic training and testing feature matrices (100 samples, 21 features)
    X_train = np.random.normal(loc=25.0, scale=8.0, size=(100, 21))
    X_test = np.random.normal(loc=25.0, scale=8.0, size=(30, 21))

    result = scale_features(X_train, X_test)

    # Unpack output whether function returns (X_train_scaled, X_test_scaled) or (X_train_scaled, X_test_scaled, scaler)
    if isinstance(result, tuple):
        X_train_scaled = result[0]
        X_test_scaled = result[1]
    else:
        X_train_scaled = result
        X_test_scaled = None

    # Check shapes
    assert X_train_scaled.shape == (100, 21), f"Expected shape (100, 21), got {X_train_scaled.shape}"
    if X_test_scaled is not None:
        assert X_test_scaled.shape == (30, 21), f"Expected shape (30, 21), got {X_test_scaled.shape}"

    # Calculate column-wise mean and standard deviation on training set
    train_mean = np.mean(X_train_scaled, axis=0)
    train_std = np.std(X_train_scaled, axis=0)

    # Standardized features should have mean ~ 0 and std ~ 1
    np.testing.assert_allclose(train_mean, 0.0, atol=1e-6, err_msg="Scaled training mean is not 0")
    np.testing.assert_allclose(train_std, 1.0, atol=1e-6, err_msg="Scaled training std is not 1")


def test_split_data():
    """
    Test dataset splitting into train and test partitions.
    Verifies that split sizes match the specified test_size ratio.
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 21

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.choice([0, 1, 2], size=n_samples, p=[0.8, 0.05, 0.15]))

    test_size = 0.2
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=42)

    expected_test_samples = int(n_samples * test_size)
    expected_train_samples = n_samples - expected_test_samples

    assert len(X_train) == expected_train_samples, f"Expected {expected_train_samples} train samples, got {len(X_train)}"
    assert len(X_test) == expected_test_samples, f"Expected {expected_test_samples} test samples, got {len(X_test)}"
    assert len(y_train) == expected_train_samples, "y_train size mismatch"
    assert len(y_test) == expected_test_samples, "y_test size mismatch"


def test_check_class_distribution(capsys):
    """
    Test class distribution display function.
    Verifies that the function executes without exceptions and prints informative output.
    """
    np.random.seed(42)
    # Synthetic target with imbalanced classes
    y = pd.Series(
        np.random.choice([0, 1, 2], size=300, p=[0.84, 0.02, 0.14]),
        name="Diabetes_012",
    )
    df = pd.DataFrame({"Diabetes_012": y})

    # Support invocation with either Series or DataFrame
    try:
        check_class_distribution(y)
    except (TypeError, KeyError):
        check_class_distribution(df, target_col="Diabetes_012")

    captured = capsys.readouterr()
    output = captured.out + captured.err

    # Verify something was printed and class values appear in output
    assert len(output.strip()) > 0, "check_class_distribution did not print any output"
    assert any(str(c) in output for c in [0, 1, 2]), "Output does not reference target class labels"


def test_split_data_stratified():
    """
    Test that stratified splitting preserves the class distribution proportions in both train and test sets.
    """
    np.random.seed(42)
    n_samples = 1000
    n_features = 21

    X = np.random.randn(n_samples, n_features)
    # Strongly imbalanced class distribution similar to CDC BRFSS dataset
    # Class 0: 84%, Class 1: 3%, Class 2: 13%
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.84, 0.03, 0.13])

    try:
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=True
        )
    except TypeError:
        # Default behavior of split_data in project is stratified
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.2, random_state=42
        )

    # Compute proportions
    original_props = pd.Series(y).value_counts(normalize=True).sort_index()
    train_props = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_props = pd.Series(y_test).value_counts(normalize=True).sort_index()

    # Stratification must keep class proportions within a tight threshold (~2.5%)
    for cls in [0, 1, 2]:
        assert abs(train_props[cls] - original_props[cls]) < 0.025, (
            f"Train proportion for class {cls} deviates: {train_props[cls]} vs {original_props[cls]}"
        )
        assert abs(test_props[cls] - original_props[cls]) < 0.025, (
            f"Test proportion for class {cls} deviates: {test_props[cls]} vs {original_props[cls]}"
        )
