"""
preprocessing.py
================
Data preprocessing module for Diabetes Risk Classification Using MLP.

This module provides essential data preparation steps:
1. Stratified train/test splitting to preserve class proportions.
2. Standard feature scaling (StandardScaler) fitted strictly on the training partition
   to prevent data leakage into the test set.
3. Class distribution verification to inspect dataset balance.
4. End-to-end preprocessing pipeline combining all steps into a single workflow.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset, get_class_names, get_feature_names


def check_class_distribution(y, title='Dataset'):
    """
    Computes and displays the frequency and percentage breakdown for each target class.
    
    This is especially important for the CDC Diabetes dataset because classes
    are significantly imbalanced (e.g., Prediabetes is a minority class).
    
    Args:
        y (pd.Series or np.ndarray): Target class labels.
        title (str): Descriptive label for the current data split being analyzed.
        
    Returns:
        pd.DataFrame: Summary table with counts, percentages, and class descriptions.
    """
    class_map = get_class_names()
    y_series = pd.Series(y)
    
    counts = y_series.value_counts().sort_index()
    percentages = (y_series.value_counts(normalize=True).sort_index()) * 100
    
    summary_df = pd.DataFrame({
        'Class Label': counts.index,
        'Class Name': [class_map.get(lbl, f'Class {lbl}') for lbl in counts.index],
        'Count': counts.values,
        'Percentage (%)': percentages.values.round(2)
    })
    
    print(f"\n==========================================")
    print(f" Class Distribution: {title}")
    print(f" Total Samples: {len(y_series):,}")
    print(f"==========================================")
    for _, row in summary_df.iterrows():
        print(f"  [{int(row['Class Label'])}] {row['Class Name']:<12}: "
              f"{int(row['Count']):>7,} ({row['Percentage (%)']:>5.2f}%)")
    print(f"==========================================\n")
    
    return summary_df


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Splits the dataset into training and testing subsets using scikit-learn's train_test_split.
    
    Args:
        X (pd.DataFrame or np.ndarray): Input feature matrix.
        y (pd.Series or np.ndarray): Target label vector.
        test_size (float): Fraction of data reserved for the test set (default 0.20 for an 80/20 split).
        random_state (int): Seed for reproducibility.
        stratify (bool): If True, preserves the class distribution ratios in both train and test splits.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_target = y if stratify else None
    
    print(f"Splitting dataset: {int((1-test_size)*100)}% Train / {int(test_size*100)}% Test "
          f"(stratified={stratify}, random_state={random_state})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target
    )
    
    print(f"Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Test set shape : X_test={X_test.shape}, y_test={y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Scales features to zero mean and unit variance using StandardScaler.
    
    CRITICAL ACADEMIC NOTE:
    The scaler MUST be fit only on the training set (X_train) and then used to
    transform both X_train and X_test. Fitting the scaler on the combined data
    or test data causes 'data leakage' (information leakage from test to train).
    
    Args:
        X_train (pd.DataFrame or np.ndarray): Unscaled training features.
        X_test (pd.DataFrame or np.ndarray): Unscaled testing features.
        
    Returns:
        tuple (np.ndarray, np.ndarray, StandardScaler):
            - X_train_scaled: Scaled training feature matrix.
            - X_test_scaled: Scaled testing feature matrix.
            - scaler: The fitted StandardScaler instance for future inference.
    """
    print("Fitting StandardScaler on training data and transforming train & test sets...")
    scaler = StandardScaler()
    
    # Fit strictly on training data, then transform training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the parameters (mean, std) learned from train
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Features scaled successfully.")
    print(f"Train feature mean (approx): {np.mean(X_train_scaled):.4f}, std: {np.std(X_train_scaled):.4f}")
    print(f"Test feature mean (approx) : {np.mean(X_test_scaled):.4f}, std: {np.std(X_test_scaled):.4f}")
    
    return X_train_scaled, X_test_scaled, scaler


def save_scaler(scaler, filepath='models/scaler.pkl'):
    """
    Saves a fitted StandardScaler instance to disk using joblib.
    
    Args:
        scaler (StandardScaler): The fitted scaler.
        filepath (str): Destination file path.
    """
    import joblib
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"Scaler saved successfully to: {filepath}")


def load_scaler(filepath='models/scaler.pkl'):
    """
    Loads a saved StandardScaler instance from disk using joblib.
    
    Args:
        filepath (str): Path to the saved scaler file.
        
    Returns:
        StandardScaler: The loaded scaler object.
    """
    import joblib
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Scaler file not found at '{filepath}'.")
    scaler = joblib.load(filepath)
    print(f"Scaler loaded successfully from: {filepath}")
    return scaler


def preprocess_pipeline(data_or_path=None, test_size=0.2, random_state=42):
    """
    Executes the complete end-to-end preprocessing pipeline:
    1. Loads dataset from CSV, DataFrame, or (X, y) tuple.
    2. Performs stratified train/test split.
    3. Analyzes and displays class distributions.
    4. Applies standard feature scaling.
    
    Args:
        data_or_path (str, pd.DataFrame, tuple, optional): 
            Path to CSV, or loaded DataFrame, or (X, y) tuple.
            Defaults to 'data/diabetes_012_health_indicators_BRFSS2015.csv'.
        test_size (float): Fraction of data for testing (default 0.2).
        random_state (int): Random seed for reproducible splitting.
        
    Returns:
        dict: Dictionary containing:
            - 'X_train': Scaled training feature matrix (np.ndarray)
            - 'X_test': Scaled testing feature matrix (np.ndarray)
            - 'y_train': Training target labels (pd.Series or np.ndarray)
            - 'y_test': Testing target labels (pd.Series or np.ndarray)
            - 'scaler': Fitted StandardScaler instance
            - 'feature_names': List of 21 feature names
            - 'class_names': Mapping dictionary of target classes
            - 'X_train_raw': Original unscaled training DataFrame
            - 'X_test_raw': Original unscaled testing DataFrame
    """
    print("\n==========================================")
    print(" Starting Preprocessing Pipeline")
    print("==========================================")
    
    # 1. Resolve Dataset Input
    if data_or_path is None or isinstance(data_or_path, str):
        filepath = data_or_path if data_or_path is not None else os.path.join('data', 'diabetes_012_health_indicators_BRFSS2015.csv')
        X, y = load_dataset(filepath=filepath)
    elif isinstance(data_or_path, tuple) and len(data_or_path) == 2:
        X, y = data_or_path
    elif isinstance(data_or_path, pd.DataFrame):
        target_col = 'Diabetes_012'
        if target_col in data_or_path.columns:
            y = data_or_path[target_col].astype(int)
            X = data_or_path.drop(columns=[target_col])
        else:
            raise KeyError(f"Target column '{target_col}' not found in provided DataFrame.")
    else:
        raise TypeError(f"Unsupported data input type for preprocess_pipeline: {type(data_or_path)}")
        
    feature_names = list(X.columns) if hasattr(X, 'columns') else get_feature_names()
    class_names = get_class_names()
    
    # 2. Check full dataset distribution
    check_class_distribution(y, title="Full Dataset")
    
    # 3. Stratified Train-Test Split
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=True
    )
    
    # 4. Check split distributions to verify stratification worked
    check_class_distribution(y_train, title="Train Set Split")
    check_class_distribution(y_test, title="Test Set Split")
    
    # 5. Standard Feature Scaling
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_raw, X_test_raw)
    
    print("Preprocessing pipeline completed successfully!\n")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names,
        'class_names': class_names,
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw
    }


if __name__ == '__main__':
    print("preprocessing.py module loaded successfully.")
