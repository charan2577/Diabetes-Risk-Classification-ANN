"""
data_loader.py
==============
Data loading module for the Diabetes Risk Classification project.

This module provides functions to:
1. Load the CDC Diabetes Health Indicators dataset from a CSV file.
2. Download the dataset directly from the UCI Machine Learning Repository (ID: 891).
3. Retrieve standard feature names and target class mappings.

Dataset Details:
- Name: CDC Diabetes Health Indicators (BRFSS 2015)
- Records: 253,680
- Features: 21 clinical, behavioral, and demographic indicators
- Target: Diabetes_012 (0 = Healthy, 1 = Prediabetes, 2 = Diabetes)
"""

import os
import pandas as pd


def get_feature_names():
    """
    Returns the list of 21 feature names present in the dataset.
    
    Returns:
        list of str: Ordered list of feature column names.
    """
    return [
        'HighBP',
        'HighChol',
        'CholCheck',
        'BMI',
        'Smoker',
        'Stroke',
        'HeartDiseaseorAttack',
        'PhysActivity',
        'Fruits',
        'Veggies',
        'HvyAlcoholConsump',
        'AnyHealthcare',
        'NoDocbcCost',
        'GenHlth',
        'MentHlth',
        'PhysHlth',
        'DiffWalk',
        'Sex',
        'Age',
        'Education',
        'Income'
    ]


def get_class_names():
    """
    Returns a dictionary mapping numeric target labels to human-readable names.
    
    Returns:
        dict: Mapping of {0: 'Healthy', 1: 'Prediabetes', 2: 'Diabetes'}
    """
    return {
        0: 'Healthy',
        1: 'Prediabetes',
        2: 'Diabetes'
    }


def download_dataset(output_dir='data'):
    """
    Downloads the CDC Diabetes Health Indicators dataset from the UCI ML Repository
    using the 'ucimlrepo' package and saves it locally as a CSV file.
    
    Args:
        output_dir (str): Directory where the downloaded CSV will be stored.
        
    Returns:
        str: Path to the saved CSV file.
    """
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise ImportError(
            "The 'ucimlrepo' package is required to download the dataset.\n"
            "Install it via: pip install ucimlrepo"
        ) from e

    print("Fetching dataset from UCI Machine Learning Repository (Dataset ID: 891)...")
    cdc_diabetes = fetch_ucirepo(id=891)

    # Extract features (X) and targets (y)
    X = cdc_diabetes.data.features
    y = cdc_diabetes.data.targets

    # Combine target and features into a single DataFrame
    df = pd.concat([y, X], axis=1)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'diabetes_012_health_indicators_BRFSS2015.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully downloaded and saved to: {output_path}")
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return output_path


def load_dataset(filepath='data/diabetes_012_health_indicators_BRFSS2015.csv'):
    """
    Loads the diabetes dataset from a CSV file and separates features and target.
    
    Args:
        filepath (str): Relative or absolute path to the dataset CSV file.
        
    Returns:
        tuple (pd.DataFrame, pd.Series):
            - X: DataFrame containing the 21 predictor features.
            - y: Series containing the multi-class target 'Diabetes_012'.
            
    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        KeyError: If the target column 'Diabetes_012' is not found.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset file not found at '{filepath}'.\n"
            "Please ensure the CSV file is placed in the 'data/' folder, or run:\n"
            "  from src.data_loader import download_dataset\n"
            "  download_dataset()\n"
            "to automatically download it from UCI."
        )

    print(f"Loading dataset from: {filepath} ...")
    df = pd.read_csv(filepath)

    target_col = 'Diabetes_012'
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Convert target to integer type in case it was loaded as float
    df[target_col] = df[target_col].astype(int)

    # Separate target (y) and input features (X)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    print("\n--- Dataset Loading Summary ---")
    print(f"Total Records (Samples) : {X.shape[0]:,}")
    print(f"Number of Features      : {X.shape[1]}")
    print(f"Features list           : {list(X.columns)}")
    print(f"Missing values in data  : {df.isnull().sum().sum()}")
    print("--------------------------------\n")

    return X, y


if __name__ == '__main__':
    # Quick self-test when running this module directly
    print("Testing data_loader module...")
    classes = get_class_names()
    features = get_feature_names()
    print(f"Registered Classes ({len(classes)}): {classes}")
    print(f"Registered Features ({len(features)}): {features}")
