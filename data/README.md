# Data Directory

## About
This folder stores the CDC Diabetes Health Indicators dataset.

## Dataset Source
- **Name:** CDC Diabetes Health Indicators
- **URL:** https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- **Kaggle:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

## Expected File
After downloading, place the CSV file here:
```
data/diabetes_012_health_indicators_BRFSS2015.csv
```

## Dataset Details
- **Records:** 253,680
- **Features:** 21 input features + 1 target column
- **Target column:** `Diabetes_012`
  - `0` = Healthy (no diabetes)
  - `1` = Prediabetes
  - `2` = Diabetes

## Note
The CSV file is listed in `.gitignore` and will **not** be committed to Git.
You must download it yourself (see `src/data_loader.py`).
