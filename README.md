# Diabetes Risk Classification Using Multilayer Perceptron (MLP)

> **CSEN3011 — Artificial Neural Networks**  
> Academic Course Project | Multi-Class Risk Stratification Pipeline

---

## 📌 Project Overview
This project develops an end-to-end Artificial Neural Network (ANN)—specifically a feedforward **Multilayer Perceptron (MLP)**—to predict and categorize individual diabetes risk into three distinct clinical states: **Healthy (0)**, **Prediabetes (1)**, and **Diagnosed Diabetes (2)**. Built upon public health survey indicators from the Centers for Disease Control and Prevention (CDC), the system standardizes heterogeneous features, performs full-batch gradient descent optimization, generates comprehensive evaluation analytics, and provides an interactive web dashboard for real-time risk assessment.

---

## 📊 Dataset Information
- **Dataset Name:** CDC Diabetes Health Indicators
- **Source Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- **Total Records:** 253,680 survey responses
- **Features ($D$):** 21 clinical, lifestyle, and demographic features:
  - `HighBP`, `HighChol`, `CholCheck`, `BMI`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`, `Sex`, `Age`, `Education`, `Income`
- **Target Variable (`Diabetes_012`):**
  - `0`: Healthy / No Diabetes (~84.2%)
  - `1`: Prediabetes (~1.8%)
  - `2`: Diabetes (~13.9%)

---

## 📁 Project Structure

```text
Diabetes-Risk-Classification-ANN/
│
├── data/
│   └── diabetes_binary_health_indicators_BRFSS2015.csv  # CDC BRFSS dataset
│
├── docs/
│   ├── project_report.md         # Full academic project report template (CSEN3011)
│   └── viva_questions.md         # 22 curated Viva voce questions and detailed answers
│
├── models/
│   ├── mlp_model.joblib          # Serialized trained MLP model
│   └── scaler.joblib             # Fitted StandardScaler object
│
├── plots/
│   ├── confusion_matrix.png      # 3x3 Confusion matrix visualization
│   ├── learning_curve.png        # Full-batch gradient descent loss curve
│   └── roc_curves.png            # Multi-class One-vs-Rest ROC-AUC curves
│
├── src/
│   ├── __init__.py               # Package marker
│   ├── preprocessing.py          # Scaling, stratified splitting, and distribution checks
│   ├── model.py                  # MLP model building, training, and persistence
│   └── evaluation.py             # Classification report, metric logs, and plotters
│
├── tests/
│   ├── __init__.py               # Package marker
│   ├── test_preprocessing.py     # Pytest unit tests for preprocessing module
│   └── test_model.py             # Pytest unit tests for MLP model module
│
├── app.py                        # Streamlit web application for real-time risk inference
├── evaluate.py                   # Model evaluation, metric logging, and plot generator
├── train.py                      # Full-batch MLP model training script
├── requirements.txt              # Project Python dependencies
└── README.md                     # Project documentation and quickstart guide
```

---

## ⚙️ Installation & Setup

Follow these steps to set up the project environment on your local system:

### 1. Clone or Download the Repository
```bash
git clone https://github.com/your-username/Diabetes-Risk-Classification-ANN.git
cd Diabetes-Risk-Classification-ANN
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) and place the CSV file inside the `data/` folder:
```text
data/diabetes_binary_health_indicators_BRFSS2015.csv
```

---

## 🚀 Usage Instructions

### 1. Model Training (Full-Batch Gradient Descent)
Train the 3-layer MLP classifier using full-batch gradient descent (`solver='sgd'` with batch size matching training samples):
```bash
python train.py
```
*Trained model weights and scaler will be saved to `models/mlp_model.joblib` and `models/scaler.joblib`.*

### 2. Model Evaluation
Evaluate the model on the stratified holdout test partition ($N_{\text{test}} = 50,736$):
```bash
python evaluate.py
```
*Evaluation reports and figures (`confusion_matrix.png`, `learning_curve.png`, `roc_curves.png`) are generated in the `plots/` directory.*

### 3. Interactive Web Application
Launch the Streamlit web dashboard for real-time patient assessment:
```bash
streamlit run app.py
```
*Access the local web dashboard at `http://localhost:8501` in your browser.*

### 4. Automated Testing
Run the Pytest unit testing suite:
```bash
pytest tests/ -v
```

---

## 🛠️ Technologies & Libraries

- **Language:** Python 3.10+
- **Numerical & Data Processing:** NumPy, Pandas
- **Machine Learning & Neural Networks:** scikit-learn (`MLPClassifier`, `StandardScaler`)
- **Data Visualization:** Matplotlib, Seaborn
- **Web Dashboard:** Streamlit
- **Testing & Quality Assurance:** Pytest
- **Model Serialization:** Joblib

---

## 📈 Experimental Results

> **[TODO: Add finalized empirical evaluation metrics after training completes]**

| Metric | Score |
|---|---|
| **Test Accuracy** | `[TODO: e.g., XX.X%]` |
| **Macro Precision** | `[TODO: e.g., 0.XX]` |
| **Macro Recall** | `[TODO: e.g., 0.XX]` |
| **Macro F1-Score** | `[TODO: e.g., 0.XX]` |
| **Macro ROC-AUC** | `[TODO: e.g., 0.XX]` |

*See [`docs/project_report.md`](docs/project_report.md) for full breakdown and per-class precision/recall curves.*

---

## 📄 License
This project is licensed under the **MIT License** (or Academic Educational Use). See LICENSE file for details.

---

## 🙏 Acknowledgements
- **UCI Machine Learning Repository:** For hosting and facilitating public access to the dataset.
- **Centers for Disease Control and Prevention (CDC):** For conducting and maintaining the Behavioral Risk Factor Surveillance System (BRFSS).
- **Course Staff:** Instructor and teaching assistants of **CSEN3011: Artificial Neural Networks**.
