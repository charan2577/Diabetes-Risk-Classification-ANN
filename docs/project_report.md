# Diabetes Risk Classification Using Multilayer Perceptrons (MLP)
## Course Project Report — CSEN3011: Artificial Neural Networks

---

### Academic Information & Submission Details
- **Course Title:** Artificial Neural Networks (CSEN3011)
- **Academic Term / Year:** [TODO: Insert Academic Term, e.g., Fall 2026 / Spring 2027]
- **Project Title:** Multi-Class Diabetes Risk Classification Using Full-Batch Multilayer Perceptrons
- **Student Name:** [TODO: Insert Student Full Name]
- **Student Registration / Roll ID:** [TODO: Insert Student ID / Roll Number]
- **Department:** Department of Computer Science & Engineering
- **Faculty Supervisor / Instructor:** [TODO: Insert Instructor Name]
- **Date of Submission:** [TODO: Insert Submission Date, e.g., October 25, 2026]

---

## 2. Abstract
> **[TODO: Abstract Instructions]**
> *Write a concise 150–200 word summary here after concluding experiments. Highlight: (1) Background context of early diabetes detection, (2) Dataset overview (CDC BRFSS with 253,680 records), (3) Model architecture (3-hidden-layer feedforward MLP: 21 → 128 → 64 → 32 → 3), (4) Full-batch gradient descent optimization, and (5) Key quantitative evaluation outcomes (Test Accuracy, Macro F1-Score, and AUC).*

**Abstract Draft Template:**
Diabetes mellitus remains one of the most pervasive chronic health conditions worldwide, causing substantial mortality, long-term cardiovascular complications, and economic burden. Timely identification of prediabetic and diabetic states enables preventive clinical interventions that drastically curb disease progression. In this academic investigation, we design, implement, and analyze an Artificial Neural Network (ANN)—specifically a Multilayer Perceptron (MLP)—to classify individuals into three health categories: Healthy (0), Prediabetes (1), and Diagnosed Diabetes (2). Utilizing the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset comprising 253,680 records and 21 clinical, demographic, and behavioral attributes, our pipeline standardizes continuous features, stratifies splits to maintain class proportions, and optimizes a 3-layer architecture (128 → 64 → 32 hidden units with ReLU activations) using Full-Batch Gradient Descent with Cross-Entropy Loss. [TODO: Insert final test performance metrics, e.g., achieving an overall accuracy of XX.X% and macro ROC-AUC of 0.XX]. This study demonstrates the efficacy of deep feedforward networks for public health screening and underscores the practical challenges of non-communicable disease risk stratification under extreme class imbalance.

---

## 3. Introduction

### 3.1 Healthcare Domain Background & Global Statistics
Diabetes mellitus is a metabolic disorder characterized by sustained hyperglycemia resulting from defects in insulin secretion, insulin action, or both. According to the International Diabetes Federation (IDF), more than 537 million adults globally were living with diabetes as of recent epidemiological estimates, with projections surpassing 783 million by 2045. Furthermore, an estimated 541 million adults exhibit impaired glucose tolerance (prediabetes), placing them at elevated risk of transitioning into full-blown Type 2 diabetes. 

Unmanaged chronic diabetes accelerates microvascular complications (diabetic retinopathy, nephropathy, and neuropathy) and macrovascular morbidities (coronary artery disease, myocardial infarction, and ischemic stroke). Because early-stage diabetes and prediabetes frequently progress asymptomatically, non-invasive risk screening mechanisms are paramount in contemporary public health strategies.

### 3.2 Problem Formulation & Motivation
Traditional clinical diagnostic criteria—such as Fasting Plasma Glucose (FPG), Oral Glucose Tolerance Tests (OGTT), and Glycated Hemoglobin (HbA1c)—require physical blood draws, laboratory infrastructure, and patient clinic visits. Consequently, millions of susceptible individuals remain undiagnosed until irreversible organ damage occurs. 

By contrast, behavioral, socioeconomic, and preliminary medical survey indicators (e.g., blood pressure status, BMI, age, physical activity, dietary habits, and smoking history) can be gathered remotely and rapidly at minimal cost. Machine learning, particularly Artificial Neural Networks (ANNs), offers the ability to uncover intricate, non-linear relationships among these diverse behavioral determinants, providing actionable risk probabilities to clinicians and patients.

### 3.3 Project Objectives
The overarching aim of this project is to construct an end-to-end multi-class classification pipeline using a Multilayer Perceptron (MLP) capable of categorizing patients into three health states. The specific objectives include:
1. **Data Preprocessing & Hygiene:** Standardize heterogeneous features (continuous, binary, and ordinal scales) and formulate stratified data splits to reflect true population distributions.
2. **Architecture Design:** Engineer a deep Multilayer Perceptron consisting of an input layer (21 neurons), three hidden layers with non-linear Rectified Linear Unit (ReLU) activations, and a 3-neuron Softmax output layer.
3. **Full-Batch Gradient Descent Optimization:** Implement full-batch gradient descent (`solver='sgd'`, `batch_size=N_train`) to explore true deterministic gradient convergence across the entire dataset.
4. **Evaluation & Diagnosis:** Quantitatively evaluate the network utilizing multi-class metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and One-vs-Rest ROC-AUC).
5. **Interactive Deployment:** Develop a lightweight, user-friendly Streamlit web interface enabling real-time clinical risk inference based on patient input parameters.

---

## 4. Literature Review
> **[TODO: Literature Review Instructions]**
> *Review 3 to 5 peer-reviewed publications or benchmark papers on diabetes risk prediction and neural networks. Summarize findings, algorithms compared (e.g., Logistic Regression, Random Forest, XGBoost, Deep Neural Networks), and limitations identified in prior works.*

### Literature Review Notes & Placeholders
- **Study 1:** [TODO: Cite author(s), year, methodology, and outcome regarding machine learning applied to the CDC BRFSS survey data.]
- **Study 2:** [TODO: Cite comparative analyses between traditional tree-based ensembles (Random Forest, XGBoost) and Deep Feedforward Networks on medical survey datasets.]
- **Study 3:** [TODO: Discuss prior techniques addressing severe class imbalance in multi-class disease diagnosis (e.g., SMOTE, class-weighted loss, threshold adjustment).]

---

## 5. Dataset Description

### 5.1 Dataset Origin & Summary
The dataset utilized in this project is the **CDC Diabetes Health Indicators** dataset, sourced from the **UCI Machine Learning Repository** and derived from the Centers for Disease Control and Prevention's (CDC) Behavioral Risk Factor Surveillance System (BRFSS).
- **Source Repository:** UCI Machine Learning Repository
- **Source URL:** [https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- **Total Records ($N$):** 253,680 individual survey responses
- **Total Features ($D$):** 21 predictive features
- **Target Variable:** `Diabetes_012`
  - `0`: Healthy / No Diabetes (~84.2% of dataset)
  - `1`: Prediabetes (~1.8% of dataset)
  - `2`: Diabetes (~13.9% of dataset)

### 5.2 Feature Inventory & Variable Definitions
The table below specifies all 21 input features utilized by the model:

| # | Feature Name | Data Type | Value Range / Units | Description |
|---|--------------|-----------|---------------------|-------------|
| 1 | `HighBP` | Binary | 0, 1 | 0 = No high blood pressure; 1 = High blood pressure diagnosed |
| 2 | `HighChol` | Binary | 0, 1 | 0 = No high blood cholesterol; 1 = High cholesterol diagnosed |
| 3 | `CholCheck` | Binary | 0, 1 | 0 = No cholesterol check in past 5 years; 1 = Checked |
| 4 | `BMI` | Continuous | 12 – 98 ($\text{kg/m}^2$) | Body Mass Index calculated from weight and height |
| 5 | `Smoker` | Binary | 0, 1 | Have smoked at least 100 cigarettes in lifetime (0 = No, 1 = Yes) |
| 6 | `Stroke` | Binary | 0, 1 | Ever told had a stroke (0 = No, 1 = Yes) |
| 7 | `HeartDiseaseorAttack` | Binary | 0, 1 | Coronary heart disease or myocardial infarction (0 = No, 1 = Yes) |
| 8 | `PhysActivity` | Binary | 0, 1 | Physical activity in past 30 days excluding regular job (0 = No, 1 = Yes) |
| 9 | `Fruits` | Binary | 0, 1 | Consume fruit 1 or more times per day (0 = No, 1 = Yes) |
| 10 | `Veggies` | Binary | 0, 1 | Consume vegetables 1 or more times per day (0 = No, 1 = Yes) |
| 11 | `HvyAlcoholConsump` | Binary | 0, 1 | Heavy alcohol drinkers (Men >14 drinks/week; Women >7 drinks/week) |
| 12 | `AnyHealthcare` | Binary | 0, 1 | Any healthcare coverage including insurance/HMO (0 = No, 1 = Yes) |
| 13 | `NoDocbcCost` | Binary | 0, 1 | Needed doctor in past 12 mos but could not see due to cost (0 = No, 1 = Yes) |
| 14 | `GenHlth` | Ordinal | 1 – 5 | Self-rated general health (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor) |
| 15 | `MentHlth` | Discrete | 0 – 30 (Days) | Number of days of poor mental health during past 30 days |
| 16 | `PhysHlth` | Discrete | 0 – 30 (Days) | Number of days of physical illness/injury during past 30 days |
| 17 | `DiffWalk` | Binary | 0, 1 | Serious difficulty walking or climbing stairs (0 = No, 1 = Yes) |
| 18 | `Sex` | Binary | 0, 1 | Biological sex (0 = Female, 1 = Male) |
| 19 | `Age` | Ordinal | 1 – 13 | 13-level age category (1 = 18–24, 2 = 25–29, ..., 13 = 80 or older) |
| 20 | `Education` | Ordinal | 1 – 6 | Highest education scale (1=None/KG, 2=Elem, 3=Some HS, 4=HS Grad, 5=Some College, 6=College Grad 4+ yrs) |
| 21 | `Income` | Ordinal | 1 – 8 | Annual household income scale (1= <$10k, 2= $10k–$15k, ..., 8= ≥$75k) |

---

## 6. Methodology

### 6.1 Data Preprocessing Pipeline
1. **Data Cleaning & Missing Value Verification:** The raw UCI BRFSS dataset contains zero null or missing entries across all 253,680 records.
2. **Stratified Train-Test Splitting:** To preserve the minority class proportions (prediabetes at 1.8% and diabetes at 13.9%), stratified sampling is executed with an 80/20 train-to-test ratio:
   - Training Set: $N_{\text{train}} = 202,944$ samples (80%)
   - Test Set: $N_{\text{test}} = 50,736$ samples (20%)
3. **Feature Scaling (Z-Score Standardization):** Neural networks require balanced input feature variance to facilitate gradient stability during backpropagation. The continuous, discrete, and ordinal attributes exhibit varied scales (e.g., `BMI` ranges from 12 to 98, whereas `GenHlth` ranges 1 to 5). We fit a `StandardScaler` strictly on $X_{\text{train}}$ and transform both $X_{\text{train}}$ and $X_{\text{test}}$:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

### 6.2 MLP Architecture Design
The network architecture is a fully connected Multilayer Perceptron (MLP) structured as follows:

```
[Input Layer: 21 Features]
          │
          ▼
[Dense Layer 1: 128 Neurons] ──► ReLU Activation
          │
          ▼
[Dense Layer 2: 64 Neurons]  ──► ReLU Activation
          │
          ▼
[Dense Layer 3: 32 Neurons]  ──► ReLU Activation
          │
          ▼
[Output Layer: 3 Neurons]    ──► Softmax Activation (P(Healthy), P(Prediabetes), P(Diabetes))
```

- **Input Dimension:** $D = 21$ features.
- **Hidden Layers:**
  - Hidden Layer 1: $128$ neurons ($21 \times 128 + 128 = 2,816$ parameters)
  - Hidden Layer 2: $64$ neurons ($128 \times 64 + 64 = 8,256$ parameters)
  - Hidden Layer 3: $32$ neurons ($64 \times 32 + 32 = 2,080$ parameters)
- **Output Layer:** $K = 3$ neurons ($32 \times 3 + 3 = 99$ parameters)
- **Total Trainable Parameters:** $2,816 + 8,256 + 2,080 + 99 = 13,251$ parameters.

### 6.3 Activation Functions
1. **Rectified Linear Unit (ReLU)** for hidden layers:
   $$f(z) = \max(0, z)$$
   *Rationale:* ReLU mitigates the vanishing gradient problem inherent in Sigmoid or Hyperbolic Tangent functions, accelerates gradient convergence, and induces sparse representations.
2. **Softmax** for multi-class output layer:
   $$P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad k \in \{0, 1, 2\}$$
   *Rationale:* Ensures the network output forms a valid probability distribution over the three health categories ($P(y=k) \in [0, 1]$ and $\sum_{k=0}^2 P(y=k) = 1$).

### 6.4 Optimization: Full-Batch Gradient Descent
In our configuration, the optimization solver is configured as Full-Batch Gradient Descent (`solver='sgd'`, `batch_size = N_train = 202,944`):
$$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \, \nabla_{\mathbf{W}} \mathcal{L}_{\text{total}}(\mathbf{W}^{(t)})$$
where:
$$\nabla_{\mathbf{W}} \mathcal{L}_{\text{total}}(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^N \nabla_{\mathbf{W}} \mathcal{L}_i(\mathbf{W})$$

#### Comparison of Gradient Descent Variants:
| Characteristic | Full-Batch Gradient Descent (Used Here) | Mini-Batch Gradient Descent | Stochastic Gradient Descent (SGD) |
|---|---|---|---|
| **Sample Size per Update** | Entire training set ($N$) | Small subset $B$ (e.g., 32, 64, 128) | Single sample ($B = 1$) |
| **Gradient Trajectory** | Smooth, deterministic, true gradient | Controlled stochastic oscillation | Highly noisy, erratic |
| **Memory Requirement** | High (loads full dataset into RAM) | Low to moderate | Minimal |
| **Computation Speed** | Fewer updates per epoch; slow per step | Fast GPU parallelization | High overhead per sample |
| **Escape from Local Minima**| May plateau in shallow basins | Stochastic noise assists escaping | High variance helps escape |

### 6.5 Loss Function & The Class Imbalance Problem
The baseline cost function in multi-class classification is categorical cross-entropy:
$$\mathcal{L}_{CE} = - \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} y_{i, k} \log(\hat{y}_{i, k})$$
where $y_{i, k} \in \{0, 1\}$ is the one-hot ground-truth indicator and $\hat{y}_{i, k} = \text{Softmax}(\mathbf{z}_i)_k$ is the predicted posterior probability.

#### Theoretical Origin of Minority-Class Collapse
In the CDC BRFSS dataset, Class 0 (Healthy) accounts for 84.24% of records ($N_0 = 170,962$ in train), Class 2 (Diabetes) represents 13.93% ($N_2 = 28,277$), and Class 1 (Prediabetes) represents only 1.83% ($N_1 = 3,705$).
The gradient of the cross-entropy loss with respect to the output layer logits $z_{ik}$ is:
$$\frac{\partial \mathcal{L}_{CE}}{\partial z_{ik}} = \frac{1}{N} (\hat{y}_{ik} - y_{ik})$$
Because Class 0 constitutes 84.24% of all summation terms in the full-batch gradient:
$$\nabla_{\mathbf{W}} \mathcal{L}_{CE} = \frac{1}{N} \left[ \sum_{i \in \mathcal{C}_0} \nabla \ell_i + \sum_{i \in \mathcal{C}_1} \nabla \ell_i + \sum_{i \in \mathcal{C}_2} \nabla \ell_i \right]$$
the gradient direction is overwhelmingly dictated by Class 0 errors. If the neural network predicts Class 0 for 100% of samples, the empirical loss on the 1.83% minority class contributes less than 0.02 to the total loss. Consequently, unweighted Full-Batch Gradient Descent quickly converges into a degenerate local basin where:
$$\arg\max_k \hat{y}_{ik} \ne 1 \quad \forall \, i$$
yielding **zero predicted prediabetes cases (Recall = 0.0000)** despite a superficially high nominal accuracy of ~84.60%.

### 6.6 Mathematically Justified Solution: Class-Weighted Cross-Entropy
To eliminate minority-class collapse without violating the assignment's strict requirement of Full-Batch Gradient Descent, we formulate **Class-Weighted Cross-Entropy**. Each observation $i$ belonging to true class $k = y_i$ is assigned a scalar penalty weight $W_k$:
$$\mathcal{L}_{\text{weighted}}(\mathbf{W}) = - \frac{1}{\sum_{i=1}^N W_{y_i}} \sum_{i=1}^N W_{y_i} \sum_{k=0}^{K-1} y_{i, k} \log(\hat{y}_{i, k})$$
Using the balanced weighting theorem (King & Zeng, 2001):
$$W_k = \frac{N}{K \cdot N_k}$$
where $N = 202,944$ is the training sample size, $K=3$ is the number of classes, and $N_k$ is the frequency of class $k$ in the training partition. Crucially, weights are calculated **strictly from training labels** to prevent data leakage.

For the training set:
- **Class 0 (Healthy):** $W_0 = \frac{202,944}{3 \times 170,962} \approx 0.3957$
- **Class 1 (Prediabetes):** $W_1 = \frac{202,944}{3 \times 3,705} \approx 18.2586$
- **Class 2 (Diabetes):** $W_2 = \frac{202,944}{3 \times 28,277} \approx 2.3917$

#### Mathematical Impact on Full-Batch Backpropagation
In backpropagation, the error delta backpropagated from the Softmax layer becomes:
$$\delta_{ik} = W_{y_i} (\hat{y}_{ik} - y_{ik})$$
Summing the weighted gradients across the entire training batch yields:
$$\sum_{i: y_i = k} W_{y_i} = N_k \cdot \left( \frac{N}{K \cdot N_k} \right) = \frac{N}{K}$$
As a result, each of the three clinical classes contributes exactly $\frac{1}{K} = \frac{1}{3} \approx 33.33\%$ to the total gradient update vector per epoch. This mathematically forces the full-batch SGD optimizer to position decision boundaries that actively identify prediabetic and diabetic individuals.

---

## 7. Implementation

### 7.1 Software Tools & Library Stack
- **Python 3.14:** Execution environment.
- **NumPy & Pandas:** Vectorized linear algebra and tabular dataset operations.
- **scikit-learn:** `MLPClassifier`, `StandardScaler`, `train_test_split`, `DummyClassifier`, and metric calculations.
- **Matplotlib & Seaborn:** Confusion matrices, loss trajectories, and ROC curve plotting.
- **Streamlit:** Interactive web application for clinical risk inference.
- **Pytest:** Automated test suite verifying data hygiene and full-batch constraint compliance.

### 7.2 Modular Project Structure
```text
Diabetes-Risk-Classification-ANN/
│
├── data/
│   └── diabetes_012_health_indicators_BRFSS2015.csv  # CDC BRFSS dataset (253,680 records)
│
├── docs/
│   ├── project_report.md         # Comprehensive academic project report
│   └── viva_questions.md         # 22 Viva exam questions and answers
│
├── models/
│   ├── mlp_model.pkl             # Serialized trained MLP weights (Joblib)
│   └── scaler.pkl                # Fitted StandardScaler object
│
├── reports/
│   └── figures/
│       ├── class_distribution.png # Target class frequency bar chart
│       ├── confusion_matrix.png   # Multi-class confusion matrix heatmap
│       ├── learning_curve.png     # Full-batch training loss trajectory
│       ├── roc_curves.png         # Multi-class One-vs-Rest ROC-AUC curves
│       └── feature_importance.png # Input layer weight magnitude ranking
│
├── src/
│   ├── __init__.py               # Python package initialization
│   ├── data_loader.py            # Dataset loading and schema validation
│   ├── preprocessing.py          # Scaling, stratified splitting, and distribution checks
│   ├── model.py                  # FullBatchMLPClassifier and architecture factory
│   ├── train_utils.py            # Full-batch GD training engine and loss logging
│   ├── evaluation.py             # Classification report, metric logs, and ROC-AUC
│   └── visualization.py          # Plotting utilities for figures and reports
│
├── tests/
│   ├── test_preprocessing.py     # Unit tests for scaling and stratified split
│   └── test_model.py             # Unit tests for FullBatchMLPClassifier and class weighting
│
├── app.py                        # Streamlit interactive web application
├── evaluate.py                   # Standalone evaluation and plotting script
├── train.py                      # Standalone full-batch model training script
├── requirements.txt              # Production dependency specifications
└── README.md                     # Project overview and reproduction guidelines
```

---

## 8. Results and Discussion

### 8.1 Rigorous Model Selection & Validation Comparison
To select the optimal configuration without data contamination, the full dataset (253,680 samples) was first split into an 80% development set (202,944 samples) and a 20% held-out test set (50,736 samples). The test set was **strictly preserved** and not used for model selection.
The development set was partitioned into Sub-Train (162,355 samples) and Validation (40,589 samples). Four configurations were evaluated on the validation partition:
1. **Dummy Baseline:** `DummyClassifier(strategy='most_frequent')`
2. **Config A (Unweighted Full-Batch GD):** Classical full-batch SGD with uniform sample weights.
3. **Config B (Balanced Class-Weighted Full-Batch GD):** Full-batch SGD with $W_k = \frac{N}{K \cdot N_k}$.
4. **Config C (Square-Root Class-Weighted Full-Batch GD):** Full-batch SGD with softened weights $W_k = \sqrt{\frac{N}{K \cdot N_k}}$.

#### Actual Empirical Validation Results Table:
| Metric | Dummy Baseline | Config A (Unweighted) | Config B (Balanced Weighted) | Config C (Sqrt Weighted) |
|---|---|---|---|---|
| **Overall Accuracy** | 84.24% | 84.24% | **63.66%** | 80.92% |
| **Macro Precision** | 28.08% | 28.08% | **42.58%** | 42.19% |
| **Macro Recall** | 33.33% | 33.33% | **49.18%** | 43.73% |
| **Macro F1-Score** | 30.48% | 30.48% | **41.02%** | 42.92% |
| **Class 0 (Healthy) Precision** | 84.24% | 84.24% | **93.70%** | 88.88% |
| **Class 0 (Healthy) Recall** | 100.00% | 100.00% | **64.84%** | 89.09% |
| **Class 0 (Healthy) F1** | 91.45% | 91.45% | **76.64%** | 88.98% |
| **Class 1 (Prediabetes) Precision** | 0.00% | 0.00% | **2.65%** | 0.00% |
| **Class 1 (Prediabetes) Recall** | 0.00% | 0.00% | **20.51% (152 cases)** | 0.00% |
| **Class 1 (Prediabetes) F1** | 0.00% | 0.00% | **4.70%** | 0.00% |
| **Class 2 (Diabetes) Precision** | 0.00% | 0.00% | **31.40%** | 37.70% |
| **Class 2 (Diabetes) Recall** | 0.00% | 0.00% | **62.18% (3,516 cases)** | 42.10% |
| **Class 2 (Diabetes) F1** | 0.00% | 0.00% | **41.72%** | 39.78% |
| **Class 0 ROC-AUC** | 0.5000 | 0.7766 | **0.7810** | 0.7891 |
| **Class 1 ROC-AUC** | 0.5000 | 0.6096 | **0.6103** | 0.6235 |
| **Class 2 ROC-AUC** | 0.5000 | 0.7813 | **0.7847** | 0.7903 |
| **Macro-Average ROC-AUC** | 0.5000 | 0.7225 | **0.7253** | 0.7343 |

#### Validation Confusion Matrices:
- **Dummy Baseline:**
  ```text
  [34193,     0,     0]
  [  741,     0,     0]
  [ 5655,     0,     0]
  ```
- **Config A (Unweighted Full-Batch GD):**
  ```text
  [34193,     0,     0]
  [  741,     0,     0]
  [ 5655,     0,     0]
  ```
- **Config B (Balanced Class-Weighted Full-Batch GD):**
  ```text
  [22171,  4689,  7333]
  [  239,   152,   350]
  [ 1251,   888,  3516]
  ```

### 8.2 Clinical Discussion & Sensitivity-Specificity Trade-Off
1. **The Deception of Raw Accuracy:**
   The Dummy Classifier and Config A both report an accuracy of 84.24%. However, both models predict zero prediabetes cases and zero diabetes cases. In a healthcare diagnostic setting, this represents catastrophic failure: 100% of diabetic and prediabetic patients would be sent home without intervention.
2. **Clinical Utility of Config B:**
   With balanced class weighting ($W_1 \approx 18.26$), the neural network correctly detects **20.51% of prediabetic individuals** (152 patients) and **62.18% of diabetic individuals** (3,516 patients). Macro recall jumps from 33.33% to **49.18%**, and Macro F1 rises from 30.48% to **41.02%**.
3. **The Healthcare Screening Trade-off:**
   The nominal accuracy drops from 84.24% to 63.66% because healthy patients are flagged with higher risk probabilities (elevated false positives). In population screening, high sensitivity (recall) is universally prioritized over specificity: a false alarm triggers an inexpensive confirmatory blood test (HbA1c / FPG), whereas a false negative allows progressive, irreversible vascular damage to remain undetected.

## 9. Challenges and Limitations

1. **Extreme Class Imbalance:**
   Prediabetes accounts for only ~1.8% of the total dataset, while healthy individuals comprise >84%. Standard cross-entropy minimization incentivizes the network to prioritize the majority class, occasionally leading to poor recall for prediabetic individuals.
2. **Computational Overhead of Full-Batch Updates:**
   Calculating true gradients over 202,944 samples per step demands substantial RAM and makes parameter updates infrequent relative to wall-clock time compared to mini-batch SGD or Adam.
3. **Survey Subjectivity & Self-Reporting:**
   The BRFSS data relies on self-reported phone interviews. Variables such as `GenHlth`, `MentHlth`, and dietary intake are inherently subjective and prone to recall bias.
4. **Coarse Clinical Quantization:**
   Critical biomarkers like fasting blood glucose and continuous HbA1c percentages are absent; proxy categorical variables (`HighBP`, `HighChol`) restrict fine-grained diagnostic sensitivity.

---

## 10. Conclusion and Future Work
> **[TODO: Conclusion Instructions]**
> *Summarize the project accomplishments and provide 3–4 concrete avenues for future enhancements.*

**Conclusion Draft:**
This project successfully designed and validated an Artificial Neural Network architecture for multi-class diabetes risk stratification based on 21 public health survey indicators. The feedforward Multilayer Perceptron demonstrates that non-invasive demographic and lifestyle indicators carry meaningful predictive signal for diabetes detection.

**Future Directions:**
1. **Class Balancing Techniques:** Incorporate Synthetic Minority Over-sampling Technique (SMOTE) or focal cross-entropy loss to improve prediabetes detection sensitivity.
2. **Adaptive Optimizers:** Compare full-batch gradient descent against modern adaptive algorithms (Adam, AdamW, RMSProp) with Cosine Annealing learning rate schedules.
3. **Model Explainability:** Integrate SHAP (SHapley Additive exPlanations) or LIME to elucidate which specific lifestyle factors contribute most significantly to a patient's predicted risk score.
4. **Clinical Validation:** Validate the neural network on multi-center hospital electronic health records (EHR) containing continuous biochemical laboratory markers.

---

## 11. References
1. Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System (BRFSS) Survey Data and Documentation*. U.S. Department of Health and Human Services.
2. UCI Machine Learning Repository. *CDC Diabetes Health Indicators Dataset*. [https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. International Diabetes Federation. *IDF Diabetes Atlas*, 10th edn. Brussels, Belgium: 2021.
