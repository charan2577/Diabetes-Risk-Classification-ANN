# CSEN3011: Artificial Neural Networks — Viva Voce Questions & Answers
## Project: Diabetes Risk Classification Using Multilayer Perceptron (MLP)

This document contains oral exam (viva voce) questions and concise, beginner-friendly answers tailored to this project. Each response is formulated in 2–5 sentences with clear theoretical grounding and practical references to the CDC Diabetes Health Indicators implementation.

---

### Q1. What is an Artificial Neural Network (ANN)?
**Answer:**  
An Artificial Neural Network (ANN) is a computational model inspired by the biological neural networks of the human brain. It consists of interconnected processing units called artificial neurons or nodes arranged in layers, where each connection carries an adaptable weight. ANNs learn by passing inputs forward to generate predictions and iteratively updating their connection weights through optimization algorithms to minimize prediction errors on complex, non-linear problems.

---

### Q2. What is a Multilayer Perceptron (MLP)?
**Answer:**  
A Multilayer Perceptron (MLP) is a foundational class of feedforward artificial neural networks consisting of at least three layers: an input layer, one or more hidden layers, and an output layer. In an MLP, information flows strictly in one direction from input to output without cycles or feedback loops. Because hidden neurons apply non-linear activation functions, MLPs can approximate any continuous non-linear function, distinguishing them from simple single-layer perceptrons.

---

### Q3. Explain the layers and dimensions in your specific MLP architecture.
**Answer:**  
Our network architecture comprises an input layer with 21 neurons corresponding to the 21 health and demographic survey features, followed by three fully connected hidden layers with 128, 64, and 32 neurons respectively, and a final output layer of 3 neurons. Each hidden layer applies the ReLU non-linear activation function, and the output layer utilizes Softmax to produce probabilities for the three target classes (Healthy, Prediabetes, Diabetes). In total, this architecture contains 13,251 trainable weights and biases across all layer transitions.

---

### Q4. What is an activation function, and why did you choose ReLU for the hidden layers?
**Answer:**  
An activation function introduces non-linearity into a neuron's output, enabling neural networks to learn complex mathematical mappings beyond linear combinations of inputs. We chose the Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$) because it is computationally efficient, induces sparse representations, and avoids the vanishing gradient problem common to Sigmoid and Tanh activations. This allows gradients to flow smoothly through our 3 hidden layers during backpropagation without degrading.

---

### Q5. What is the Softmax function and where is it used in this project?
**Answer:**  
The Softmax function transforms an arbitrary vector of real-valued raw logits into a valid probability distribution where each value is bounded between 0 and 1, and all values sum to exactly 1. In our project, Softmax is placed at the 3-neuron output layer to convert the final layer's net inputs into posterior class probabilities for Healthy ($P_0$), Prediabetes ($P_1$), and Diabetes ($P_2$). The class corresponding to the highest resulting probability is assigned as the model's final prediction.

---

### Q6. What is backpropagation?
**Answer:**  
Backpropagation (backward propagation of errors) is the fundamental algorithm used to train neural networks by calculating the gradient of the loss function with respect to every trainable weight and bias. It systematically applies the mathematical chain rule of calculus from the output layer backward toward the input layer. Once these partial derivatives are computed, optimization algorithms like gradient descent use them to update parameters in the direction that decreases the loss.

---

### Q7. What is gradient descent? Explain the difference between Batch, Mini-Batch, and Stochastic Gradient Descent (SGD).
**Answer:**  
Gradient descent is an iterative first-order optimization algorithm that adjusts network weights in the opposite direction of the loss gradient to reach a minimum. **Batch (Full-Batch) Gradient Descent** computes the average gradient across the entire training dataset before taking a single parameter step; **Stochastic Gradient Descent (SGD)** updates weights after each individual training instance; and **Mini-Batch Gradient Descent** balances both by computing updates over small subsets (e.g., 32, 64, or 128 samples). While Mini-Batch is common in deep learning, Full-Batch provides deterministic, non-noisy gradient estimates.

---

### Q8. Why did you use Full-Batch Gradient Descent in this project?
**Answer:**  
In our implementation, we configured `solver='sgd'` with `batch_size` set to the entire training set size ($N_{\text{train}} = 202,944$) to fulfill the academic objective of studying true, deterministic gradient descent. Calculating the gradient over all 202,944 records simultaneously eliminates stochastic noise, yielding a smooth and monotonically decreasing learning curve. This provides students and researchers with a clear baseline to observe pure gradient dynamics without mini-batch variance.

---

### Q9. What is a loss (cost) function, and why is Cross-Entropy used here?
**Answer:**  
A loss function mathematically measures the discrepancy between the network's predicted probabilities and the true categorical labels for given training samples. We utilized Multi-Class Cross-Entropy Loss (Log-Loss) because it is the standard maximum-likelihood objective for probabilistic classification. It heavily penalizes confident but incorrect predictions (as $-\log(P) \to \infty$ when $P \to 0$), driving the network to calibrate accurate class probabilities.

---

### Q10. What is learning rate, and what happens if it is set too high or too low?
**Answer:**  
The learning rate ($\eta$) is a fundamental hyperparameter that controls the step size taken along the negative gradient vector during each parameter update. If the learning rate is set too high, the optimization may oscillate wildly, overshoot the optimal minimum, or diverge completely. Conversely, if it is set too low, the model's convergence becomes excessively slow, requiring impractical numbers of training epochs and risking getting trapped in shallow local plateaus.

---

### Q11. What is overfitting, and what mechanisms prevent it in MLPs?
**Answer:**  
Overfitting occurs when a neural network memorizes noise, outliers, and quirks of the training data rather than generalizing underlying patterns, resulting in superior training accuracy but poor test performance. In our project, overfitting is prevented through L2 weight regularization ($\alpha$ penalty), early stopping based on validation score plateaus, keeping network capacity balanced with the dataset volume, and holding out an independent test set for unbiased evaluation.

---

### Q12. Explain the difference between Accuracy, Precision, Recall, and F1-score.
**Answer:**  
**Accuracy** measures the overall proportion of correct predictions across all samples. **Precision** measures the ratio of true positive identifications among all instances predicted as that class (minimizing false alarms). **Recall (Sensitivity)** measures the fraction of actual positive instances correctly captured by the model (minimizing missed cases). The **F1-score** is the harmonic mean of Precision and Recall, providing a balanced metric especially vital when assessing imbalanced classes.

---

### Q13. What is a Confusion Matrix, and why is it essential for multi-class medical diagnosis?
**Answer:**  
A Confusion Matrix is a tabular visualization where rows represent true ground-truth classes and columns represent the model's predicted classes. In our $3 \times 3$ matrix, the diagonal cells show correct predictions, while off-diagonal cells detail exact diagnostic errors (e.g., misclassifying a diabetic patient as healthy). For healthcare screening, it is vital because misclassifying diseased patients (false negatives) is far more dangerous than over-predicting healthy cases (false positives).

---

### Q14. What is ROC-AUC, and how is it calculated for multi-class classification?
**Answer:**  
The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate across all classification probability thresholds, and the Area Under the Curve (AUC) summarizes this discriminative ability into a single scalar between 0.5 (random guessing) and 1.0 (perfect ranking). In our 3-class problem, we compute ROC-AUC using a One-vs-Rest (OvR) approach, where each class is individually compared against all other classes combined, followed by macro-averaging to assess overall performance.

---

### Q15. Why did you scale/standardize the features before training the MLP?
**Answer:**  
We applied Z-score standardization (`StandardScaler`) so that all 21 input features have a mean of 0 and a variance of 1. Features in our survey dataset possess disparate ranges—such as `BMI` spanning 12 to 98 versus binary indicators spanning 0 to 1—which would cause features with larger numeric magnitudes to dominate gradient updates and distort the loss surface. Standardization ensures balanced gradient descent convergence and numerical stability across all neuron weights.

---

### Q16. What is stratified splitting and why was it necessary for this dataset?
**Answer:**  
Stratified splitting divides a dataset into training and testing partitions while ensuring that the percentage of each target class is strictly preserved across both subsets. In the CDC dataset, prediabetes accounts for only ~1.8% of records and diabetes for ~13.9%. A standard random split might under-represent or omit the rare prediabetes class in either partition, whereas stratified splitting guarantees that both train and test sets mirror the true epidemiological proportions.

---

### Q17. What are the 3 target classes and 21 features in your dataset?
**Answer:**  
The target variable is `Diabetes_012`, categorized into Class 0 (Healthy / No Diabetes), Class 1 (Prediabetes), and Class 2 (Diagnosed Diabetes). The 21 features include physiological markers (`HighBP`, `HighChol`, `CholCheck`, `BMI`), medical history (`Stroke`, `HeartDiseaseorAttack`), lifestyle behaviors (`Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`), healthcare access (`AnyHealthcare`, `NoDocbcCost`), physical/mental health metrics (`GenHlth`, `MentHlth`, `PhysHlth`, `DiffWalk`), and demographic variables (`Sex`, `Age`, `Education`, `Income`).

---

### Q18. How many total records are in the CDC dataset, and how are they split?
**Answer:**  
The dataset contains a total of 253,680 individual survey records without missing values. Using an 80/20 stratified split, 202,944 records are allocated to the training set for gradient descent optimization, and 50,736 records are reserved for the holdout test set to perform unbiased model evaluation and metric reporting.

---

### Q19. Why is class imbalance a major challenge in this dataset, and how can it be handled?
**Answer:**  
With Class 0 representing over 84% of records and Class 1 representing under 2%, a naive neural network could achieve ~84% raw accuracy simply by predicting Class 0 for every patient while completely failing to detect prediabetes. This challenge can be mitigated by adjusting class weights in the loss function to penalize minority misclassifications more heavily, applying oversampling methods like SMOTE, tuning decision thresholds away from the default 0.33, or evaluating models using Macro F1 and ROC-AUC rather than pure accuracy.

---

### Q20. How would you improve your model's performance in future iterations?
**Answer:**  
To enhance performance, we could: (1) implement Focal Loss or class-weighted cross-entropy to boost sensitivity on the prediabetes class, (2) transition to adaptive optimizers such as Adam or AdamW to speed up convergence, (3) conduct systematic hyperparameter tuning using Bayesian optimization for layer depth and dropout rates, and (4) enrich the dataset with objective clinical biomarkers like continuous HbA1c and fasting blood sugar levels.

---

### Bonus Q21. How is this project deployed for real-world end users?
**Answer:**  
The trained MLP model and fitted `StandardScaler` are serialized onto disk as Joblib artifacts (`models/mlp_model.pkl` and `models/scaler.pkl`). We built an interactive Streamlit web dashboard (`app.py`) where patients or healthcare workers can input the 21 survey parameters through sliders and dropdowns. The app standardizes the inputs on the fly, feeds them into the loaded MLP, and displays real-time predicted probabilities and clinical risk interpretations.

---

### Bonus Q22. How did you verify the software reliability of your code?
**Answer:**  
We established automated unit testing with Pytest across two dedicated suites: `tests/test_preprocessing.py` and `tests/test_model.py`. These tests validate that feature scaling normalizes means to 0 and variances to 1, dataset splitting preserves specified sizes and stratified ratios, the MLPClassifier initializes with the SGD solver and specified layer architectures, and model serialization correctly preserves predictive outputs.

---

### Bonus Q23. What is Class-Weighted Cross-Entropy, and how does it mathematically alter Full-Batch Gradient Descent?
**Answer:**  
Class-Weighted Cross-Entropy assigns a class-specific penalty weight $W_k = \frac{N}{K \cdot N_k}$ to each training sample based on its ground truth class. In backpropagation, the error delta backpropagated from the Softmax layer is scaled as $\delta_{ik} = W_{y_i}(\hat{y}_{ik} - y_{ik})$. In Full-Batch Gradient Descent, this ensures that the sum of gradient weights across all samples in any class equals $\frac{N}{K}$, forcing each class to contribute exactly one-third (33.3%) of the total gradient update energy regardless of class size differences.

---

### Bonus Q24. Why did your Unweighted Model and the Dummy Classifier achieve the exact same 84.24% accuracy?
**Answer:**  
Because Class 0 (Healthy) accounts for 84.24% of the population, predicting Class 0 for every single patient guarantees an accuracy of 84.24% with minimal average cross-entropy loss. In unweighted training, the gradient is overwhelmed by majority-class healthy samples, trapping the network in a local minimum where the minority prediabetes class ($1.83\%$) is never predicted. This demonstrates why raw accuracy is deceptive in medical diagnostics—a model with 84.24% accuracy can have 0% sensitivity on diseased patients.

---

### Bonus Q25. How did you ensure that model selection did not cause data leakage into the test set?
**Answer:**  
We enforced a strict 3-way partition: the 253,680 records were first split into an 80% training set (202,944 samples) and a 20% test set (50,736 samples). The test set was locked away and never touched. We then subdivided the 202,944 training records into an 80% sub-train set (162,355) and a 20% validation set (40,589), fitted our StandardScaler and class weights strictly on sub-train data, and compared configurations exclusively on the validation set.

---

### Bonus Q26. Explain the clinical trade-off between the Unweighted Model and the Balanced Model.
**Answer:**  
The Unweighted Model achieved 84.24% accuracy but had 0.0% recall for prediabetes and 0.0% recall for diabetes at 30 epochs (and only 10.7% recall for diabetes at 200 epochs). The Balanced Model achieved 63.66% accuracy but detected 20.51% of prediabetic patients and 62.18% of diabetic patients, elevating Macro Recall from 33.33% to 49.18%. In population health screening, missing diseased patients (false negatives) leads to irreversible organ damage, so high sensitivity is clinically preferred over nominal overall accuracy.

