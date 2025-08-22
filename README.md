# 💳 Credit Card Fraud Detection using Machine Learning & Deep Learning

This project implements an intelligent fraud detection system for banking transactions using state-of-the-art machine learning and deep learning techniques. It focuses on handling real-world challenges like class imbalance and hyperparameter optimization to achieve high accuracy and reliability in identifying fraudulent transactions.

---

## 🧠 Overview

Credit card fraud detection is a critical task in the financial industry. This project demonstrates an end-to-end pipeline that includes:

- Data preprocessing & feature selection
- Class imbalance handling using class weight tuning
- Bayesian optimization for hyperparameter tuning
- Training and evaluation of multiple ML models
- Ensemble learning using majority voting
- A deep learning model for enhanced pattern recognition

---

## 📦 Tech Stack

- **Python** (NumPy, Pandas, Scikit-learn)
- **LightGBM**, **XGBoost**, **CatBoost**
- **Keras (TensorFlow backend)** for Deep Learning
- **Bayesian Optimization** (via `bayesian-optimization` library)
- **Matplotlib**, **Seaborn** for visualization

---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Transactions**: 284,807  
- **Frauds**: 492 (~0.17%)  
- **Features**: 30 total  
  - Anonymized PCA features (`V1` to `V28`)
  - `Time`, `Amount`, and `Class` (target variable: 1 for fraud, 0 for legitimate)

---

## 📈 Model Pipeline

1. **Exploratory Data Analysis**  
   Visualize distribution of features, class balance, correlations, etc.

2. **Feature Selection**  
   Information Gain used to select top features.

3. **Model Training**  
   - Logistic Regression
   - LightGBM
   - XGBoost
   - CatBoost
   - Deep Neural Network (ANN)

4. **Hyperparameter Tuning**  
   Bayesian Optimization to fine-tune model parameters.

5. **Ensemble Learning**  
   Majority Voting (hard and soft) across top models.

6. **Evaluation Metrics**
   - ROC-AUC
   - Precision / Recall
   - F1-Score
   - MCC (Matthews Correlation Coefficient)

---

## 📌 Results (Summary)

| Model                | ROC-AUC | Precision | Recall | F1-Score | MCC  |
|---------------------|---------|-----------|--------|----------|------|
| Logistic Regression | 0.95    | 0.74      | 0.76   | 0.75     | 0.75 |
| LightGBM + XGBoost  | 0.95    | 0.79      | 0.80   | 0.79     | 0.79 |
| Deep Learning (ANN) | 0.94    | 0.80      | 0.82   | 0.81     | 0.81 |

---
