💳 Credit Card Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning techniques.
By analyzing transaction data, the model aims to identify suspicious activities and help financial institutions minimize losses from fraudulent behavior.

🚀 Features

🧹 Data Preprocessing

Dropped irrelevant features

Encoded categorical variables using One-Hot Encoding with ColumnTransformer

🌲 Model Training

Implemented Logistic Regression for binary classification

📊 Model Evaluation

Measured performance using Accuracy Score, Classification Report, and Confusion Matrix

🔍 Visualization

Fraud vs Non-Fraud class distribution

Model performance visualized using Seaborn heatmaps

🛠 Tech Stack
Python

Pandas

NumPy

scikit-learn

Matplotlib

Seaborn

📂 Dataset

The dataset contains anonymized credit card transactions, with features representing transaction details and a target variable indicating whether the transaction is fraudulent.

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

📊 Model Workflow

Load and explore dataset

Preprocess data (drop irrelevant features, encode categorical variables)

Split dataset into training and testing sets

Train Logistic Regression model

Evaluate model using accuracy, confusion matrix, and classification report

Visualize fraud detection performance

📌 Output Example

Model Accuracy: 0.9472

Confusion Matrix:

[[283  15]

 [ 12 390]]
 
📈 Results

Achieved 94%+ accuracy in detecting fraudulent transactions

Balanced detection of fraud cases with low false positives

