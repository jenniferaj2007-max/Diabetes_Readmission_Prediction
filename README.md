# DIABETES READMISSION PREDICTION

## 1. Project Overview

This project uses Machine Learning techniques to predict hospital readmission outcomes among patients with diabetes.

The project uses historical patient data to classify whether a patient was not readmitted, readmitted within 30 days, or readmitted after 30 days.

## 2. Dataset Description

Dataset Name: Diabetes 130-US Hospitals Dataset

Dataset Period: 1999–2008

Number of Records: 101,766 patient encounters

Number of Original Columns: 50

The dataset contains patient information, hospital admission details, medical procedures, laboratory results, medications, and readmission status.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Visual Studio Code

## 4. Machine Learning Algorithms

Three machine learning algorithms are implemented:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Random Forest Classifier

## 5. Project Workflow

1. Data Loading
2. Data Understanding
3. Data Cleaning
4. Exploratory Data Analysis
5. Feature Preparation
6. Train-Test Splitting
7. Data Encoding and Scaling
8. Model Training
9. Model Comparison
10. Confusion Matrix Visualization
11. Detailed Model Evaluation
12. Feature Importance Analysis
13. Final Prediction

## 6. Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

## 7. Prediction Categories

The target variable is `readmitted`.

- NO: No readmission recorded.
- <30: Readmitted within 30 days.
- >30: Readmitted after 30 days.

## 8. Expected Outcome

The project compares the performance of three machine learning algorithms and identifies their effectiveness in classifying hospital readmission outcomes.

## 9. Limitations

This project is an educational machine learning prototype based on historical hospital records. Its predictions are not intended for clinical decision-making.

## 10. Conclusion

The project demonstrates the application of machine learning techniques to healthcare data through data preprocessing, model development, evaluation, comparison, and prediction.