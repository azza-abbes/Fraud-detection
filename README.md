# Credit Card Fraud detection
Project Overview
This project aims to identify fraudulent transactions in credit card data using various machine learning algorithms. The dataset undergoes preprocessing and analysis to train and evaluate four models, with the Random Forest model demonstrating the highest performance.

## Dataset
The dataset utilized for this project is the Fraud Detection Dataset, available on Kaggle:
[Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data?select=fraudTrain.csv)

It consists of two files: fraudTrain.csv and fraudTest.csv, which serve as the training and testing datasets, respectively.
The dataset is highly imbalanced, with fraudulent transactions representing a small fraction of the total data. To enhance the performance of the machine learning models, data balancing techniques were implemented:
Undersampling: Reduced the size of the majority class to match the minority class.
## Machine Learning Models
The following models were trained and assessed:
  * Logistic Regression: A linear approach for binary classification tasks.
  * K-Nearest Neighbors (KNN): A classification algorithm based on proximity in feature space.
  * Random Forest: An ensemble technique that aggregates decision trees for robust predictions.
  * Support Vector Machine (SVM): A classification model leveraging kernel functions.
## Evaluation Metrics
The models were evaluated using the following metrics:
  * Precision
  * Recall
  * F1 Score
