# bank-customer-churn-prediction
A machine learning project that predicts bank customer churn using Python and Logistic Regression. Includes data preprocessing, categorical encoding, feature scaling, model training, evaluation, and churn probability prediction.

BANK CUSTOMER CHURN PREDICTION


Project Overview

This project explores bank customer churn prediction using a Logistic
Regression model. It uses customer demographic, account, and activity
features to predict whether a customer is likely to leave the bank.

The work is implemented in a Jupyter Notebook using Python, Pandas,
NumPy, scikit-learn, and statsmodels.

Problem Statement

Customer churn occurs when a customer stops using a bank's services.
Identifying customers who may leave can help a bank better understand
customer retention and plan appropriate customer-support strategies.

Dataset

The notebook reads a CSV file named:

    bank_customer_churn_synthetic_10000.csv

The dataset is described by its filename as a synthetic dataset of
10,000 bank customers. The notebook uses features including:

- Age
- Gender
- Credit score
- Account balance
- Tenure
- Number of bank products
- Credit-card ownership
- Active-member status
- Estimated salary
- Country

The target column is `exited`, indicating whether a customer left.

Note: The CSV dataset is not included automatically with this README.
Add the dataset to the repository if you have permission to share it.
If it is generated or obtained from another source, document that source.

Workflow

1. Load the customer churn CSV dataset with Pandas.
2. Remove the customer_id column.
3. Convert categorical variables (gender and country) into dummy
   variables using Pandas one-hot encoding.
4. Select and organize the model features and target column.
5. Split the data into training and test sets (80% training and
   20% testing in the notebook).
6. Standardize selected numerical features using StandardScaler through
   a custom CustomerScaler transformer.
7. Train a Logistic Regression classifier.
8. Evaluate the model using the training and test accuracy scores.
9. Generate class predictions and churn probabilities.
10. Save the trained Logistic Regression model and fitted scaler using
    Python pickle.

Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- statsmodels
- Logistic Regression
- StandardScaler
- Pickle

Repository Contents

- Untitled.ipynb: Jupyter Notebook containing data preparation,
  feature engineering, model training, evaluation, and model saving.
- README.txt: Project documentation.

How to Run

1. Clone or download this repository.
2. Ensure Python is installed.
3. Install the required packages:

       pip install numpy pandas scikit-learn statsmodels jupyter

4. Place `bank_customer_churn_synthetic_10000.csv` in the same directory
   as the notebook, or update the CSV path in the notebook.
5. Open Jupyter Notebook:

       jupyter notebook

6. Open the notebook and run its cells from top to bottom.

Saved Model Artifacts

The notebook saves two pickle files:

- model: the trained Logistic Regression estimator
- scaler: the fitted CustomerScaler preprocessing object

Only load pickle files from sources you trust. Pickle files can execute
code during loading.

Important Notes and Future Improvements

- The dataset filename indicates that the data is synthetic. Model
  performance on synthetic data does not establish performance on real
  banking customers.
- The notebook currently uses Logistic Regression as a baseline model.
- Set and document a random_state in train_test_split to make results
  reproducible.
- Evaluate more than accuracy. Useful classification metrics include
  precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Check class balance and consider the business impact of false
  positives and false negatives.
- Fit preprocessing only on the training data, then apply the fitted
  transformer to test or new data.
- For deployment, package the feature preparation and model together
  in a single reproducible preprocessing/model pipeline.

Disclaimer

This is an educational machine-learning project. It is not a production
banking risk system and should not be used to make real customer
decisions without appropriate validation, privacy safeguards, fairness
assessment, and domain review.
