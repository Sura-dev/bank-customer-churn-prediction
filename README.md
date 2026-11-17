# bank-customer-churn-prediction
A machine learning project that predicts bank customer churn using Python and Logistic Regression. Includes data preprocessing, categorical encoding, feature scaling, model training, evaluation, and churn probability prediction.



BANK CUSTOMER CHURN PREDICTION

Project Overview

This project explores bank customer churn prediction using a Logistic
Regression classifier. It includes a model-training workflow and a
separate prediction workflow for applying saved model artifacts to
customer CSV files.

The model uses customer demographic, account, and activity features to
predict whether a customer is likely to leave the bank. The prediction
workflow also returns the estimated probability of churn.

Problem Statement

Customer churn occurs when a customer stops using a bank's services.
Identifying customers who may leave can help a bank understand customer
retention and plan appropriate customer-support strategies.

Dataset
-------
The training notebook reads:
    bank_customer_churn_synthetic_10000.csv

The prediction notebook reads:
    bank_customer_churn_synthetic_10000.csv
    new_bank_customers_100.csv

The features include age, gender, credit score, balance, tenure, number
of bank products, credit-card ownership, active-member status, estimated
salary, and country. The target column is `exited`, indicating whether
a customer left. The target column is not required for new-customer
prediction.

The CSV datasets are not included automatically. Add them to the
repository if you have permission to share them, and document their
source.

Project Workflow

1. Model training
   - Load the customer churn CSV dataset with Pandas.
   - Remove columns not used as features, including customer_id.
   - Convert gender and country into dummy variables.
   - Select the features and target column (`exited`).
   - Split the data into training and test sets.
   - Scale selected numerical features with the custom CustomerScaler.
   - Train a Logistic Regression classifier and evaluate accuracy.
   - Save the trained model and fitted scaler as pickle files.

2. Prediction
   - Import ChurnPredictor from churn_model.py.
   - Load the saved model and scaler.
   - Read a customer CSV file and remove columns not used for prediction.
   - Convert gender and country values into dummy variables.
   - Align input columns with the model's expected feature columns.
   - Apply the fitted scaler.
   - Generate churn probabilities and predicted categories.
   - Return the original customer data with Probability and Prediction
     columns.

Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- scikit-learn
- Logistic Regression
- StandardScaler
- Python pickle

Repository Contents

- Untitled.ipynb: Original notebook for data preparation, feature
  engineering, model training, evaluation, and saving model artifacts.
- churn_model.py: Python module containing CustomerScaler and
  ChurnPredictor, including customer-data preparation, prediction, and
  model/scaler training and saving methods.
- Untitled2.ipynb: Notebook demonstrating how to import and use
  ChurnPredictor with saved artifacts to predict churn for customer CSVs.
- model.pkl: Saved Logistic Regression classifier expected by the
  prediction notebook.
- scaler.pkl: Saved fitted CustomerScaler expected by the prediction
  notebook.
- README.txt: Project documentation.

How to Run

1. Clone or download the repository.
2. Install Python and the required packages:

       pip install numpy pandas scikit-learn jupyter

3. Ensure these files are in the project directory:
   - churn_model.py
   - model.pkl
   - scaler.pkl
   - bank_customer_churn_synthetic_10000.csv
   - new_bank_customers_100.csv (for the new-customer example)

4. If model.pkl and scaler.pkl do not exist, run the training notebook
   (Untitled(4).ipynb) to create them.
5. Start Jupyter:

       jupyter notebook

6. Open Untitled2.ipynb and run its cells to load the saved artifacts
   and generate predictions.

Prediction Example

    from churn_model import ChurnPredictor

    model = ChurnPredictor("model.pkl", "scaler.pkl")
    model.load_and_clean_data("new_bank_customers_100.csv")
    results = model.predicted_outputs()

The returned DataFrame contains the original customer columns plus:
- Probability: predicted probability that `exited = 1`.
- Prediction: predicted class from the Logistic Regression classifier.

Saved Model Artifacts

- model.pkl: trained Logistic Regression classifier.
- scaler.pkl: fitted CustomerScaler preprocessing object.

Only load pickle files from sources you trust. Pickle files can execute
code during loading.

Important Notes and Future Improvements

- The filenames indicate that the datasets are synthetic. Performance
  on synthetic data does not establish performance on real customers.
- The prediction workflow depends on the feature names and preprocessing
  used during training. Keep both workflows consistent.
- Confirm that model.pkl and scaler.pkl were created using the feature
  naming convention expected by churn_model.py.
- Review and test CustomerScaler.transform to ensure every configured
  numerical column is replaced with its scaled value.
- Consider a reproducible train/test split and stratification where
  appropriate.
- Evaluate precision, recall, F1-score, ROC-AUC, and a confusion matrix
  in addition to accuracy.
- For deployment, consider packaging preprocessing and the classifier
  into one reproducible scikit-learn Pipeline.

Disclaimer

This is an educational machine-learning project, not a production
banking risk system. Do not use it for real customer decisions without
appropriate validation, privacy safeguards, fairness assessment, and
domain review.
