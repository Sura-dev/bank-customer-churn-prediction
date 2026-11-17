"""Bank customer churn: scaler and prediction model in one module."""
import pickle
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class CustomerScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns, copy=True, with_mean=True, with_std=True):
        self.columns = columns
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler(
            copy=self.copy, with_mean=self.with_mean, with_std=self.with_std
        )
        self.scaler_.fit(X[list(self.columns)])
        self.mean_ = pd.Series(self.scaler_.mean_, index=self.columns)
        self.var_ = pd.Series(self.scaler_.var_, index=self.columns)
        return self

    def transform(self, X):
        result = X.copy()
        # Scale selected numerical columns
        scaled_values = self.scaler_.transform(X[list(self.columns)])
        # Replace each column with float values
        for i, column in enumerate(self.columns):
            result[column] = scaled_values[:, i]
            return result


class ChurnPredictor:
    def __init__(self, model_file, scaler_file):
        # Load the trained model
        with open(model_file, "rb") as file:
            self.reg = pickle.load(file)

        # Load the fitted scaler
        with open(scaler_file, "rb") as file:
            self.scaler = pickle.load(file)

        self.data = None
        self.preprocessed_data = None
        self.df_with_predictions = None

    def load_and_clean_data(self, data_file):
        # Load new customer data
        df = pd.read_csv(data_file)

        # Preserve the original dataset
        self.df_with_predictions = df.copy()

        # Remove columns not used for prediction
        df = df.drop(
            columns=["customer_id", "exited"],
            errors="ignore"
        )

        # Convert gender into dummy variables
        if "gender" in df.columns:
            gender_columns = pd.get_dummies(
                df["gender"],
                dtype=int
            )
            df = df.drop(columns=["gender"])
            df = pd.concat(
                [df, gender_columns],
                axis=1
            )

        # Convert country into dummy variables
        if "country" in df.columns:
            country_columns = pd.get_dummies(
                df["country"],
                dtype=int
            )
            df = df.drop(columns=["country"])
            df = pd.concat(
                [df, country_columns],
                axis=1
            )

        # Ensure all training features exist
        if hasattr(self.reg, "feature_names_in_"):
            expected_columns = list(
                self.reg.feature_names_in_
            )
        elif hasattr(self.scaler, "feature_names_in_"):
            expected_columns = list(
                self.scaler.feature_names_in_
            )
        else:
            raise ValueError(
                "The saved model does not contain feature names. "
                "Save the training feature columns alongside the model."
            )

        # Add missing dummy columns with zero values
        missing_columns = [
            col for col in expected_columns
            if col not in df.columns
        ]

        for col in missing_columns:
            if col in [
                "Male", "Female",
                "France", "Germany", "Spain"
            ]:
                df[col] = 0
            else:
                raise ValueError(
                    f"Missing required feature: {col}"
                )

        # Reject unexpected features
        unexpected_columns = [
            col for col in df.columns
            if col not in expected_columns
        ]

        if unexpected_columns:
            raise ValueError(
                f"Unexpected features: {unexpected_columns}"
            )

        # Reorder columns to match training data
        df = df[expected_columns]

        # Validate missing values
        if df.isna().any().any():
            raise ValueError(
                "Input contains missing values. "
                "Clean them before prediction."
            )

        # Preserve preprocessed features
        self.preprocessed_data = df.copy()

        # Apply the fitted scaler
        self.data = self.scaler.transform(df)

        return self.data

    def predicted_probability(self):
        if self.data is None:
            raise ValueError(
                "Load customer data before predicting."
            )

        # Probability that exited = 1
        class_index = list(
            self.reg.classes_
        ).index(1)

        return self.reg.predict_proba(
            self.data
        )[:, class_index]

    def predicted_output_category(self):
        if self.data is None:
            raise ValueError(
                "Load customer data before predicting."
            )

        return self.reg.predict(self.data)

    def predicted_outputs(self):
        if self.data is None:
            raise ValueError(
                "Load customer data before predicting."
            )

        results = self.df_with_predictions.copy()

        results["Probability"] = (
            self.predicted_probability()
        )

        results["Prediction"] = (
            self.predicted_output_category()
        )

        return results

    
    @staticmethod
    def train_and_save(
        training_csv,
        model_file="model.pkl",
        scaler_file="scaler.pkl"
    ):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(training_csv)

        if "exited" not in df.columns:
            raise ValueError("Training data must contain 'exited'.")

        y = df["exited"]
        X = df.drop(columns=["customer_id", "exited"], errors="ignore")

        X = pd.get_dummies(
            X,
            columns=["gender", "country"],
            dtype=int
        )

        X = X.rename(columns={
            "gender_Male": "Male",
            "gender_Female": "Female",
            "country_France": "France",
            "country_Germany": "Germany",
            "country_Spain": "Spain"
        })

        feature_columns = [
            "age", "Male", "Female", "credit_score",
            "balance", "tenure", "num_of_products",
            "has_credit_card", "is_active_member",
            "estimated_salary", "France", "Germany", "Spain"
        ]

        X = X.reindex(columns=feature_columns, fill_value=0)

        x_train, x_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        numeric_columns = [
            "age", "credit_score", "balance",
            "tenure", "num_of_products", "estimated_salary"
        ]

        scaler = CustomerScaler(columns=numeric_columns)
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        reg = LogisticRegression(max_iter=1000)
        reg.fit(x_train_scaled, y_train)

        accuracy = reg.score(x_test_scaled, y_test)

        with open(model_file, "wb") as file:
            pickle.dump(reg, file)

        with open(scaler_file, "wb") as file:
            pickle.dump(scaler, file)

        return accuracy