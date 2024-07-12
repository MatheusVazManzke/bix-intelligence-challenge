import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import operator


class RandomNoiseColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_seed=42):
        self.random_seed = random_seed

    def fit(self, X):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        np.random.seed(self.random_seed)
        X_transformed["random_noise"] = np.random.randn(len(X_transformed))
        return X_transformed


class DropNATransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.columns_to_drop = None

    def fit(self, X, y=None):
        # Calculate the threshold number of NaN values per column
        threshold_count = int(len(X) * self.threshold)

        # Identify columns with more NaN values than the threshold
        self.columns_to_drop = X.columns[X.isnull().sum() > threshold_count].tolist()

        return self

    def transform(self, X):
        # Drop columns identified during fit
        X_transformed = X.drop(columns=self.columns_to_drop, errors="ignore")
        return X_transformed


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        self.columns_to_exlude = list(
            X[X.columns.difference(self.columns)].columns
        )  # List the columns that are not in the list
        self.columns_to_exlude = [col.lower().replace(" ", "_") for col in self.columns]
        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed = X_transformed.drop(columns=self.columns_to_exlude)

        return X_transformed


class KeepColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        self.columns_to_keep = self.columns  # List the columns that are not in the list
        self.columns_to_keep = [col.lower().replace(" ", "_") for col in self.columns]
        return self

    def transform(self, X):
        X_transformed = X.copy()

        X_transformed = X_transformed[self.columns]

        return X_transformed


def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)


def save_data(data, filepath):
    """Save data to a CSV file."""
    data.to_csv(filepath, index=False)
