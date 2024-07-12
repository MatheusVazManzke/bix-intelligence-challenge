import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import operator
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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


class TypeFloatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X.replace("na", np.nan, inplace=True)
        if "class" in X.columns:
            X["class"].replace({"neg": 0, "pos": 1}, inplace=True)
        X = X.astype("float64")
        return X


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_transformed = self.imputer.transform(X)
        return pd.DataFrame(X_transformed, columns=X.columns)


class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaled = StandardScaler()
        self.columns_to_scale = None

    def fit(self, X, y=None):
        self.columns_to_scale = [col for col in X.columns if col != "class"]
        self.scaled.fit(X[self.columns_to_scale])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns_to_scale] = self.scaled.transform(X[self.columns_to_scale])
        return X_transformed


def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)


def save_data(data, filepath):
    """Save data to a CSV file."""
    data.to_csv(filepath, index=False)
