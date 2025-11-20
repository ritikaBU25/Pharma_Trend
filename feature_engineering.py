import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, date_col="ds", value_col="y"):
        self.date_col = date_col
        self.value_col = value_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df = df.rename(columns={self.date_col: "ds", self.value_col: "y"})

        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"])

        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["y"])

        df["y"] = df["y"].clip(lower=1)

        df["day"] = df["ds"].dt.day
        df["month"] = df["ds"].dt.month
        df["year"] = df["ds"].dt.year
        df["dayofweek"] = df["ds"].dt.dayofweek

        return df
