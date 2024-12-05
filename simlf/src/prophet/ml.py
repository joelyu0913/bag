import copy
import os

import joblib
import numpy as np


class Model(object):
    def __init__(self):
        self.model = None

    def fit(self, X: np.array, Y: np.array, **kwargs) -> None:
        assert len(X.shape) == 2
        assert len(X) == len(Y)
        self.model.fit(X, Y, **kwargs)

    def predict(self, X: np.array, **kwargs) -> np.array:
        assert len(X.shape) == 2
        return self.model.predict(X, **kwargs)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def clone(self):
        return copy.copy(self)


class LGBM(Model):
    def __init__(self, kwargs):
        Model.__init__(self)
        import lightgbm

        default_args = {
            "learning_rate": 0.2,
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mse",
            "sub_feature": 0.5,
            "num_leaves": 10,
            "min_data": 50,
            "max_depth": 10,
            "num_threads": 1,
        }
        self.model = lightgbm.LGBMRegressor(**{**default_args, **kwargs})


class LinearRegression(Model):
    def __init__(self, kwargs):
        Model.__init__(self)
        import sklearn.linear_model

        default_args = {
            "fit_intercept": False,
        }
        if "random_state" in kwargs:
            kwargs.pop("random_state")
        self.model = sklearn.linear_model.LinearRegression(**{**default_args, **kwargs})


class MLPRegressor(Model):
    def __init__(self, kwargs):
        Model.__init__(self)
        import sklearn.neural_network

        default_args = {
            "hidden_layer_sizes": (16, 8, 8),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.0001,
            "learning_rate_init": 0.00008,
        }
        self.model = sklearn.neural_network.MLPRegressor(**{**default_args, **kwargs})


class XGBRegressor(Model):
    def __init__(self, kwargs):
        Model.__init__(self)
        import xgboost

        default_args = {
            "n_estimators": 2,
        }
        self.model = xgboost.XGBRegressor(**{**default_args, **kwargs})
