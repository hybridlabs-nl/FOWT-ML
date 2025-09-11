"""Module to handle linear models."""

import sklearn.linear_model as lm
from fowt_ml.base import BaseModel


class LinearModels(BaseModel):
    """Class to handle linear models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "LinearRegression": lm.LinearRegression,
        "RidgeRegression": lm.Ridge,
        "LassoRegression": lm.Lasso,
        "ElasticNetRegression": lm.ElasticNet,
        "LeastAngleRegression": lm.Lars,
    }
