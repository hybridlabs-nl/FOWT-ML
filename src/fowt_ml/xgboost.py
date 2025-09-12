"""The module for XGBoost model training and evaluation."""

import xgboost as xgb
from fowt_ml.base import BaseModel


class XGBoost(BaseModel):
    """Class to handle linear models and metrics for comparison."""

    ESTIMATOR_NAMES = {
        "XGBoostRegression": xgb.XGBRegressor,
    }
