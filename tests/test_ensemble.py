import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from fowt_ml.ensemble import EnsembleModel


@pytest.fixture
def regression_data():
    """Setup regression dataset."""
    return make_regression(
        n_samples=100,
        n_features=5,
        random_state=0,
        shuffle=True,
        n_targets=2,
    )


@pytest.fixture
def train_test_data(regression_data):
    x, y = regression_data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    return x_train, x_test, y_train, y_test


class TestEnsembleModel:
    def test_init_random_forest_is_valid_estimator(self):
        model = EnsembleModel(estimator="RandomForest")
        assert isinstance(model.estimator, RandomForestRegressor)

    def test_init_extra_trees_is_valid_estimator(self):
        model = EnsembleModel(estimator="ExtraTrees")
        assert isinstance(model.estimator, ExtraTreesRegressor)

    def test_init_wrong_estimator_name_raise_error(self):
        with pytest.raises(ValueError):
            EnsembleModel(estimator="Linear")

    def test_init_properly_set_estimator_params(self):
        n_estimators = 10
        model = EnsembleModel(estimator="RandomForest", n_estimators=n_estimators)
        assert model.estimator.n_estimators == n_estimators

    def test_init_accept_estimator_instance(self):
        estimator = RandomForestRegressor()
        model = EnsembleModel(estimator=estimator)
        assert isinstance(model.estimator, RandomForestRegressor)

    def test_oob_score_works_with_default_scoring(self, regression_data):
        x, y = regression_data
        # oob_score=True means that r2 scoring is used
        model = EnsembleModel(estimator="RandomForest", oob_score=True)
        score = model.oob_score(x, y)
        assert score > 0

    def test_oob_score_automatically_enables_oob_score(self, regression_data):
        x, y = regression_data
        model = EnsembleModel(estimator="RandomForest")
        # model.oob_score sets oob_score to True and raises a warning
        with pytest.warns():
            score = model.oob_score(x, y)
        assert score > 0

    def test_oob_score_works_with_custom_scoring(self, regression_data):
        x, y = regression_data
        model = EnsembleModel(estimator="RandomForest")
        # model.oob_score sets oob_score to custom score metric and raises a warning
        with pytest.warns():
            score = model.oob_score(x, y, scoring="neg_root_mean_squared_error")
        # score is negative to make it such that higher is better
        assert score < 0

    def test_calculate_score_works_with_default_scoring(self, train_test_data):
        x_train, x_test, y_train, y_test = train_test_data
        model = EnsembleModel(estimator="RandomForest")
        score = model.calculate_score(x_train, x_test, y_train, y_test)
        # default score is r2
        assert score > 0

    def test_calculate_score_works_with_custom_scoring(self, train_test_data):
        x_train, x_test, y_train, y_test = train_test_data
        model = EnsembleModel(estimator="RandomForest")
        score = model.calculate_score(
            x_train,
            x_test,
            y_train,
            y_test,
            scoring="neg_root_mean_squared_error",
        )
        # score is negative to make it such that higher is better
        assert score["neg_root_mean_squared_error"] < 0.0

    def test_calculate_score_works_with_multiple_scorings(self, train_test_data):
        x_train, x_test, y_train, y_test = train_test_data
        model = EnsembleModel(estimator="RandomForest")
        scorings = ["r2", "neg_root_mean_squared_error"]
        scores = model.calculate_score(
            x_train, x_test, y_train, y_test, scoring=scorings
        )
        # a list of scores is returned
        assert len(scores) == len(scorings)
        assert all(s in scores for s in scorings)
