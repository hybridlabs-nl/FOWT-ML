import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import check_scoring
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from fowt_ml.base import BaseModel
from fowt_ml.base import TimeSeriesStandardScaler
from fowt_ml.datasets import create_segments
from fowt_ml.neural_network import NeuralNetwork


class DummyModel(BaseModel):
    ESTIMATOR_NAMES = {
        "LinearRegression": LinearRegression,
    }


class TestInit:
    def test_init_with_string(self):
        model = DummyModel("LinearRegression")
        assert isinstance(model.estimator, LinearRegression)

    def test_init_with_estimator(self):
        estimator = LinearRegression()
        model = DummyModel(estimator)
        assert model.estimator is estimator

    def test_init_with_kwrags(self):
        model = DummyModel("LinearRegression", fit_intercept=False)
        assert model.estimator.fit_intercept is False


class TestCalculateScore:
    def test_calculate_score_default(self):
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])
        x_test = np.array([[6], [7], [8]])
        y_test = np.array([7, 8, 9])

        model.estimator.fit(x_train, y_train)
        metrics = ["r2"]
        scorer = check_scoring(model.estimator, scoring=metrics)
        expected_scores = scorer(model.estimator, x_test, y_test)

        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)
        assert actual_scores == expected_scores

    def test_calculate_score_metrics(self):
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])
        x_test = np.array([[6], [7], [8]])
        y_test = np.array([7, 8, 9])

        model.estimator.fit(x_train, y_train)
        scorer = check_scoring(model.estimator, scoring=["r2"])
        expected_scores = scorer(model.estimator, x_test, y_test)

        metrics = ["r2", "model_fit_time", "model_predict_time"]
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)
        assert actual_scores["r2"] == expected_scores["r2"]
        assert actual_scores["model_fit_time"] >= 0
        assert actual_scores["model_predict_time"] >= 0

    def test_calculate_score_3d(self):
        dtype = np.float32
        x_train = np.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=dtype)
        y_train = np.asarray([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=dtype)
        x_test = np.asarray([[6, 7], [7, 8], [8, 9]], dtype=dtype)
        y_test = np.asarray([[7, 7], [8, 8], [9, 9]], dtype=dtype)
        params = {
            "input_size": 2,
            "hidden_size": 6,
            "output_size": 2,
            "num_layers": 2,
        }

        # reshape data to 3d (samples, 1, features)
        x_train = create_segments(x_train, 2)
        x_test = create_segments(x_test, 2)

        # reshape targets to 2d (samples, 2)
        y_train = y_train[2 - 1 :]  # adjust for sequence length
        y_test = y_test[2 - 1 :]

        model = NeuralNetwork("RNNRegressor", **params)
        model.estimator.fit(x_train, y_train)
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, ["r2"])
        assert "r2" in actual_scores


class TestCrossValidate:
    def test_cross_validate_default(self):
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])

        metrics = ["model_fit_time"]
        cv_results = model.cross_validate(x_train, y_train, scoring=metrics)
        assert "fit_time" not in cv_results
        assert "model_fit_time" in cv_results
        assert len(cv_results["model_fit_time"]) == 5

    def test_cross_validate_kwargs(self):
        model = DummyModel("LinearRegression")
        x_train = np.array([[1], [2], [3], [4], [5]])
        y_train = np.array([2, 3, 4, 5, 6])

        metrics = ["r2", "model_fit_time", "model_predict_time"]
        cv_results = model.cross_validate(x_train, y_train, scoring=metrics, cv=3)
        assert "test_r2" not in cv_results
        assert "r2" in cv_results
        assert "model_fit_time" in cv_results
        assert "model_predict_time" in cv_results
        assert len(cv_results["r2"]) == 3

    def test_cross_validate_3d(self):
        dtype = np.float32
        x_train = np.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=dtype)
        y_train = np.asarray([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=dtype)
        params = {
            "input_size": 2,
            "hidden_size": 6,
            "output_size": 2,
            "num_layers": 1,
        }

        # reshape data to 3d (samples, 2, features)
        x_train = create_segments(x_train, 2)

        # reshape targets to 2d (samples, 2)
        y_train = y_train[2 - 1 :]  # adjust for sequence length

        model = NeuralNetwork("RNNRegressor", **params)
        metrics = ["r2"]
        cv_results = model.cross_validate(x_train, y_train, scoring=metrics, cv=2)
        assert "test_r2" not in cv_results
        assert "r2" in cv_results
        assert len(cv_results["r2"]) == 2


class TestUseScaledData:
    def test_use_scaled_data_default(self):
        model = DummyModel("LinearRegression")

        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42
        )
        y_train = y_train * 1000  # pretend target is in dollars
        model.use_scaled_data()
        model.estimator.fit(x_train, y_train)
        y_predict = model.estimator.predict(x_train)

        # Check scaling wrappers
        assert isinstance(model.estimator, TransformedTargetRegressor), (
            "Estimator should be wrapped in TransformedTargetRegressor"
        )

        pipeline = model.estimator.regressor
        assert isinstance(pipeline, Pipeline), "Regressor should be a Pipeline"
        assert isinstance(pipeline.named_steps["scaler"], StandardScaler), (
            "Pipeline should contain a StandardScaler for X"
        )

        assert isinstance(model.estimator.transformer_, StandardScaler), (
            "Target transformer should be StandardScaler"
        )

        # Check prediction accuracy
        r2 = r2_score(y_train, y_predict)
        np.testing.assert_almost_equal(
            r2,
            0.99,
            decimal=2,
            err_msg=f"R2 score should be approximately {r2} after scaling",
        )

    def test_use_scaled_data_calculate_score(self):
        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42
        )
        y_train = y_train * 1000  # pretend target is in dollars
        x_test, y_test = make_regression(
            n_samples=50, n_features=3, noise=10, random_state=24
        )
        y_test = y_test * 1000

        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        x_test = np.asarray(x_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)

        model = DummyModel("LinearRegression")
        model.estimator.fit(x_train, y_train)
        metrics = ["r2", "neg_mean_squared_error"]
        expected_scores = model.calculate_score(
            x_train, x_test, y_train, y_test, metrics
        )

        # now scale data and check scores again
        model.use_scaled_data()
        actual_scores = model.calculate_score(x_train, x_test, y_train, y_test, metrics)

        np.testing.assert_almost_equal(
            actual_scores["r2"],
            expected_scores["r2"],
            decimal=2,
            err_msg="R2 scores should match after scaling",
        )
        np.testing.assert_almost_equal(
            actual_scores["neg_mean_squared_error"],
            expected_scores["neg_mean_squared_error"],
            decimal=2,
            err_msg="Neg MSE scores should match after scaling",
        )

    def test_use_scaled_data_cv(self):
        model = DummyModel("LinearRegression")

        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42
        )
        y_train = y_train * 1000  # pretend target is in dollars

        x_train = np.asarray(x_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        model.estimator.fit(x_train, y_train)
        metrics = ["r2", "neg_mean_squared_error"]
        expected_scores = model.cross_validate(x_train, y_train, metrics)

        # now scale data and check scores again
        model.use_scaled_data()
        actual_scores = model.cross_validate(x_train, y_train, metrics)

        np.testing.assert_almost_equal(
            actual_scores["r2"],
            expected_scores["r2"],
            decimal=2,
            err_msg="R2 scores should match after scaling",
        )
        np.testing.assert_almost_equal(
            actual_scores["neg_mean_squared_error"],
            expected_scores["neg_mean_squared_error"],
            decimal=2,
            err_msg="Neg MSE scores should match after scaling",
        )

    def test_use_scaled_data_rnn(self):
        x_train, y_train = make_regression(
            n_samples=200, n_features=3, noise=10, random_state=42, n_targets=2
        )
        y_train = y_train * 1000  # pretend target is in dollars
        x_test, y_test = make_regression(
            n_samples=50, n_features=3, noise=10, random_state=24, n_targets=2
        )
        y_test = y_test * 1000
        x_train = np.asarray(x_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        x_test = np.asarray(x_test, dtype=np.float32)
        y_test = np.asarray(y_test, dtype=np.float32)

        params = {
            "input_size": 3,
            "hidden_size": 6,
            "output_size": 2,
            "num_layers": 2,
        }

        model = NeuralNetwork("RNNRegressor", **params)
        model.use_scaled_data()
        actual_scores = model.calculate_score(
            x_train, x_test, y_train, y_test, ["neg_mean_absolute_error"]
        )

        y_pred = model.estimator.predict(x_test)
        expected_neg_mae = -mean_absolute_error(
            y_test, y_pred, multioutput="uniform_average"
        )
        np.testing.assert_almost_equal(
            actual_scores["neg_mean_absolute_error"],
            expected_neg_mae,
            decimal=5,
            err_msg="Neg MAE scores should match after scaling",
        )
        assert y_pred.shape == y_test.shape
        assert "neg_mean_absolute_error" in actual_scores

    def test_use_scaled_data_rnn_with_segments(self):
        dtype = np.float32
        x_train = np.asarray([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=dtype)
        y_train = np.asarray([[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], dtype=dtype)
        x_test = np.asarray([[6, 7], [7, 8], [8, 9]], dtype=dtype)
        y_test = np.asarray([[7, 7], [8, 8], [9, 9]], dtype=dtype)
        params = {
            "input_size": 2,
            "hidden_size": 6,
            "output_size": 2,
            "num_layers": 2,
        }

        # reshape data to 3d (samples, 3, features)
        seq_len = 3
        x_train = create_segments(x_train, seq_len)
        x_test = create_segments(x_test, seq_len)

        # reshape targets to 2d (samples, 2)
        y_train = y_train[seq_len - 1 :]  # adjust for sequence length
        y_test = y_test[seq_len - 1 :]

        model = NeuralNetwork("RNNRegressor", **params)
        model.use_scaled_data(data_3d=True)
        actual_scores = model.calculate_score(
            x_train, x_test, y_train, y_test, ["neg_mean_absolute_error"]
        )

        y_pred = model.estimator.predict(x_test)
        expected_neg_mae = -mean_absolute_error(
            y_test, y_pred, multioutput="uniform_average"
        )
        np.testing.assert_almost_equal(
            actual_scores["neg_mean_absolute_error"],
            expected_neg_mae,
            decimal=5,
            err_msg="Neg MAE scores should match after scaling",
        )
        assert y_pred.shape == y_test.shape
        assert "neg_mean_absolute_error" in actual_scores


class TestTimeSeriesStandardScaler:
    def test_timeseries_standard_scaler(self):
        dtype = np.float32
        x_train = np.asarray([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=dtype)

        scaler = TimeSeriesStandardScaler()
        x_scaled = scaler.fit_transform(x_train)

        # Check mean and std dev of scaled data
        mean = np.mean(x_scaled, axis=(0, 1))
        std = np.std(x_scaled, axis=(0, 1))

        np.testing.assert_almost_equal(
            mean,
            0.0,
            decimal=6,
            err_msg="Mean of scaled data should be approximately 0",
        )
        np.testing.assert_almost_equal(
            std,
            1.0,
            decimal=6,
            err_msg="Std dev of scaled data should be approximately 1",
        )
