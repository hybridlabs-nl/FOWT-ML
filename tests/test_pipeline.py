from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted
from fowt_ml.config import Config
from fowt_ml.pipeline import Pipeline
from . import creat_dummy_config
from . import create_dummy_mat_file


class TestPipelineInit:
    def test_init_config_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)

        my_pipeline = Pipeline(config_file)
        my_pipeline.work_dir = tmp_path

        assert hasattr(my_pipeline, "predictors_labels")
        assert hasattr(my_pipeline, "target_labels")
        assert "LinearRegression" in my_pipeline.model_names

    def test_init_config_dict(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        cfg = Config.from_yaml(config_file)

        my_pipeline = Pipeline(cfg)

        assert hasattr(my_pipeline, "predictors_labels")
        assert hasattr(my_pipeline, "target_labels")
        assert "LinearRegression" in my_pipeline.model_names


class TestPipelineGetData:
    def test_get_data(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test get_data
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        assert df.shape == (50, 5)


class TestPipelineSplit:
    def test_train_test_split_default(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test train_test_split
        my_pipeline = Pipeline(config_file)

        df = my_pipeline.get_data("exp1")
        my_pipeline.X_data = df[my_pipeline.predictors_labels]
        my_pipeline.Y_data = df[my_pipeline.target_labels]
        results = my_pipeline.train_test_split()
        assert len(results) == 4

        # X_train and Y_train have same number of samples
        assert results[0].shape[0] == results[2].shape[0]

        # X_test and Y_test have same number of samples
        assert results[1].shape[0] == results[3].shape[0]

        # X_train and X_test have same number of features
        assert results[0].shape[1] == results[1].shape[1]

    def test_train_test_split_kwargs(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test train_test_split
        my_pipeline = Pipeline(config_file)

        df = my_pipeline.get_data("exp1")
        my_pipeline.X_data = df[my_pipeline.predictors_labels]
        my_pipeline.Y_data = df[my_pipeline.target_labels]
        results = my_pipeline.train_test_split(**{"train_size": 0.2})
        assert len(results) == 4

        # train size is 0.2
        assert results[0].shape[0] == 0.2 * my_pipeline.X_data.shape[0]

        # test size is 0.8
        assert results[1].shape[0] == 0.8 * my_pipeline.X_data.shape[0]


class TestPipelineGetModels:
    def test_get_models(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test get_models
        my_pipeline = Pipeline(config_file)
        models = my_pipeline.get_models()
        assert "LinearRegression" in models
        assert "SklearnGPRegressor" in models
        assert "RNNRegressor" in models
        assert len(models) == 3


class TestPipelineSetup:
    def test_setup_str(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.work_dir = tmp_path
        my_pipeline.setup(data="exp1")
        assert hasattr(my_pipeline, "X_data")
        assert hasattr(my_pipeline, "Y_data")
        assert hasattr(my_pipeline, "X_train")
        assert hasattr(my_pipeline, "X_test")
        assert hasattr(my_pipeline, "Y_train")
        assert hasattr(my_pipeline, "Y_test")
        assert hasattr(my_pipeline, "model_instances")

        # check working directory set
        assert my_pipeline.work_dir.exists()

    def test_setup_mlflow(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.work_dir = tmp_path
        my_pipeline.log_experiment = True
        my_pipeline.setup(data="exp1")

        # check mlflow directory created
        assert Path(tmp_path / "mlruns").exists()

    def test_setup_df(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        df = my_pipeline.get_data("exp1")
        my_pipeline.setup(df)
        assert hasattr(my_pipeline, "X_data")
        assert hasattr(my_pipeline, "Y_data")
        assert hasattr(my_pipeline, "X_train")
        assert hasattr(my_pipeline, "X_test")
        assert hasattr(my_pipeline, "Y_train")
        assert hasattr(my_pipeline, "Y_test")
        assert hasattr(my_pipeline, "model_instances")

    def test_setup_segment_with_segments(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.data_segmentation_kwargs = {"sequence_length": 2}
        my_pipeline.work_dir = tmp_path
        my_pipeline.setup(data="exp1")
        assert my_pipeline.X_train.ndim == 3
        assert len(my_pipeline.X_train) == len(my_pipeline.Y_train)
        assert my_pipeline.X_test.ndim == 2
        assert len(my_pipeline.X_test) == len(my_pipeline.Y_test)

    def test_setup_segment_with_segments_no_rnn(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.data_segmentation_kwargs = {"sequence_length": 2}
        my_pipeline.model_names = {"LinearRegression": {}}
        my_pipeline.work_dir = tmp_path
        with pytest.raises(
            ValueError,
            match="Timeseries segmentation is only applicable for RNN models.",
        ):
            my_pipeline.setup(data="exp1")


class TestPipelineCompare:
    def test_compare_models_default(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.work_dir = tmp_path
        my_pipeline.setup(data="exp1")
        models, scores = my_pipeline.compare_models()
        assert isinstance(scores, pd.DataFrame)
        assert "r2" in scores
        assert "model_fit_time" in scores
        assert isinstance(models, dict)
        assert "LinearRegression" in models
        assert "SklearnGPRegressor" in models
        assert "RNNRegressor" in models
        # check model is fitted
        assert check_is_fitted(models["LinearRegression"]) is None
        assert Path(tmp_path / "grid_scores.csv").exists()

        # scale_data is true in config, so joblib model should exist
        assert Path(tmp_path / "best_model.joblib").exists()
        # check sorting of scores
        assert scores["r2"].iloc[0] >= scores["r2"].iloc[1]

    def test_compare_models_sort(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.setup(data="exp1")

        _, scores = my_pipeline.compare_models(sort="model_fit_time")
        assert scores["model_fit_time"].iloc[0] <= scores["model_fit_time"].iloc[1]

        _, scores = my_pipeline.compare_models(sort="model_predict_time")
        assert (
            scores["model_predict_time"].iloc[0] <= scores["model_predict_time"].iloc[1]
        )

    def test_compare_models_sort_cv(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.setup(data="exp1")

        _, scores = my_pipeline.compare_models(
            sort="model_fit_time", cross_validation=True
        )
        assert scores["model_fit_time"].iloc[0] <= scores["model_fit_time"].iloc[1]

        _, scores = my_pipeline.compare_models(
            sort="model_predict_time", cross_validation=True
        )
        assert (
            scores["model_predict_time"].iloc[0] <= scores["model_predict_time"].iloc[1]
        )

    def test_compare_models_onnx(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)

        # ONNX does not support TransformedTargetRegressor
        my_pipeline.scale_data = False
        my_pipeline.model_names = {
            "LinearRegression": {},
        }  # choose one model to control the test
        my_pipeline.setup(data="exp1")
        model, scores = my_pipeline.compare_models()
        best_model_name = scores.index[0]
        fitted_model = model[best_model_name]

        expected_pred = fitted_model.predict(my_pipeline.X_test)

        # read the onnx model
        import onnxruntime as rt

        sess = rt.InferenceSession("best_model.onnx")
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        actual_pred = sess.run([output_name], {input_name: my_pipeline.X_test})[0]

        np.testing.assert_allclose(expected_pred, actual_pred, rtol=1e-5)

    def test_compare_models_onnx_non_sklearn_model(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)

        # ONNX does not support TransformedTargetRegressor
        my_pipeline.scale_data = False
        my_pipeline.model_names = {
            "RNNRegressor": {
                "input_size": 3,
                "hidden_size": 5,
                "num_layers": 2,
                "output_size": 1,
                "max_epochs": 5,
            },
        }  # choose one model to control the test

        my_pipeline.setup(data="exp1")
        model, scores = my_pipeline.compare_models()
        best_model_name = scores.index[0]
        fitted_model = model[best_model_name]

        expected_pred = fitted_model.predict(my_pipeline.X_test)

        # read the onnx model
        import onnxruntime as rt

        sess = rt.InferenceSession("best_model.onnx")
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        actual_pred = sess.run([output_name], {input_name: my_pipeline.X_test})[0]

        np.testing.assert_allclose(expected_pred, actual_pred, rtol=1e-3)

    def test_compare_models_joblib(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.model_names = {
            "LinearRegression": {},
        }  # choose one model to control the test
        my_pipeline.setup(data="exp1")
        model, scores = my_pipeline.compare_models()
        best_model_name = scores.index[0]
        fitted_model = model[best_model_name]

        expected_pred = fitted_model.predict(my_pipeline.X_test)

        # read the joblib model
        import joblib

        loaded_model = joblib.load("best_model.joblib")
        actual_pred = loaded_model.predict(my_pipeline.X_test)

        np.testing.assert_allclose(expected_pred, actual_pred, rtol=1e-5)

    def test_compare_models_joblib_non_sklearn_model(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        my_pipeline.model_names = {
            "SklearnGPRegressor": {
                "num_inducing": 50,
                "num_latents": 3,
                "num_epochs": 1,
            },
        }  # choose one model to control the test
        my_pipeline.setup(data="exp1")
        model, scores = my_pipeline.compare_models()
        best_model_name = scores.index[0]
        fitted_model = model[best_model_name]
        expected_pred = fitted_model.predict(my_pipeline.X_test)

        # read the joblib model
        import joblib

        loaded_model = joblib.load("best_model.joblib")
        actual_pred = loaded_model.predict(my_pipeline.X_test)

        np.testing.assert_allclose(expected_pred, actual_pred, rtol=1e-5)

    def test_compare_models_cv(self, tmp_path):
        # create dummy files
        config_file = tmp_path / "config.yaml"
        mat_file = tmp_path / "data.mat"
        creat_dummy_config(config_file, mat_file)
        create_dummy_mat_file(mat_file)

        # test setup
        my_pipeline = Pipeline(config_file)
        # TODO: when SklearnGPRegressor is refactored, it can be added back
        my_pipeline.model_names = {
            "LinearRegression": {},
            "RNNRegressor": {
                "input_size": 3,
                "hidden_size": 5,
                "num_layers": 2,
                "output_size": 1,
                "max_epochs": 5,
            },
        }  # choose one model to control the test
        my_pipeline.work_dir = tmp_path
        my_pipeline.work_dir = tmp_path
        my_pipeline.setup(data="exp1")
        models, scores = my_pipeline.compare_models(cross_validation=True)
        assert isinstance(scores, pd.DataFrame)
        assert "r2" in scores
        assert "model_fit_time" in scores
        assert isinstance(models, dict)
        assert "LinearRegression" in models
        assert "RNNRegressor" in models
        # check model is fitted
        assert check_is_fitted(models["LinearRegression"]) is None
        assert Path(tmp_path / "grid_scores.csv").exists()
        assert Path(tmp_path / "best_model.joblib").exists()
        # check sorting of scores
        assert scores["r2"].iloc[0] >= scores["r2"].iloc[1]
