"""Test utilities for the `tests` package."""


def create_dummy_mat_file(file_name):
    import h5py
    import numpy as np

    data_id = "exp1"
    with h5py.File(file_name, "w") as hdf:
        # Create the top-level group
        grp = hdf.create_group(data_id)

        # Create the 'X' subgroup with 'Data' dataset
        x_grp = grp.create_group("X")

        # Example time data
        x_grp.create_dataset("Data", data=np.linspace(0, 10, 50))

        # Create the 'Y' subgroup with 'Name' and 'Data' datasets
        y_grp = grp.create_group("Y")

        # Example names and data
        names = [
            "pos_act6[0]",
            "force_tt_meas6[0]",
            "force_tt_meas6[1]",
            "force_tt_meas6[2]",
        ]
        data = [np.random.rand(50) for _ in names]

        # Create references for the names and data
        name_refs = []
        data_refs = []
        for i, (name, data_array) in enumerate(zip(names, data, strict=False)):
            # Create datasets for each name and data
            name_ds = hdf.create_dataset(
                f"name_{i}", data=np.array([[ord(c)] for c in name], dtype="uint8")
            )
            data_ds = hdf.create_dataset(f"data_{i}", data=data_array)

            # Store references
            name_refs.append(name_ds.ref)
            data_refs.append(data_ds.ref)

        # Store references in the 'Y' group
        y_grp.create_dataset("Name", data=np.array(name_refs, dtype=h5py.ref_dtype))
        y_grp.create_dataset("Data", data=np.array(data_refs, dtype=h5py.ref_dtype))


def creat_dummy_config(config_file, mat_file):
    model_names = {
        "LinearRegression": {},
        "SklearnGPRegressor": {"num_inducing": 50, "num_latents": 3, "num_epochs": 1},
        "RNNRegressor": {
            "input_size": 3,
            "hidden_size": 5,
            "num_layers": 2,
            "output_size": 1,
            "max_epochs": 5,
        },
    }
    train_test_split_kwargs = {"train_size": 0.75, "shuffle": True, "random_state": 42}

    config = f"""
    name: "dummy_experiment"
    description: "A dummy experiment for testing."
    data:
        exp1:
            path_file: {mat_file}
    ml_setup:
        targets: ["pos_act6[0]"]
        predictors: ["force_tt_meas6[0]", "force_tt_meas6[1]", "force_tt_meas6[2]"]
        model_names: {model_names}
        n_jobs: 1
        use_gpu: False
        scale_data: True
        save_grid_scores: True
        save_best_model: True
        log_experiment: False
        train_test_split_kwargs: {train_test_split_kwargs}
        cross_validation_kwargs:
            cv: 5
        metric_names: ["r2", "model_fit_time", "model_predict_time"]
    session_setup:
        work_dir: "."
    """
    with open(config_file, "w") as f:
        f.write(config)
