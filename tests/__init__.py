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
        names = ["acc_tb_meas3[0]", "acc_tb_meas3[1]", "acc_tb_meas3[2]"]
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
    config = f"""
    data:
        exp1:
            mat_file: {mat_file}
    ml_setup:
        target: ["acc_tb_meas3[0]"]
        predictors: ["acc_tb_meas3[1]", "acc_tb_meas3[2]"]
        models: ["en", "lr"]
        n_jobs: 1
        use_gpu: False
        system_log: False
    """
    with open(config_file, "w") as f:
        f.write(config)
