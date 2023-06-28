from typing import Dict, Any, List, Union, Optional, Tuple
import os
import shutil

import numpy as np
import pandas as pd
import yaml


from src.cyc_gbm.logger import CycGBMLogger
from src.cyc_gbm.distributions import initiate_distribution
from src.cyc_gbm.tune_kappa import tune_kappa
from src.cyc_gbm import CycGBM, CycGLM, Intercept


def _load_config(config_file: str) -> dict:
    """
    Load the config file.

    :param config_file: The path to the config file.
    :return: The config.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def _get_run_id(config: Dict[str, Any]) -> int:
    """Get the id for the run.
    The run id is the largest run id in the output path plus one.

    :param config: The config.
    :return: The run id.
    """
    prev_runs = [int(f.split("_")[1]) for f in os.listdir(config["output_path"]) if f[:3]=='run']
    return max(prev_runs, default=0) + 1


def _create_output_path(config: Dict[str, Any], run_id: int) -> str:
    """Create the output path for the run.

    :param config: The config.
    :param run_id: The run id.
    :return: The output path.
    """
    output_path = os.path.join(config["output_path"], f"run_{run_id}")
    os.makedirs(output_path)
    return output_path


def _initiate_logger(
    config: Dict[str, Any], run_id: int, output_path: str
) -> CycGBMLogger:
    """Initiate the logger.

    :param config: The config.
    :param run_id: The run id.
    :param output_path: The output path.
    :return: The logger.
    """
    logger = CycGBMLogger(
        run_id=run_id,
        verbose=config["verbose"],
        output_path=output_path,
    )
    return logger


def _simulate_data(
    config: Dict[str, Any], rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Simulate data for the simulation study.

    :param config: The config.
    :return: The data.
    """
    n = config["n"]
    p = config["p"]
    exec(config["parameter_function"], globals())
    parameter_function = eval("z")
    distribution = initiate_distribution(distribution=config["distribution"])

    X = rng.normal(size=(n, p))
    z = parameter_function(X=X)
    w = np.ones(n)
    y = distribution.simulate(z=z, w=w, rng=rng)
    return {"X": X, "y": y, "w": w, "z": z}


def _load_data(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load the data for the simulation study.

    :param config: The config.
    :return: The data.
    """
    df = pd.read_csv(config["input_path"])
    y = df.pop("y").to_numpy()
    w = df.pop("w").to_numpy()
    X = df.to_numpy()
    return {"X": X, "y": y, "w": w}


def _train_test_split(
    full_data: Dict[str, np.ndarray], config: Dict[str, Any], rng: np.random.Generator
) -> Dict[str, Dict[str, np.ndarray]]:
    """Split the data into train and test.

    :param full_data: The full data.
    :param config: The config.
    :param rng: The random number generator.
    :return: The train and test data.
    """

    n = full_data["X"].shape[0]
    n_test = int(n * config["test_size"])
    idx = np.arange(n)
    rng.shuffle(idx)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]
    data = {
        "train": {
            "X": full_data["X"][idx_train],
            "y": full_data["y"][idx_train],
            "w": full_data["w"][idx_train],
        },
        "test": {
            "X": full_data["X"][idx_test],
            "y": full_data["y"][idx_test],
            "w": full_data["w"][idx_test],
        },
    }
    if "z" in full_data:
        data["train"]["z"] = full_data["z"][:, idx_train]
        data["test"]["z"] = full_data["z"][:, idx_test]
    return data


def _get_data(
    config: Dict[str, Any], rng: np.random.Generator
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get the data for the simulation study.

    :param config: The config.
    :return: The data split into train and test.
    """
    if config["data_source"] == "simulation":
        full_data = _simulate_data(config=config, rng=rng)
    elif config["data_source"] == "real":
        full_data = _load_data(config=config)
    else:
        raise ValueError("Invalid data source.")

    data = _train_test_split(full_data=full_data, config=config, rng=rng)
    return data


def _fit_models(
    config: Dict[str, Any],
    data_train: Dict[str, np.ndarray],
    rng: np.random.Generator,
    logger: CycGBMLogger,
) -> Dict[str, Any]:
    """Fit the models.

    :param config: The config.
    :param data: The data.
    :param rng: The random number generator.
    :param logger: The logger.
    :return: The fitted models.
    """
    distribution = initiate_distribution(distribution=config["distribution"])
    models = {}
    for model_name in config["models"]:
        logger.append_format_level(model_name)
        logger.log("setting up model")
        if model_name == "intercept":
            model = Intercept(distribution=distribution)
        elif model_name == "cyc-glm":
            model = CycGLM(
                distribution=distribution,
                max_iter=config["glm_parameters"]["max_iter"],
                eps = float(config["glm_parameters"]["eps"]),
                tol=config["glm_parameters"]["tol"],
            )
        elif model_name in ["uni-gbm", "cyc-gbm"]:
            kappa = tune_kappa(
                X=data_train["X"],
                y=data_train["y"],
                w=data_train["w"],
                kappa_max=[config["gbm_parameters"]["kappa_max"], 0]
                if model_name == "uni_gbm"
                else config["gbm_parameters"]["kappa_max"],
                eps=config["gbm_parameters"]["eps"],
                max_depth=config["gbm_parameters"]["max_depth"],
                min_samples_leaf=config["gbm_parameters"]["min_samples_leaf"],
                distribution=distribution,
                n_splits=config["gbm_parameters"]["n_splits"],
                rng=rng,
                logger=logger,
            )["kappa"]
            model = CycGBM(
                distribution=distribution,
                kappa=kappa,
                eps=config["gbm_parameters"]["eps"],
                max_depth=config["gbm_parameters"]["max_depth"],
                min_samples_leaf=config["gbm_parameters"]["min_samples_leaf"],
            )
        else:
            raise ValueError("Invalid model name.")
        logger.log("fitting model")
        model.fit(
            X=data_train["X"], y=data_train["y"], w=data_train["w"], logger=logger
        )
        models[model_name] = model
        logger.remove_format_level()

    return models


def _get_model_predictions(
    models: Dict[str, Any],
    data: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get the model predictions.

    :param models: The models.
    :param data: The data.
    :return: The model predictions.
    """
    z_hat = {}
    for data_set in data:
        z_hat[data_set] = {}
        if "z" in data[data_set]:
            z_hat[data_set]["true"] = data[data_set]["z"]
        for model_name in models:
            z_hat[data_set][model_name] = models[model_name].predict(
                data[data_set]["X"]
            )

    return z_hat


def _get_model_losses(
    z_hat: Dict[str, Dict[str, np.ndarray]],
    data: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Get the model losses.

    :param z_hat: The model predictions.
    :param data: The data.
    :return: The model losses.
    """
    distribution = initiate_distribution(distribution=config["distribution"])
    losses = {}
    for data_set in data:
        losses[data_set] = {}
        for model_name in z_hat[data_set]:
            losses[data_set][model_name] = distribution.loss(
                y=data[data_set]["y"],
                z=z_hat[data_set][model_name],
                w=data[data_set]["w"],
            )
    return losses


def _save_results(
    config: Dict[str, Any],
    output_path: str,
    data: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    losses: Dict[str, Dict[str, float]],
) -> None:
    """Save the results.

    :param config: The config.
    :param output_path: The output path.
    :param data: The data.
    :param z_hat: The model predictions.
    :param losses: The model losses.
    """
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/sub_config.yaml", "w") as file:
        yaml.dump(config, file)
    if config["copy_data"]:
        np.savez(f"{output_path}/data", **data)
    np.savez(f"{output_path}/z_hat", **z_hat)
    np.savez(f"{output_path}/losses", **losses)


def numerical_illustration(
    config: Union[Dict[str, Any], str],
    output_path: str,
    rng: np.random.Generator,
    logger: Optional[CycGBMLogger] = None,
) -> Dict[str, Any]:
    """Run the numerical illustration.

    :param config: The configuration dictionary or path to configuration file.
    :param output_path: The output path.
    :param rng: The random number generator.
    :param logger: The logger.
    :return: The results.
    """
    if logger is None:
        logger = CycGBMLogger(verbose = -1)
    logger.log("loading data.")
    if isinstance(config, str):
        config = _load_config(config_file=config)
    data = _get_data(config=config, rng=rng)

    logger.log("fitting models.")
    models = _fit_models(
        config=config, data_train=data["train"], rng=rng, logger=logger
    )

    logger.log("predicting parameters.")
    z_hat = _get_model_predictions(models=models, data=data)

    logger.log("evaluating predictions.")
    losses = _get_model_losses(config=config, z_hat=z_hat, data=data)

    logger.log("saving results.")
    _save_results(
        config=config,
        output_path=output_path,
        data=data,
        z_hat=z_hat,
        losses=losses,
    )

    logger.log("finished.")
    return {"data": data, "z_hat": z_hat, "losses": losses}


def _consolidate_configs(
    master_config: Dict[str, Any],
    sub_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Consolidate the master and sub configs.

    :param master_config: The master config.
    :param sub_config: The sub config.
    :return: The consolidated config.
    """
    # Keys to be filled in from master config if they are missing
    master_keys = [
        "n",
        "p",
        "test_size",
        "parameter_function",
        "models",
        "glm_parameters",
        "gbm_parameters",
        "copy_data",
    ]
    config = sub_config.copy()
    for key in master_keys:
        if key not in config and key in master_config:
            config[key] = master_config[key]
    return config


def _get_sub_config_list(
    master_config: Dict[str, Any],
) -> List[str]:
    sub_config_folder_path = master_config["sub_configs"]
    config_files = [
        f"{sub_config_folder_path}/{file}"
        for file in os.listdir(sub_config_folder_path)
        if file.endswith(".yaml")
    ]
    return config_files


def numerical_illustrations(
    master_config_file: str,
) -> Dict[str, Dict[str, Any]]:
    master_config = _load_config(master_config_file)
    run_id = _get_run_id(config=master_config)
    output_path = _create_output_path(config=master_config, run_id=run_id)
    shutil.copy(master_config_file, output_path)
    logger = _initiate_logger(
        config=master_config, run_id=run_id, output_path=output_path
    )
    rng = np.random.default_rng(master_config["random_seed"])
    config_files = _get_sub_config_list(master_config=master_config)

    results = {}

    for config_file in config_files:
        sub_config = _load_config(config_file)
        config = _consolidate_configs(
            master_config=master_config, sub_config=sub_config
        )
        data_name = config["data_name"]
        logger.append_format_level(data_name)
        results[data_name] = numerical_illustration(
            config=config,
            output_path=f"{output_path}/{data_name}",
            rng=rng,
            logger=logger,
        )
        logger.remove_format_level()

    return results


if __name__ == "__main__":
    config_path = "../../config/cas_data_study"
    config_file = "master_config.yaml"
    results = numerical_illustrations(master_config_file=f"{config_path}/{config_file}")