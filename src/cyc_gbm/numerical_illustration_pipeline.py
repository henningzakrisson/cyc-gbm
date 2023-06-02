import os
from typing import List, Union, Callable, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import yaml
import shutil
import pandas as pd

from src.cyc_gbm import CycGBM, CycGLM
from src.cyc_gbm.distributions import initiate_distribution, Distribution
from src.cyc_gbm.tune_kappa import tune_kappa
from src.cyc_gbm.logger import CycGBMLogger


# TODO: Add real data capability
# TODO: Add progress to logger
# TODO: Remove the real data files (by using gitignore)

def numerical_illustration(
    config_file: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Run a study from a configuration file.

    :param config_file: The configuration file.
    :return: The results of the numerical illustration study as a dictionary.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)
    run_id = _get_run_id(output_path=output_path)
    if output_path is not None:
        os.makedirs(f"{output_path}/run_{run_id}")
    logger = CycGBMLogger(
        run_id=run_id,
        data_type=config["data"],
        verbose=config["verbose"],
        output_path=output_path,
    )
    rng = np.random.default_rng(config["random_seed"])

    if config["data"] == "simulation":
        logger.log(f"initiating simulation")
        n = config["n"]
        p = config["p"]
        X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, p - 1))])
        data = {}
        dists = {}
        for data_set, parameter_function in config["parameter_functions"].items():
            logger.append_format_level(data_set)
            logger.log("simulating data")
            exec(parameter_function, globals())
            parameter_function = eval("z")
            data[data_set] = _simulate_data(
                X=X,
                dist=data_set,
                parameter_function=parameter_function,
                rng=rng,
                test_size=config["test_size"],
            )
            dists[data_set] = data_set
            logger.remove_format_level()
    elif config["data"] == "real":
        logger.log(f"initiating data load")
        data = {}
        dists = {}
        for data_set in config["data_sets"]:
            logger.append_format_level(data_set)
            logger.log(f"loading data")
            data[data_set] = _load_data(
                config=config,
                data_set=data_set,
                rng=rng,
            )
            dists[data_set] = config["dists"][data_set]
            logger.remove_format_level()
    else:
        raise ValueError(f"Data type {config['data']} not supported.")

    z_hat = {}
    losses = {}

    logger.log(f"initiating model training")
    for data_set, dist in dists.items():
        logger.append_format_level(data_set)
        logger.log(f"running models")
        distribution = initiate_distribution(distribution=dist)
        z_hat[data_set] = _get_model_predictions(
            data=data[data_set],
            models=config["models"],
            distribution=distribution,
            rng=rng,
            hyper_parameters=config["hyper_parameters"][data_set],
            logger=logger,
        )
        logger.log(f"calculating losses")
        losses[data_set] = _get_model_losses(
            data=data[data_set],
            z_hat=z_hat[data_set],
            distribution=distribution,
        )

        if output_path is not None:
            logger.log(f"saving results")
            _save_data(
                data=data[data_set],
                run_id=run_id,
                z_hat=z_hat[data_set],
                losses=losses[data_set],
                config=config,
                config_file=config_file,
                distribution=distribution,
                data_set=data_set,
            )
        logger.remove_format_level()
    logger.log(f"finished numerical illustration")
    return {"data": data, "losses": losses, "z": z_hat}


def _get_run_id(output_path: Union[str, None]) -> int:
    """Get the id for the run.
    The run id is the largest run id in the output path plus one.

    :param output_path: The output path.
    :return: The run id.
    """
    if not output_path:
        return 0
    else:
        return (
            max(
                [
                    int(f.split("_")[1])
                    for f in os.listdir(output_path)
                    if os.path.isdir(os.path.join(output_path, f))
                ],
                default=0,
            )
            + 1
        )


def _load_data(
    config: Dict[str, Any],
    data_set: str,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, np.ndarray]]:
    df = pd.read_csv(config["input_paths"][data_set])
    y = df.pop("y").to_numpy()
    w = df.pop("w").to_numpy()
    X = df.to_numpy()

    X_train, X_test, y_train, y_test, z_train,z_test,w_train, w_test= train_test_split(
        X=X, y=y, w=w, test_size=config["test_size"], rng=rng
    )
    data = {
        "train": {
            "X": X_train,
            "y": y_train,
            "z": z_train,
            "w": w_train,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "z": z_test,
            "w": w_test,
        },
    }
    return data


def _simulate_data(
    X: np.ndarray,
    parameter_function: Callable[[np.ndarray], np.ndarray],
    dist: str,
    rng: np.random.Generator,
    test_size: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Simulate data from a distribution and a parameter function.

    :param X: The covariates.
    :param parameter_function: The parameter function to use.
    :param dist: The distribution to simulate from.
    :param rng: The random number generator.
    :param test_size: The size of the test set.
    :return: A dictionary with the simulated data.
    """
    w = np.ones(X.shape[0])
    z = parameter_function(X)
    distribution = initiate_distribution(distribution=dist)
    y = distribution.simulate(z, w=w, rng=rng)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        z_train,
        z_test,
        w_train,
        w_test,
    ) = train_test_split(X=X, y=y, z=z, w=w, test_size=test_size, rng=rng)
    simulation_result = {
        "train": {
            "X": X_train,
            "y": y_train,
            "z": z_train,
            "w": w_train,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "z": z_test,
            "w": w_test,
        },
    }
    return simulation_result


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    w: Union[np.ndarray, None] = None,
    z: Union[np.ndarray, None] = None,
    test_size: float = 0.8,
    random_state: Union[int, None] = None,
    rng: Union[np.random.Generator, None] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
    """Split X, y and z into a training set and a test set.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target vector of shape (n_samples,).
    :param w: The weights for the training data, of shape (n_samples,). Default is 1 for all samples.
    :param z: The parameter vector of shape (n_parameters, n_samples).
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The seed used by the random number generator.
    :param rng: The random number generator.
    :return: X_train, X_test, y_train, y_test, z_train, z_test
    """
    if rng is None:
        rng = np.random.default_rng(random_state)
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    idx = rng.permutation(n_samples)
    idx_test = idx[:n_test]
    idx_train = idx[n_test:]
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    z_train, z_test = (
        (z[:, idx_train], z[:, idx_test]) if z is not None else (None, None)
    )
    w_train, w_test = (w[idx_train], w[idx_test]) if w is not None else (None, None)
    return X_train, X_test, y_train, y_test, z_train, z_test, w_train, w_test


def _get_model_predictions(
    data: Dict[str, Dict[str, np.ndarray]],
    models: List[str],
    distribution: Distribution,
    rng: np.random.Generator,
    hyper_parameters: Dict[str, Union[int, float]],
    logger: Union[None, CycGBMLogger] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get the predictions from the models.

    :param data: The data.
    :param models: The models to use.
    :param distribution: The distribution.
    :param rng: The random number generator.
    :param hyper_parameters: The hyper parameters (for gbm and glm).
    :param logger: The logger.
    :return: A dictionary with the predictions and a dictionary with the losses.
    """

    X_train = data["train"]["X"]
    X_test = data["test"]["X"]
    w_train = data["train"]["w"]
    y_train = data["train"]["y"]

    z_hat = {"train": {}, "test": {}}
    if data["train"]["z"] is not None:
        z_hat["train"]["true"] = data["train"]["z"]
        z_hat["test"]["true"] = data["test"]["z"]

    for model in models:
        logger.append_format_level(model)
        logger.log("running model")
        if model == "intercept":
            z_hat_train, z_hat_test = _run_intercept_model(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_test=X_test,
                distribution=distribution,
            )
        elif model == "cyc-glm":
            z_hat_train, z_hat_test = _run_glm_model(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_test=X_test,
                distribution=distribution,
                parameters=hyper_parameters["glm"],
            )
        elif model == "uni-gbm":
            z_hat_train, z_hat_test = _run_gbm_model(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_test=X_test,
                distribution=distribution,
                rng=rng,
                cyclical=False,
                parameters=hyper_parameters["gbm"],
                logger=logger,
            )
        elif model == "cyc-gbm":
            z_hat_train, z_hat_test = _run_gbm_model(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                X_test=X_test,
                distribution=distribution,
                rng=rng,
                cyclical=True,
                parameters=hyper_parameters["gbm"],
                logger=logger,
            )
        else:
            raise ValueError(f"Model {model} not recognized.")
        z_hat["train"][model] = z_hat_train
        z_hat["test"][model] = z_hat_test
        logger.remove_format_level()

    return z_hat


def _run_intercept_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run intercept model.

    :param X_train: Training covariates.
    :param y_train: Training response.
    :param w_train: Training weights.
    :param X_test: Test covariates.
    :param distribution: Distribution object.
    :return: Tuple of training and test predictions.
    """
    z0 = distribution.mle(y=y_train, w=w_train)
    z_hat_train = np.tile(z0, (len(X_train), 1)).T
    z_hat_test = np.tile(z0, (len(X_test), 1)).T
    return z_hat_train, z_hat_test


def _run_glm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
    parameters: Dict[str, Union[float, int, List[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run cyclic GLM model.

    :param X_train: Training covariates.
    :param y_train: Training response.
    :param w_train: Training weights.
    :param X_test: Test covariates.
    :param distribution: Distribution object.
    :param parameters: Dictionary of parameters.
    :return: Tuple of training and test predictions.
    """
    max_iter = parameters["max_iter"]
    eps = parameters["eps"]
    tol = parameters["tol"]
    glm = CycGLM(distribution=distribution, max_iter=max_iter, eps=eps, tol=tol)
    glm.fit(X=X_train, y=y_train, w=w_train)
    z_hat_train = glm.predict(X=X_train)
    z_hat_test = glm.predict(X=X_test)
    return z_hat_train, z_hat_test


def _run_gbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
    rng: np.random.Generator,
    cyclical: bool,
    parameters: Dict[str, Union[float, int, List[float], List[int]]],
    logger: Union[None, CycGBMLogger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run univariate GBM model.

    :param X_train: Training covariates.
    :param y_train: Training response.
    :param w_train: Training weights.
    :param X_test: Test covariates.
    :param distribution: Distribution object.
    :param rng: Random number generator for the cross-validation.
    :param parameters: Dictionary of parameters.
    :param logger: Custom CycGBM logger.
    :return: Tuple of training and test predictions.
    """

    kappa_max = parameters["kappa_max"]
    if not cyclical and isinstance(kappa_max, list):
        kappa_max = kappa_max[0]
    eps = parameters["eps"]
    max_depth = parameters["max_depth"]
    min_samples_leaf = parameters["min_samples_leaf"]
    n_splits = parameters["n_splits"]

    kappa = tune_kappa(
        X=X_train,
        y=y_train,
        w=w_train,
        distribution=distribution,
        kappa_max=kappa_max if cyclical else [kappa_max, 0],
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_splits=n_splits,
        rng=rng,
        logger=logger,
    )["kappa"]
    gbm = CycGBM(
        distribution=distribution,
        kappa=kappa,
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    gbm.fit(X=X_train, y=y_train, w=w_train)
    z_hat_train = gbm.predict(X=X_train)
    z_hat_test = gbm.predict(X=X_test)
    return z_hat_train, z_hat_test


def _get_model_losses(
    data: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    distribution: Distribution,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Get the model losses.

    :param data: The input data.
    :param z_hat: The model predictions.
    :param distribution: The distribution object.
    :return: The model losses.
    """
    losses = {"train": {}, "test": {}}
    for model in z_hat["train"].keys():
        for data_set in ["train", "test"]:
            losses[data_set][model] = distribution.loss(
                y=data[data_set]["y"],
                z=z_hat[data_set][model],
                w=data[data_set]["w"],
            )
    return losses


def _save_data(
    distribution: Distribution,
    data_set: str,
    data: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    losses: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    run_id: int,
    config: Dict[str, Any],
    config_file: str,
):
    output_path = config["output_path"]
    shutil.copy(config_file, f"{output_path}/run_{run_id}")
    os.makedirs(f"{output_path}/run_{run_id}/{data_set}")
    np.savez(f"{output_path}/run_{run_id}/{data_set}/data", **data)
    np.savez(f"{output_path}/run_{run_id}/{data_set}/z_hat", **z_hat)
    np.savez(f"{output_path}/run_{run_id}/{data_set}/losses", **losses)

    if config["output_figures"]:
        _save_output_figures(
            data=data,
            z_hat=z_hat,
            output_path=f"{output_path}/run_{run_id}/{data_set}",
            distribution=distribution,
        )


def _save_output_figures(
    data: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    distribution: Distribution,
    figure_format: str = "png",
):
    """Save the output figures.

    :param output_path: The output path.
    :param data: The data.
    :param z_hat: The predicted z values.
    :param dist: The distribution.
    :param figure_format: The figure format.

    """

    figure_path = f"{output_path}/figures"
    os.makedirs(figure_path)
    fig_data = _create_data_plots(
        z=data["train"]["z"],
        y=data["train"]["y"],
        w=data["train"]["w"],
        distribution=distribution,
    )
    fig_results = _create_result_plots(
        z_hat=z_hat["test"],
        y=data["test"]["y"],
        w=data["test"]["w"],
        distribution=distribution,
    )
    for figure_name, figure in [
        ("data", fig_data),
        ("results", fig_results),
    ]:
        if figure_format == "tikz":
            tikzplotlib.save(
                f"{figure_path}/{figure_name}.tex",
                figure=figure,
            )
        else:
            figure.savefig(f"{figure_path}/{figure_name}.{figure_format}")


def _create_data_plots(
    z: Union[None,np.ndarray],
    y: np.ndarray,
    w: np.ndarray,
    distribution: Distribution,
) -> plt.Figure:
    """
    Create plots for the data.

    :param z: Array of parameters.
    :param y: Array of outcomes.
    :param w: Array of weights.
    :param distribution: Distribution object.
    :return: Figure with plots.
    """
    if z is not None:
        return _create_simulated_data_plots(
            z=z,
            y=y,
            w=w,
            distribution=distribution,
        )
    else:
        return _create_real_data_plots(
            y=y,
        )


def _create_simulated_data_plots(
    z: Union[None,np.ndarray],
    y: np.ndarray,
    w: np.ndarray,
    distribution: Distribution,
) -> plt.Figure:
    """
    Create plots for the simulated data.

    :param z: True parameters.
    :param y: Outcomes.
    :param w: Data weights.
    :param distribution: Distribution object.
    :return: A figure with plots.
    """
    n = len(y)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs = axs.flatten()
    for parameter in [0, 1]:
        axs[parameter].set_title(f"Parameter {parameter}")
        sort_order = np.argsort(z[parameter, :])
        axs[parameter].plot(z[parameter, sort_order])
        axs[parameter].set_xlim([0, n])

    window = n // 100
    for moment_order in [1, 2]:
        axs[1 + moment_order].set_title(f"Moment {moment_order}")
        moment = distribution.moment(z=z, w=w, k=moment_order)
        sort_order = np.argsort(moment)

        if moment_order == 1:
            empirical_moment = _moving_average(y[sort_order], window)
        else:
            mean = distribution.moment(z=z, w=w, k=1)
            empirical_moment = _moving_variance(y[sort_order], mean[sort_order], window)
        axs[1 + moment_order].plot(moment[sort_order], label="True")
        axs[1 + moment_order].plot(empirical_moment, label="Empirical")
        axs[1 + moment_order].legend()
        axs[1 + moment_order].set_xlim([window, n - window])

    return fig


def _create_real_data_plots(
        y: np.ndarray,
) -> plt.Figure:
    """
    Create plots for the real data.

    :param y: Outcomes.
    :return: A figure with plots.
    """
    n = len(y)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs = axs.flatten()

    window = n // 100
    for moment_order in [1, 2]:
        sort_order = np.argsort(y)

        if moment_order == 1:
            empirical_moment = _moving_average(y[sort_order], window)
        else:
            mean = empirical_moment
            empirical_moment = _moving_variance(y[sort_order], mean[sort_order], window)
        axs[moment_order-1].set_title(f"Moment {moment_order}")
        axs[moment_order-1].plot(empirical_moment, label="Empirical")
        axs[moment_order-1].legend()
        axs[moment_order-1].set_xlim([window, n - window])

    return fig


def _moving_average(y: np.ndarray, window: int = 100):
    """Compute moving average of y with window size window.

    :param y: Array of values.
    :param window: Window size.
    :return: Moving average of y.
    """
    return np.convolve(y, np.ones(window), mode="same") / window


def _moving_variance(y: np.ndarray, mean: np.ndarray, window: int = 100) -> object:
    """Compute moving variance of y with window size window.

    :param y: Array of values.
    :param mean: Array of means.
    :param window: Window size.
    :return: Moving variance of y.
    """
    return _moving_average((y - mean) ** 2, window)


def _create_result_plots(
    z_hat: Dict[str, np.ndarray],
    y: np.ndarray,
    w: np.ndarray,
    distribution: Distribution,
) -> plt.Figure:
    """
    Create plots of the results.

    :param z_hat: Dictionary of parameter estimates.
    :param distribution: Distribution object.
    :return: Figure with plots.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs = axs.flatten()

    for parameter in [0, 1]:
        axs[parameter].set_title(f"Parameter {parameter}")
        if "true" in z_hat.keys():
            sort_order = np.argsort(z_hat["true"][parameter, :])
        else:
            sort_order = np.argsort(y/w)
        for model in z_hat.keys():
            axs[parameter].plot(
                z_hat[model][parameter, sort_order],
                label=model,
            )
        axs[parameter].legend()
        axs[parameter].set_xlim([0, len(y)])

    for moment_order in [1, 2]:
        axs[1 + moment_order].set_title(f"Moment {moment_order}")
        if "true" in z_hat.keys():
            sort_order = np.argsort(distribution.moment(z=z_hat["true"], w=w, k=moment_order))
        else:
            sort_order = np.argsort(y/w)
        for model in z_hat.keys():
            axs[1 + moment_order].plot(
                distribution.moment(z=z_hat[model], w=w, k=moment_order)[sort_order],
                label=model,
            )
        axs[1 + moment_order].legend()
        axs[1 + moment_order].set_xlim([0, len(y)])

    return fig


if __name__ == "__main__":
    config_file = "../../config/real_data_config.yaml"
    numerical_illustration(config_file=config_file)
