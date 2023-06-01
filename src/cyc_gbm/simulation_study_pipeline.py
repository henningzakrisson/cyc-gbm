import os
from typing import List, Union, Callable, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import yaml
import shutil

from src.cyc_gbm import CycGBM, CycGLM
from src.cyc_gbm.distributions import initiate_distribution, Distribution
from src.cyc_gbm.tune_kappa import tune_kappa
from src.cyc_gbm.logger import SimulationLogger


# TODO: Add weight capability
# TODO: Add real data capability


def simulation_study(
    config_file: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Run a simulation study from a configuration file.
    For parameters not specified, the default values are used.

    :param config_file: The configuration file.
    :return: The results of the simulation study.
    """
    # Save config data
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    output_path = config["output_path"]
    os.makedirs(output_path, exist_ok=True)
    run_id = _get_run_id(output_path=output_path)
    if output_path is not None:
        os.makedirs(f"{output_path}/run_{run_id}")
    logger = SimulationLogger(
        run_id=run_id,
        verbose=config["verbose"],
        output_path=output_path,
    )
    rng = np.random.default_rng(config["random_seed"])

    # Set up simulation
    n = config["n"]
    p = config["p"]
    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, p - 1))])
    parameter_functions = {}
    for distribution, parameter_function in config["parameter_functions"].items():
        exec(parameter_function, globals())
        parameter_functions[distribution] = eval("z")

    simulation_results = {}
    z_hat = {}
    losses = {}

    logger.log(f"starting simulation study")
    for dist in config["dists"]:
        logger.append_format_level(dist)

        logger.log(f"simulating data")
        distribution = initiate_distribution(distribution=dist)
        simulation_results[dist] = _simulate_data(
            X=X,
            distribution=distribution,
            parameter_function=parameter_functions[dist],
            rng=rng,
            test_size=config["test_size"],
        )

        logger.log(f"running models")
        z_hat[dist] = _get_model_predictions(
            simulation_result=simulation_results[dist],
            models=config["models"],
            distribution=distribution,
            rng=rng,
            hyper_parameters=config['hyper_parameters'][dist],
            logger=logger,
        )
        logger.log(f"calculating losses")
        losses[dist] = _get_model_losses(
            simulation_result=simulation_results[dist],
            z_hat=z_hat[dist],
            distribution=distribution,
        )

        if output_path is not None:
            logger.log(f"saving results")
            _save_data(
                simulation_result=simulation_results[dist],
                run_id=run_id,
                z_hat=z_hat[dist],
                losses=losses[dist],
                config=config,
                config_file=config_file,
                dist=dist,
            )
        logger.remove_format_level()
    logger.log(f"finished simulation study")
    return {"losses": losses, "z": z_hat}


def _get_run_id(output_path: Union[str, None]) -> int:
    """Get the run id for the simulation study.
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


def _simulate_all_data(
    config: Dict[str, Any],
    rng: np.random.Generator,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Simulate data from all distributions and parameter functions.

    :param config: The configuration.
    :param rng: The random number generator.
    :return: The simulated data.
    """
    n = config["n"]
    p = config["p"]
    dists = config["dists"]
    parameter_functions = {}
    for distribution, parameter_function in config["parameter_functions"].items():
        exec(parameter_function, globals())
        parameter_functions[distribution] = eval("z")

    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, p - 1))])
    simulation_results = {dist: {} for dist in dists}
    for dist in dists:
        distribution = initiate_distribution(distribution=dist)
        simulation_results[dist] = _simulate_data(
            X=X,
            distribution=distribution,
            parameter_function=parameter_functions[dist],
            rng=rng,
            test_size=config["test_size"],
        )
    return simulation_results


def _simulate_data(
    X: np.ndarray,
    parameter_function: Callable[[np.ndarray], np.ndarray],
    distribution: Distribution,
    rng: np.random.Generator,
    test_size: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Simulate data from a distribution and a parameter function.

    :param X: The covariates.
    :param parameter_function: The parameter function to use.
    :param distribution: The distribution to simulate from.
    :param rng: The random number generator.
    :param test_size: The size of the test set.
    :return: A dictionary with the simulated data.
    """
    z = parameter_function(X)
    y = distribution.simulate(z, rng=rng)
    X_train, X_test, y_train, y_test, z_train, z_test, _, _ = train_test_split(
        X=X, y=y, z=z, test_size=test_size, rng=rng
    )
    simulation_result = {
        "train": {
            "X": X_train,
            "y": y_train,
            "z": z_train,
        },
        "test": {
            "X": X_test,
            "y": y_test,
            "z": z_test,
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
    simulation_result: Dict[str, Dict[str, np.ndarray]],
    models: List[str],
    distribution: Distribution,
    rng: np.random.Generator,
    hyper_parameters: Dict[str, Union[int,float]],
    logger: Union[None, SimulationLogger] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get the predictions from the models.

    :param simulation_result: The simulation result.
    :param models: The models to use.
    :param distribution: The distribution.
    :param rng: The random number generator.
    :param hyper_parameters: The hyper parameters (for gbm and glm).
    :param logger: The logger.
    :return: A dictionary with the predictions and a dictionary with the losses.
    """

    X_train = simulation_result["train"]["X"]
    X_test = simulation_result["test"]["X"]
    y_train = simulation_result["train"]["y"]

    z_hat = {"train": {}, "test": {}}
    z_hat["train"]["true"] = simulation_result["train"]["z"]
    z_hat["test"]["true"] = simulation_result["test"]["z"]

    for model in models:
        logger.append_format_level(model)
        logger.log("running model")
        if model == "intercept":
            z_hat_train, z_hat_test = _run_intercept_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
            )
        elif model == "cyc-glm":
            z_hat_train, z_hat_test = _run_glm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
                parameters=hyper_parameters['glm'],
            )
        elif model == "uni-gbm":
            z_hat_train, z_hat_test = _run_gbm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
                rng=rng,
                cyclical=False,
                parameters=hyper_parameters['gbm'],
                logger=logger,
            )
        elif model == "cyc-gbm":
            z_hat_train, z_hat_test = _run_gbm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
                rng=rng,
                cyclical=True,
                parameters=hyper_parameters['gbm'],
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
    X_test: np.ndarray,
    distribution: Distribution,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run intercept model.

    :param X_train: Training covariates.
    :param X_test: Test covariates.
    :param y_train: Training response.
    :param distribution: Distribution object.
    :return: Tuple of training and test predictions.
    """
    z0 = distribution.mle(y=y_train)
    z_hat_train = np.tile(z0, (len(X_train), 1)).T
    z_hat_test = np.tile(z0, (len(X_test), 1)).T
    return z_hat_train, z_hat_test


def _run_glm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
    parameters: Dict[str, Union[float, int, List[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Run cyclic GLM model.

    :param X_train: Training covariates.
    :param X_test: Test covariates.
    :param y_train: Training response.
    :param distribution: Distribution object.
    :param parameters: Dictionary of parameters.
    :return: Tuple of training and test predictions.
    """
    max_iter = parameters["max_iter"]
    eps = parameters["eps"]
    tol = parameters["tol"]
    glm = CycGLM(distribution=distribution, max_iter=max_iter, eps=eps, tol=tol)
    glm.fit(X=X_train, y=y_train)
    z_hat_train = glm.predict(X=X_train)
    z_hat_test = glm.predict(X=X_test)
    return z_hat_train, z_hat_test


def _run_gbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
    rng: np.random.Generator,
    cyclical: bool,
    parameters: Dict[str, Union[float, int, List[float], List[int]]],
    logger: Union[None, SimulationLogger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run univariate GBM model.

    :param X_train: Training covariates.
    :param X_test: Test covariates.
    :param y_train: Training response.
    :param distribution: Distribution object.
    :param rng: Random number generator for the cross-validation.
    :param parameters: Dictionary of parameters.
    :param logger: Simulation logger.
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
    gbm.fit(X=X_train, y=y_train)
    z_hat_train = gbm.predict(X=X_train)
    z_hat_test = gbm.predict(X=X_test)
    return z_hat_train, z_hat_test


def _get_model_losses(
    simulation_result: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    distribution: Distribution,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Get the model losses.

    :param simulation_result: The simulation results.
    :param z_hat: The model predictions.
    :param distribution: The distribution object.
    :return: The model losses.
    """
    losses = {"train": {}, "test": {}}
    for model in z_hat["train"].keys():
        for data_set in ["train", "test"]:
            losses[data_set][model] = distribution.loss(
                y=simulation_result[data_set]["y"], z=z_hat[data_set][model]
            )
    return losses


def _save_data(
    dist: str,
    simulation_result: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    losses: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    run_id: int,
    config: Dict[str, Any],
    config_file: str,
):
    output_path = config["output_path"]
    shutil.copy(config_file, f"{output_path}/run_{run_id}")
    os.makedirs(f"{output_path}/run_{run_id}/{dist}")
    np.savez(f"{output_path}/run_{run_id}/{dist}/simulation", **simulation_result)
    np.savez(f"{output_path}/run_{run_id}/{dist}/z_hat", **z_hat)
    np.savez(f"{output_path}/run_{run_id}/{dist}/losses", **losses)

    if config["output_figures"]:
        _save_output_figures(
            simulation_result=simulation_result,
            z_hat=z_hat,
            output_path=f"{output_path}/run_{run_id}/{dist}",
            dist=dist,
        )


def _save_output_figures(
    simulation_result: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    dist: str,
    figure_format: str = "png",
):
    """Save the output figures.

    :param output_path: The output path.
    :param simulation_result: The simulation results.
    :param z_hat: The predicted z values.
    :param dist: The distribution.
    :param figure_format: The figure format.

    """

    figure_path = f"{output_path}/figures"
    os.makedirs(figure_path)
    fig_simulation = _create_simulation_plots(
        z=simulation_result["train"]["z"],
        y=simulation_result["train"]["y"],
        distribution=initiate_distribution(distribution=dist),
    )
    fig_results = _create_result_plots(
        z_hat=z_hat["test"],
        distribution=initiate_distribution(distribution=dist),
    )
    for figure_name, figure in [
        ("simulation", fig_simulation),
        ("results", fig_results),
    ]:
        if figure_format == "tikz":
            tikzplotlib.save(
                f"{figure_path}/{figure_name}.tex",
                figure=figure,
            )
        else:
            figure.savefig(f"{figure_path}/{figure_name}.{figure_format}")


def _create_simulation_plots(
    z: np.ndarray,
    y: np.ndarray,
    distribution: Distribution,
) -> plt.Figure:
    """
    Create plots of the simulated data.

    :param z: Array of parameters.
    :param y: Array of outcomes.
    :param distribution: Distribution object.
    :return: Figure with plots.
    """
    n = len(y)
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs = axs.flatten()
    for parameter in [0, 1]:
        axs[parameter].set_title(f"Parameter {parameter}")
        sort_order = np.argsort(z[parameter, :])
        axs[parameter].plot(z[parameter, sort_order])

    window = n // 100
    for moment_order in [1, 2]:
        axs[1 + moment_order].set_title(f"Moment {moment_order}")
        moment = distribution.moment(z=z, k=moment_order)
        sort_order = np.argsort(moment)

        if moment_order == 1:
            empirical_moment = _moving_average(y[sort_order], window)
        else:
            mean = distribution.moment(z=z, k=1)
            empirical_moment = _moving_variance(y[sort_order], mean[sort_order], window)
        axs[1 + moment_order].plot(moment[sort_order], label="True")
        axs[1 + moment_order].plot(empirical_moment, label="Empirical")
        axs[1 + moment_order].legend()

    return fig


def _moving_average(y: np.ndarray, window: int = 100):
    """Compute moving average of y with window size window.

    :param y: Array of values.
    :param window: Window size.
    :return: Moving average of y.
    """
    return np.convolve(y, np.ones(window), mode="same") / window


def _moving_variance(y: np.ndarray, mean: np.ndarray, window: int = 100):
    """Compute moving variance of y with window size window.

    :param y: Array of values.
    :param mean: Array of means.
    :param window: Window size.
    :return: Moving variance of y.
    """
    return _moving_average((y - mean) ** 2, window)


def _create_result_plots(
    z_hat: Dict[str, np.ndarray],
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
        sort_order = np.argsort(z_hat["true"][parameter, :])
        for model in z_hat.keys():
            axs[parameter].plot(
                z_hat[model][parameter, sort_order],
                label=model,
            )
        axs[parameter].legend()

    for moment_order in [1, 2]:
        axs[1 + moment_order].set_title(f"Moment {moment_order}")
        moment = distribution.moment(z=z_hat["true"], k=moment_order)
        sort_order = np.argsort(moment)
        for model in z_hat.keys():
            axs[1 + moment_order].plot(
                distribution.moment(z=z_hat[model], k=moment_order)[sort_order],
                label=model,
            )
        axs[1 + moment_order].legend()

    return fig


if __name__ == "__main__":
    config_file = "../../config/simulation_config.yaml"
    simulation_study(config_file=config_file)
