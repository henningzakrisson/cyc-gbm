import numpy as np
from src.cyc_gbm import CycGBM, CycGLM
from src.cyc_gbm.utils import tune_kappa, train_test_split
from src.cyc_gbm.distributions import initiate_distribution, Distribution
from matplotlib import pyplot as plt
import os
import tikzplotlib
from typing import List, Union, Callable, Tuple, Dict
import logging
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
formatter = logging.Formatter(
    "[%(asctime)s][%(name)s][%(levelname)s][%(message)s]", datefmt="%Y-%m-%d %H:%M"
)
logger.handlers[0].setFormatter(formatter)


# TODO: Save configuration properly (e.g. functions)
# TODO: Test the ability to load specifications from configuration file
# TODO: Replace the simulation function with the one with configuration file
# TODO: Try to reduce function headers
# TODO: Try to shorten everything and tidy up
# TODO: Allow for different hyperparameters for different distributions
# TODO: Save log file
# TODO: Add weight capability
# TODO: Add real data capability
# TODO: Add hierarchical logger (stating dist-model-etc)


def simulation_study_from_config(
    config_file: str,
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Run a simulation study from a configuration file.
    For parameters not specified, the default values are used.

    :param config_file: The configuration file.
    :return: The results of the simulation study.
    """
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    n = config.get("n", 1000)
    p = config.get("p", 9)
    dists = config.get("dists", ["normal"])
    models = config.get("models", ["intercept", "cyc-glm", "uni-gbm", "cyc-gbm"])
    parameter_functions = config.get(
        "parameter_functions", {"normal": lambda X: np.stack([X[:, 0], X[:, 0]])}
    )
    random_seed = config.get("random_seed")
    rng = config.get("rng")
    output_path = config.get("output_path")
    output_figures = config.get("output_figures", False)
    figure_format = config.get("figure_format", "png")
    test_size = config.get("test_size", 0.2)
    glm_parameters = config.get(
        "glm_parameters", {"max_iter": 1000, "eps": 1e-5, "tol": 1e-5}
    )
    gbm_parameters = config.get(
        "gbm_parameters",
        {
            "kappa_max": 1000,
            "eps": 0.01,
            "max_depth": 2,
            "min_samples_leaf": 10,
            "n_splits": 2,
        },
    )
    verbose = config.get("verbose", 0)

    return simulation_study(
        n=n,
        p=p,
        dists=dists,
        models=models,
        parameter_functions=parameter_functions,
        random_seed=random_seed,
        output_path=output_path,
        output_figures=output_figures,
        figure_format=figure_format,
        test_size=test_size,
        glm_parameters=glm_parameters,
        gbm_parameters=gbm_parameters,
        verbose=verbose,
    )


def simulation_study(
    n: int,
    p: Union[int, None],
    dists: List[Union[str, Distribution]],
    models: List[str],
    parameter_functions: Dict[str, Callable[[np.ndarray], np.ndarray]],
    random_seed: Union[int, None],
    output_path: Union[str, None],
    output_figures: bool,
    figure_format: str,
    test_size: float,
    glm_parameters: Dict[str, Union[float, int, List[float]]],
    gbm_parameters: Dict[str, Union[float, int, List[float], List[int]]],
    verbose: int,
) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """Run a simulation study.

    :param n: The number of observations.
    :param p: The number of covariates.
    :param dists: The distributions to simulate from.
    :param models: The models to use.
    :param parameter_functions: The parameter functions to use.
    :param random_seed: The random seed.
    :param output_path: The output path.
    :param output_figures: Whether to output figures.
    :param figure_format: The figure format.
    :param test_size: The test size.
    :param glm_parameters: The parameters for the GLM.
    :param gbm_parameters: The parameters for the GBM.
    :param verbose: The verbosity level.
    :return: The results of the simulation study.
    """
    rng = np.random.default_rng(random_seed)
    if output_path is not None or verbose > 0:
        run_id = _get_run_id(output_path=output_path)
    z_hat = {dist: {data_set: {} for data_set in ["train", "test"]} for dist in dists}
    losses = {dist: {data_set: {} for data_set in ["train", "test"]} for dist in dists}
    if verbose > 0:
        logger.info(f"Running simulation study with run id {run_id}")

    simulation_results = _simulate_all_data(
        n=n,
        p=p,
        dists=dists,
        parameter_functions=parameter_functions,
        rng=rng,
        test_size=test_size,
    )
    if verbose > 0:
        logger.info("Finished simulating data")

    for i, dist in enumerate(dists):
        if verbose > 0:
            logger.info(f"Running models for distribution: {dist}")
        distribution = initiate_distribution(distribution=dist)
        z_hat[dist] = _get_model_predictions(
            simulation_result=simulation_results[dist],
            models=models,
            distribution=distribution,
            rng=rng,
            gbm_parameters=gbm_parameters,
            glm_parameters=glm_parameters,
            verbose=verbose,
        )
        losses[dist] = _get_model_losses(
            z_hat=z_hat[dist],
            y_train=simulation_results[dist]["train"]["y"],
            y_test=simulation_results[dist]["test"]["y"],
            distribution=distribution,
        )
        if verbose > 0:
            logger.info(f"Finished running models for distribution: {dist}")

    if output_path is not None:
        _save_data(
            output_path=output_path,
            run_id=run_id,
            z_hat=z_hat,
            simulation_results=simulation_results,
            losses=losses,
            n=n,
            p=p,
            dists=dists,
            models=models,
            parameter_functions=parameter_functions,
            random_seed=random_seed,
            test_size=test_size,
            glm_parameters=glm_parameters,
            gbm_parameters=gbm_parameters,
            verbose=verbose,
            output_figures=output_figures,
            figure_format=figure_format,
        )
        if verbose > 0:
            logger.info(f"Saved data for run id {run_id} in {output_path}")
        if output_figures:
            _save_output_figures(
                output_path=output_path,
                run_id=run_id,
                simulation_results=simulation_results,
                z_hat=z_hat,
                dists=dists,
                figure_format=figure_format,
            )
            if verbose > 0:
                logger.info(f"Saved figures for run id {run_id} in {output_path}")
    return {"losses": losses, "z": z_hat}


def _save_data(
    output_path: str,
    run_id: int,
    simulation_results: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    z_hat: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    losses: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    n: int,
    p: Union[int, None],
    dists: List[Union[str, Distribution]],
    models: List[str],
    parameter_functions: Dict[str, Callable[[np.ndarray], np.ndarray]],
    random_seed: Union[int, None],
    output_figures: bool,
    figure_format: str,
    test_size: float,
    glm_parameters: Dict[str, Union[float, int, List[float]]],
    gbm_parameters: Dict[str, Union[float, int, List[float], List[int]]],
    verbose: int,
):
    """Save the data from the simulation study.

    :param output_path: The output path.
    :param run_id: The run id.
    :param z_hat: The predicted values.
    :param losses: The losses.
    :param n: The number of observations.
    :param p: The number of covariates.
    :param dists: The distributions to simulate from.
    :param models: The models to use.
    :param parameter_functions: The parameter functions to use.
    :param random_seed: The random seed.
    :param output_figures: Whether to output figures.
    :param figure_format: The figure format.
    :param test_size: The test size.
    :param glm_parameters: The parameters for the GLM.
    :param gbm_parameters: The parameters for the GBM.
    :param verbose: The verbosity level.
    """
    os.makedirs(f"{output_path}/run_{run_id}")
    for dist in z_hat.keys():
        os.makedirs(f"{output_path}/run_{run_id}/{dist}")
        config_dict = {
            "n": n,
            "p": p,
            "dists": dists,
            "models": models,
            "parameter_functions": parameter_functions,
            "random_seed": random_seed,
            "output_path": output_path,
            "output_figures": output_figures,
            "figure_format": figure_format,
            "test_size": test_size,
            "glm_parameters": glm_parameters,
            "gbm_parameters": gbm_parameters,
            "verbose": verbose,
        }
        with open(f"{output_path}/run_{run_id}/config.yaml", "w") as file:
            yaml.dump(config_dict, file)
        np.savez(
            f"{output_path}/run_{run_id}/{dist}/simulation", **simulation_results[dist]
        )
        np.savez(f"{output_path}/run_{run_id}/{dist}/z_hat", **z_hat[dist])
        np.savez(f"{output_path}/run_{run_id}/{dist}/losses", **losses[dist])


def _simulate_all_data(
    n: int,
    p: int,
    dists: List[str],
    parameter_functions: Dict[str, Callable[[np.ndarray], np.ndarray]],
    rng: np.random.Generator,
    test_size: float,
):
    """Simulate data from all distributions and parameter functions.

    :param n: The number of observations.
    :param p: The number of covariates.
    :param dists: The distributions to simulate from.
    :param parameter_functions: The parameter functions to use.
    :param rng: The random number generator.
    :param test_size: The test size.
    :return: A dictionary with the simulated data.
    """

    X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, p - 1))])
    simulation_results = {dist: {} for dist in dists}
    for dist in dists:
        distribution = initiate_distribution(distribution=dist)
        simulation_results[dist] = _simulate_data(
            X=X,
            distribution=distribution,
            parameter_function=parameter_functions[dist],
            rng=rng,
            test_size=test_size,
        )
    return simulation_results


def _simulate_data(
    X: np.ndarray,
    distribution: Distribution,
    parameter_function: Callable[[np.ndarray], np.ndarray],
    rng: np.random.Generator,
    test_size: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Simulate data from a distribution and a parameter function.

    :param X: The covariates.
    :param distribution: The distribution to simulate from.
    :param parameter_function: The parameter function to use.
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


def _get_model_predictions(
    simulation_result: Dict[str, np.ndarray],
    models: List[str],
    distribution: Distribution,
    rng: np.random.Generator,
    gbm_parameters: Dict[str, Union[float, int, List[float], List[int]]],
    glm_parameters: Dict[str, Union[float, int, List[float]]],
    verbose: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Get the predictions from the models.

    :param simulation_result: The simulation result.
    :param models: The models to use.
    :param distribution: The distribution.
    :param rng: The random number generator.
    :param gbm_parameters: The GBM parameters.
    :param glm_parameters: The GLM parameters.
    :param verbose: The verbosity level.
    :return: A dictionary with the predictions.
    """

    X_train = simulation_result["train"]["X"]
    X_test = simulation_result["test"]["X"]
    y_train = simulation_result["train"]["y"]
    z_train = simulation_result["train"]["z"]
    z_test = simulation_result["test"]["z"]

    z_hat = {"train": {}, "test": {}}
    z_hat["train"]["true"] = z_train
    z_hat["test"]["true"] = z_test

    for model in models:
        if model == "intercept":
            z_hat_train, z_hat_test = _run_intercept_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
            )
        elif model == "cyc-glm":
            z_hat_train, z_hat_test = _run_cyc_glm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
                parameters=glm_parameters,
            )
        elif model == "uni-gbm":
            z_hat_train, z_hat_test = _run_uni_gbm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
                rng=rng,
                parameters=gbm_parameters,
                verbose=verbose,
            )
        elif model == "cyc-gbm":
            z_hat_train, z_hat_test = _run_cyc_gbm_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                distribution=distribution,
                rng=rng,
                parameters=gbm_parameters,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Model {model} not recognized.")
        z_hat["train"][model] = z_hat_train
        z_hat["test"][model] = z_hat_test

        if verbose > 0:
            logger.info(f"Model {model} done.")

    return z_hat


def _get_model_losses(
    z_hat: Dict[str, Dict[str, np.ndarray]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    distribution: Distribution,
):
    """Calculate the losses for all models.

    :param z_hat: The predicted z values.
    :param y_train: The training outcomes.
    :param y_test: The test outcomes.
    :param distribution: The distribution.
    :return: A dictionary with the losses.
    """
    losses = {"train": {}, "test": {}}
    for data_set in ["train", "test"]:
        for model in z_hat[data_set].keys():
            losses[data_set][model] = distribution.loss(
                z=z_hat[data_set][model], y=y_train if data_set == "train" else y_test
            ).mean()
    return losses


def _get_run_id(output_path: Union[str, None]) -> int:
    if output_path is None:
        return 0
    else:
        previous_run_folders = [
            f
            for f in os.listdir(output_path)
            if os.path.isdir(os.path.join(output_path, f))
        ]
        if len(previous_run_folders) == 0:
            return 0
        previous_run_ids = [
            folder_name.split("_")[1] for folder_name in previous_run_folders
        ]
        return int(max(previous_run_ids)) + 1


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


def _save_output_figures(
    output_path: str,
    run_id: int,
    simulation_results: Dict[str, Dict[str, np.ndarray]],
    z_hat: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    dists: List[str],
    figure_format: str = "png",
):
    """Save the output figures.

    :param output_path: The output path.
    :param run_id: The run id.
    :param simulation_results: The simulation results.
    :param z_hat: The predicted z values.
    :param dists: The distributions.
    :param figure_format: The figure format.
    """
    for dist in dists:
        figure_path = f"{output_path}/run_{run_id}/{dist}/figures"
        os.makedirs(figure_path)
        fig_simulation = _create_simulation_plots(
            z=simulation_results[dist]["train"]["z"],
            y=simulation_results[dist]["train"]["y"],
            distribution=initiate_distribution(distribution=dist),
        )
        fig_results = _create_result_plots(
            z_hat=z_hat[dist]["test"],
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


def _run_cyc_glm_model(
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


def _run_uni_gbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
    rng: np.random.Generator,
    parameters: Dict[str, Union[float, int, List[float], List[int]]],
    verbose: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run univariate GBM model.

    :param X_train: Training covariates.
    :param X_test: Test covariates.
    :param y_train: Training response.
    :param distribution: Distribution object.
    :param rng: Random number generator for the cross-validation.
    :param parameters: Dictionary of parameters.
    :param verbose: Verbosity level.
    :return: Tuple of training and test predictions.
    """

    kappa_max = parameters["kappa_max"]
    if isinstance(kappa_max, list):
        kappa_max = kappa_max[0]
    eps = parameters["eps"]
    max_depth = parameters["max_depth"]
    min_samples_leaf = parameters["min_samples_leaf"]
    n_splits = parameters["n_splits"]

    kappa = tune_kappa(
        X=X_train,
        y=y_train,
        distribution=distribution,
        kappa_max=[kappa_max, 0],
        eps=eps,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_splits=n_splits,
        rng=rng,
        verbose=verbose,
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


def _run_cyc_gbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    distribution: Distribution,
    rng: np.random.Generator,
    parameters: Dict[str, Union[float, int, List[float], List[int]]],
    verbose: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run cyclic GBM model.

    :param X_train: Training covariates.
    :param X_test: Test covariates.
    :param y_train: Training response.
    :param distribution: Distribution object.
    :param rng: Random number generator for the cross-validation.
    :param parameters: Dictionary of parameters.
    :param verbose: Verbosity level.
    :return: Tuple of training and test predictions.
    """
    kappa_max = parameters["kappa_max"]
    eps_gbm = parameters["eps"]
    max_depth = parameters["max_depth"]
    min_samples_leaf = parameters["min_samples_leaf"]
    n_splits = parameters["n_splits"]
    kappa = tune_kappa(
        y=y_train,
        X=X_train,
        kappa_max=kappa_max,
        eps=eps_gbm,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_splits=n_splits,
        rng=rng,
        verbose=verbose,
        distribution=distribution,
    )["kappa"]

    gbm = CycGBM(
        distribution=distribution,
        kappa=kappa,
        eps=eps_gbm,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
    )
    gbm.fit(X=X_train, y=y_train)
    z_hat_train = gbm.predict(X=X_train)
    z_hat_test = gbm.predict(X=X_test)
    return z_hat_train, z_hat_test


if __name__ == "__main__":
    n = 1000
    p = 3
    dists = ["normal"]
    random_seed = 123

    def parameter_function(X):
        z0 = 0 + 0.2 * X[:, 1] + 0.3 * (X[:, 2] > 0)
        z1 = 0.5 + 0.04 * X[:, 2] ** 2 + 0.5 * (X[:, 2] > 0)
        return np.stack([z0, z1])

    parameter_functions = {"normal": parameter_function}

    # GBM hyperparameters
    kappa_max = 1000
    eps_gbm = 0.01
    max_depth = 2
    min_samples_leaf = 10
    n_splits = 2

    # GLM hyperparameters
    max_iter = 1000
    eps_glm = 1e-5
    tol = 0.001

    # Simulation parameters
    output_path = "../../data/results/simulation"
    # models = ["intercept", "cyc-glm", "uni-gbm", "cyc-gbm"]
    models = ["intercept", "uni-gbm"]
    verbose = 1

    simulation_study_results = simulation_study(
        n=n,
        p=p,
        dists=dists,
        models=models,
        parameter_functions=parameter_functions,
        random_seed=random_seed,
        output_path=output_path,
        output_figures=True,
        figure_format="png",
        test_size=0.2,
        glm_parameters={
            "max_iter": max_iter,
            "eps": eps_glm,
            "tol": tol,
        },
        gbm_parameters={
            "kappa_max": kappa_max,
            "eps": eps_gbm,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_splits": n_splits,
        },
        verbose=verbose,
    )
