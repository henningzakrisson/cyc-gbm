from typing import Union, List, Dict, Tuple, Optional
from joblib import Parallel, delayed

import numpy as np

from cyc_gbm import CyclicalGradientBooster
from cyc_gbm.utils.logger import CycGBMLogger


def tune_n_estimators(
    X: np.ndarray,
    y: np.ndarray,
    w: Union[np.ndarray, float] = 1.0,
    model: CyclicalGradientBooster = CyclicalGradientBooster(),
    n_estimators_max: Union[int, List[int]] = 1000,
    n_splits: int = 4,
    rng: Optional[np.random.Generator] = None,
    random_state: Optional[int] = None,
    logger: Optional[CycGBMLogger] = None,
    parallel: bool = True,
    n_jobs: int = -1,
) -> Dict[str, Union[List[int], np.ndarray]]:
    """Finds a suitable n_estimators hyperparameter of a CycGBM model using k-fold cross-validation.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param y: The target vector of shape (n_samples,).
    :param w: The weights for the training data, of shape (n_samples,). Default is 1 for all samples.
    :param model: The CycGBM model to use for the tuning. Default is a CycGBM model with default parameters.
    :param n_estimators_max: The maximum value of the n_estimators parameter to test. Dimension-wise or same for all parameter dimensions.
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param rng: The random number generator.
    :param random_state: The random state to use for the k-fold split. Will be ignored if rng is not None.
    :param logger: The simulation logger to use for logging.
    :param parallel: Whether to use parallel processing for the cross-validation.
    :param n_jobs: The number of jobs to use for parallel processing. Default is -1, which uses all available cores.
    :return: A dictionary containing the following keys:
        - "n_estimators": The optimal n_estimators parameter value for each parameter dimension.
        - "loss": The loss results for every boosting step in the cross-validation.
    """
    logger = CycGBMLogger() if logger is None else logger
    rng = np.random.default_rng(random_state) if rng is None else rng
    model.reset(n_estimators=0)

    n_estimators_max = (
        n_estimators_max
        if isinstance(n_estimators_max, list)
        else [n_estimators_max] * model.n_dim
    )

    folds = _fold_split(X=X, y=y, w=w, n_splits=n_splits, rng=rng)

    logger.log(f"performing cross-validation on {n_splits} folds")
    if parallel:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_fold)(
                fold=folds[i],
                model=model,
                n_estimators_max=n_estimators_max,
            )
            for i in folds
        )
    else:
        results = []
        for i in folds:
            logger.log(f"fold {i+1}/{n_splits}")
            results.append(
                _evaluate_fold(
                    fold=folds[i],
                    model=model,
                    n_estimators_max=n_estimators_max,
                )
            )

    loss = {
        "train": [result[0] for result in results],
        "valid": [result[1] for result in results],
    }

    n_estimators = _find_n_estimators(
        loss=np.sum(loss["valid"], axis=0),
        n_estimators_max=n_estimators_max,
        logger=logger,
    )

    return {
        "n_estimators": n_estimators,
        "loss": loss,
    }


def _fold_split(
    X: np.ndarray,
    y: np.ndarray,
    w: Union[float, np.ndarray],
    n_splits: int,
    rng: np.random.Generator,
) -> Dict[
    int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """Split data into k folds.

    :param X: The input data matrix of shape (n_samples, n_features).
    :param n_splits: The number of folds to use for k-fold cross-validation.
    :param rng: The random number generator.
    :return A dictionary containing the folds as tuples in the order
        (X_train, y_train, w_train, X_valid, y_valid, w_valid).
    """
    if isinstance(w, float):
        w = np.ones(len(y)) * w
    idx = rng.permutation(X.shape[0])
    idx_folds = np.array_split(idx, n_splits)
    folds = {}
    for i in range(n_splits):
        idx_test = idx_folds[i]
        idx_train = np.concatenate(idx_folds[:i] + idx_folds[i + 1 :])
        folds[i] = (
            X[idx_train],
            y[idx_train],
            w[idx_train],
            X[idx_test],
            y[idx_test],
            w[idx_test],
        )
    return folds


def _evaluate_fold(
    fold: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    model: CyclicalGradientBooster,
    n_estimators_max: List[int],
):
    X_train, y_train, w_train, X_valid, y_valid, w_valid = fold

    model.fit(X=X_train, y=y_train, w=w_train)
    z_train = model.predict(X=X_train)
    z_valid = model.predict(X=X_valid)

    loss_train = np.zeros((max(n_estimators_max) + 1, model.n_dim))
    loss_valid = np.zeros((max(n_estimators_max) + 1, model.n_dim))
    loss_train[0, :] = model.distribution.loss(y=y_train, z=z_train, w=w_train).sum()
    loss_valid[0, :] = model.distribution.loss(y=y_valid, z=z_valid, w=w_valid).sum()

    for k in range(1, max(n_estimators_max) + 1):
        for j in range(model.n_dim):
            if k < n_estimators_max[j]:
                model.add_tree(X=X_train, y=y_train, z=z_train, w=w_train, j=j)

                z_train[j] += model.learning_rate[j] * model.trees[j][-1].predict(
                    X_train
                )
                z_valid[j] += model.learning_rate[j] * model.trees[j][-1].predict(
                    X_valid
                )

                loss_train[k, j] = model.distribution.loss(
                    y=y_train, z=z_train, w=w_train
                ).sum()
                loss_valid[k, j] = model.distribution.loss(
                    y=y_valid, z=z_valid, w=w_valid
                ).sum()
            else:
                if j == 0:
                    loss_train[k, j] = loss_train[k - 1, -1]
                    loss_valid[k, j] = loss_valid[k - 1, -1]
                else:
                    loss_train[k, j] = loss_train[k, j - 1]
                    loss_valid[k, j] = loss_valid[k, j - 1]
        if _has_tuning_converged(
            current_loss=loss_valid[k], previous_loss=loss_valid[k - 1]
        ):
            loss_train[k + 1 :, :] = loss_train[k, -1]
            loss_valid[k + 1 :, :] = loss_valid[k, -1]
            break

    return loss_train, loss_valid


def _has_tuning_converged(
    current_loss: np.ndarray,
    previous_loss: np.ndarray,
) -> bool:
    """Check if the tuning has converged after a complete boosting iteration.

    :param current_loss: The current loss for all parameter dimensions.
    :param previous_loss: The previous loss for all parameter dimensions.
    :return: True if the tuning has converged, False otherwise.
    """
    shifted_current_loss = np.roll(current_loss, shift=1)
    shifted_current_loss[0] = previous_loss[-1]
    return np.all(current_loss >= shifted_current_loss)


def _find_n_estimators(
    loss: np.ndarray,
    n_estimators_max: Union[int, List[int]],
    logger: CycGBMLogger,
) -> List[int]:
    loss_delta = np.zeros_like(loss)
    loss_delta[1:, 0] = loss[1:, 0] - loss[:-1, -1]
    loss_delta[1:, 1:] = loss[1:, 1:] - loss[1:, :-1]
    n_estimators = np.maximum(0, np.argmax(loss_delta > 0, axis=0) - 1)
    did_not_converge = (loss_delta > 0).sum(axis=0) == 0
    n_estimators[did_not_converge] = np.array(n_estimators_max)[did_not_converge]
    if np.any(did_not_converge):
        logger.log(
            f"tuning did not converge for dimensions {np.where(did_not_converge)}",
            verbose=1,
        )
    return list(n_estimators)
