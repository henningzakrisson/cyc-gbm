import logging

import numpy as np
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.tree import DecisionTreeRegressor

from cyc_gbm.utils.distributions import Distribution, NormalDistribution

logger = logging.getLogger(__name__)


class NGBoostModel:
    """
    Wrapper around NGBRegressor with a Normal distribution and MLE score,
    exposing the same fit/predict interface as the other pipeline models.

    Only NormalDistribution is supported; other distributions raise
    NotImplementedError.

    n_estimators should be pre-tuned via CV (tune_models) and passed in at
    construction time. The final fit uses that fixed count with no early stopping,
    mirroring how GBM/CGBM are fitted after tuning.

    predict() returns z of shape (2, n_samples) where:
        z[0] = mu  (loc)
        z[1] = log(sigma)  (matches NormalDistribution parameterization)
    """

    supports_feature_importance: bool = True

    def __init__(
        self,
        distribution: Distribution,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        max_depth: int = 3,
        random_state: int = 0,
    ):
        if not isinstance(distribution, NormalDistribution):
            raise NotImplementedError(
                f"NGBoost wrapper only supports NormalDistribution, "
                f"got {type(distribution).__name__}"
            )
        self.distribution = distribution
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray | float = 1,
    ) -> None:
        self._model = NGBRegressor(
            Dist=Normal,
            Score=MLE,
            Base=DecisionTreeRegressor(criterion="squared_error", max_depth=self.max_depth),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            verbose=False,
            random_state=self.random_state,
        )
        if isinstance(w, np.ndarray) and not np.all(w == 1):
            self._model.fit(X, y, sample_weight=w)
        else:
            self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        dist = self._model.pred_dist(X)
        mu = dist.loc
        log_sigma = np.log(dist.scale)
        return np.stack([mu, log_sigma])

    def compute_feature_importances(self, j=None):
        """Compute feature importances from the fitted NGBoost model.

        Args:
            j: Parameter dimension index (0 or 1), or ``"all"`` for the
               average across dimensions.  If ``None``, returns the raw
               ``(n_params, n_features)`` array from NGBRegressor.

        Returns:
            1-d array of feature importances.
        """
        raw = self._model.feature_importances_
        if j is None:
            return raw
        if j == "all":
            return raw.mean(axis=0)
        return raw[j]
