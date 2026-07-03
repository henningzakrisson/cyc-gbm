import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder

from cyc_gbm.utils.distributions import Distribution


class CyclicGeneralizedLinearModel:

    supports_feature_importance: bool = False

    def __init__(
        self,
        distribution: Distribution,
        max_iter: int = 1000,
        tol: float = 1e-5,
        eps: list[float] | float = 1e-7,
    ) -> None:
        """
        Initialize the model.
        """
        self.distribution = distribution
        self.d = self.distribution.n_dim
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps
        self.beta = None
        self.z0 = None
        self._encoder: OneHotEncoder | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray, w: np.ndarray) -> None:
        """
        Fit the model.
        """
        if isinstance(X, pd.DataFrame):
            self._encoder = self._fit_encoder(X)
            X = self._encode(X)
        n = len(y)
        z = np.zeros((self.d, n))
        self.z0 = minimize(
            fun=lambda z0: self.distribution.loss(y=y, z=z0[:, None] + z, w=w).sum(),
            x0=self.distribution.mme(y=y, w=w),
        )["x"]
        z = np.tile(self.z0, (X.shape[0], 1)).T
        p = X.shape[1]

        beta = np.zeros((self.max_iter, self.d, p))

        for i in range(self.max_iter):
            for j in range(self.d):
                g = self.distribution.grad(y=y, z=z, w=w, j=j)
                # Update parameter estimate; divide by n so step size is data-size invariant
                beta[i, j] = beta[i - 1, j] - (self.eps / n) * g @ X
                # Update parameter estimate
                z[j] = self.z0[j] + beta[i, j] @ X.T

            # Check convergence
            if i > 0 and np.linalg.norm(beta[i] - beta[i - 1]) < self.tol:
                break

            if i == self.max_iter - 1:
                Warning("CGLM model did not converge")

        self.beta = beta[i]

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict the response for the given input data.
        """
        if isinstance(X, pd.DataFrame):
            X = self._encode(X)
        z = np.zeros((self.d, X.shape[0]))
        for j in range(self.d):
            z[j] = self.z0[j] + self.beta[j] @ X.T
        return z

    @staticmethod
    def _fit_encoder(X: pd.DataFrame) -> OneHotEncoder | None:
        """Fit a one-hot encoder for categorical columns in *X*."""
        cat_cols = [
            c for c in X.columns
            if isinstance(X[c].dtype, pd.CategoricalDtype)
        ]
        if not cat_cols:
            return None
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(X[cat_cols])
        encoder._cat_cols = cat_cols  # type: ignore[attr-defined]
        return encoder

    def _encode(self, X: pd.DataFrame) -> np.ndarray:
        """Convert a DataFrame to a numeric numpy array.

        Categorical columns are one-hot encoded; numeric columns are
        passed through.
        """
        if self._encoder is None:
            return X.to_numpy(dtype=float, na_value=0.0)
        cat_cols = self._encoder._cat_cols  # type: ignore[attr-defined]
        num_cols = [c for c in X.columns if c not in cat_cols]
        encoded = self._encoder.transform(X[cat_cols])
        numeric = X[num_cols].to_numpy(dtype=float, na_value=0.0)
        return np.hstack([numeric, encoded])
