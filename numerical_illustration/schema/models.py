"""Model configuration models (one per model type)."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .constants import ModelClass


class GradientBoostingMachineConfig(BaseModel):
    """Hyperparameters for the (non-cyclical) gradient boosting machine.

    Uses ``CyclicalGradientBooster`` under the hood with trees only on the
    first parameter dimension.

    Attributes:
        model_class: Discriminator literal, always ``"gbm"``.
        n_estimators: Maximum number of boosting iterations.
        max_depth: Maximum depth of each decision tree.
        learning_rate: Shrinkage applied to each tree.
    """

    model_class: Literal[ModelClass.GBM]
    n_estimators: int = 600
    max_depth: int = 3
    learning_rate: float = 0.05

    def n_estimators_as_list(self, n_dim: int) -> list[int]:
        """Return n_estimators as a per-dimension list for CyclicalGradientBooster.

        GBM only boosts the first parameter dimension; remaining dimensions
        get zero estimators.
        """
        return [self.n_estimators] + [0] * (n_dim - 1)


class CyclicalGradientBoostingMachineConfig(BaseModel):
    """Hyperparameters for the cyclical gradient boosting machine.

    Trees are fitted for every parameter dimension of the distribution in
    a cyclic fashion.

    Attributes:
        model_class: Discriminator literal, always ``"cgbm"``.
        n_estimators: Maximum boosting iterations per parameter dimension.
        max_depth: Maximum depth of each decision tree.
        learning_rate: Shrinkage applied to each tree (scalar or per-dimension).
    """

    model_class: Literal[ModelClass.CGBM]
    n_estimators: list[int] = Field(default_factory=lambda: [600, 600])
    max_depth: int = 3
    learning_rate: float | list[float] = 0.05


class NaturalGradientBoostingMachineConfig(BaseModel):
    """Hyperparameters for the natural gradient boosting machine (NGBoost).

    Only supports ``NormalDistribution``.

    Attributes:
        model_class: Discriminator literal, always ``"ngboost"``.
        n_estimators: Maximum number of boosting iterations.
        max_depth: Maximum depth of the base ``DecisionTreeRegressor``.
        learning_rate: Shrinkage applied to each boosting step.
    """

    model_class: Literal[ModelClass.NGBOOST]
    n_estimators: int = 600
    max_depth: int = 3
    learning_rate: float = 0.05


class CyclicalGeneralizedLinearModelConfig(BaseModel):
    """Hyperparameters for the cyclical generalized linear model.

    Attributes:
        model_class: Discriminator literal, always ``"cglm"``.
        max_iter: Maximum number of coordinate-descent iterations.
        tolerance: Convergence tolerance on the coefficient norm change.
        step_size: Learning rate for the gradient step (scaled by 1/n internally).
    """

    model_class: Literal[ModelClass.CGLM]
    max_iter: int = 2000
    tolerance: float = 1e-5
    step_size: float = 0.1


class InterceptConfig(BaseModel):
    """Configuration for the intercept-only baseline model.

    Fits a constant per distribution parameter via MLE.

    Attributes:
        model_class: Discriminator literal, always ``"intercept"``.
    """

    model_class: Literal[ModelClass.INTERCEPT]


ModelConfig = Annotated[
    GradientBoostingMachineConfig
    | CyclicalGradientBoostingMachineConfig
    | NaturalGradientBoostingMachineConfig
    | CyclicalGeneralizedLinearModelConfig
    | InterceptConfig,
    Field(discriminator="model_class"),
]
