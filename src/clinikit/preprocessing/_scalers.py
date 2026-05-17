"""Scaler transformers — thin wrappers around scikit-learn implementations.

Each class subclasses the corresponding ``sklearn.preprocessing``
transformer without behavioural changes. They exist in
``clinikit.preprocessing`` so users get a single, stable import path
that survives across sklearn versions and ships under our type
contract.

All transformers inherit ``set_output(transform="pandas"|"default")``
from sklearn's :class:`TransformerMixin`, so callers can opt into
pandas-DataFrame output:

    scaler = StandardScaler().set_output(transform="pandas")
"""

from __future__ import annotations

from sklearn.preprocessing import (
    MinMaxScaler as _SkMinMaxScaler,
)
from sklearn.preprocessing import (
    PowerTransformer as _SkPowerTransformer,
)
from sklearn.preprocessing import (
    QuantileTransformer as _SkQuantileTransformer,
)
from sklearn.preprocessing import (
    RobustScaler as _SkRobustScaler,
)
from sklearn.preprocessing import (
    StandardScaler as _SkStandardScaler,
)

__all__ = [
    "MinMaxScaler",
    "QuantileScaler",
    "RobustScaler",
    "StandardScaler",
    "YeoJohnsonScaler",
]


class StandardScaler(_SkStandardScaler):
    """Standardise features by removing the mean and scaling to unit variance.

    Behaves identically to :class:`sklearn.preprocessing.StandardScaler`.
    See the upstream documentation for full parameter and attribute
    details.
    """


class RobustScaler(_SkRobustScaler):
    """Scale features using statistics that are robust to outliers.

    Behaves identically to :class:`sklearn.preprocessing.RobustScaler`.
    """


class MinMaxScaler(_SkMinMaxScaler):
    """Scale features to a given range, default ``[0, 1]``.

    Behaves identically to :class:`sklearn.preprocessing.MinMaxScaler`.
    """


class QuantileScaler(_SkQuantileTransformer):
    """Map features to a uniform or normal distribution via quantiles.

    Thin wrapper around :class:`sklearn.preprocessing.QuantileTransformer`.
    Defaults to a normal output distribution to suit downstream
    parametric models common in this toolkit; pass
    ``output_distribution="uniform"`` to restore sklearn's default.
    """

    def __init__(
        self,
        *,
        n_quantiles: int = 1000,
        output_distribution: str = "normal",
        ignore_implicit_zeros: bool = False,
        subsample: int = 10_000,
        random_state: int | None = None,
        copy: bool = True,
    ) -> None:
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            ignore_implicit_zeros=ignore_implicit_zeros,
            subsample=subsample,
            random_state=random_state,
            copy=copy,
        )


class YeoJohnsonScaler(_SkPowerTransformer):
    """Apply the Yeo-Johnson power transform to make features Gaussian-like.

    Thin wrapper around
    :class:`sklearn.preprocessing.PowerTransformer` pinned to
    ``method="yeo-johnson"`` — the only Power-Transformer variant that
    handles both positive and non-positive values.
    """

    def __init__(self, *, standardize: bool = True, copy: bool = True) -> None:
        super().__init__(method="yeo-johnson", standardize=standardize, copy=copy)
