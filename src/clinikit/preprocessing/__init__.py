"""clinikit.preprocessing — imputers, scalers, and indicator transformers.

Scalers
-------
- :class:`StandardScaler`
- :class:`RobustScaler`
- :class:`MinMaxScaler`
- :class:`QuantileScaler`
- :class:`YeoJohnsonScaler`

Imputers
--------
- :class:`MedianImputer`
- :class:`KNNImputer`
- :class:`MICEImputer`
- :class:`MissForestImputer`
- :class:`DomainAwareImputer`

Indicators
----------
- :class:`MissingIndicator`
- :class:`OutlierFlag`

All transformers inherit from :class:`sklearn.base.TransformerMixin`
and follow the standard ``fit`` / ``transform`` / ``fit_transform``
API. They are usable inside ``sklearn.pipeline.Pipeline``,
``ColumnTransformer``, and ``FeatureUnion``.
"""

from __future__ import annotations

from clinikit.preprocessing._imputers import (
    DomainAwareImputer,
    KNNImputer,
    MedianImputer,
    MICEImputer,
    MissForestImputer,
)
from clinikit.preprocessing._indicators import (
    MissingIndicator,
    OutlierFlag,
)
from clinikit.preprocessing._scalers import (
    MinMaxScaler,
    QuantileScaler,
    RobustScaler,
    StandardScaler,
    YeoJohnsonScaler,
)

__all__ = [
    "DomainAwareImputer",
    "KNNImputer",
    "MICEImputer",
    "MedianImputer",
    "MinMaxScaler",
    "MissForestImputer",
    "MissingIndicator",
    "OutlierFlag",
    "QuantileScaler",
    "RobustScaler",
    "StandardScaler",
    "YeoJohnsonScaler",
]
