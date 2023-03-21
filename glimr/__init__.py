"""
This package contains hyperparameter tuning tools
"""

# make functions available at the package level using shadow imports
from glimr.optimization import optimization_space
from glimr.search import Search
from glimr.utils import (
    check_tunable,
    keras_losses,
    keras_metrics,
    keras_optimizer,
    set_hyperparameter,
    string_to_loss,
    string_to_metric,
)

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "check_tunable",
    "keras_losses",
    "keras_metrics",
    "keras_optimizer",
    "optimization_space",
    "set_hyperparameter",
    "string_to_loss",
    "string_to_metric",
    "Search",
)
