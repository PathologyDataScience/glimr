"""
This package contains hyperparameter tuning tools
"""

# make functions available at the package level using shadow imports
from glimr.keras import keras_losses, keras_metrics, keras_optimizer
from glimr.optimization import optimization_space
from glimr.search import Search
from glimr.utils import sample_space, prune_constants

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "keras_losses",
    "keras_metrics",
    "keras_optimizer",
    "optimization_space",
    "sample_space",
    "prune_constants",
    "Search",
)
