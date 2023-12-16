"""
This package contains hyperparameter tuning tools
"""

# make functions available at the package level using shadow imports
from glimr.analysis import (
    default_checkpoints,
    experiment_table,
    top_k_configs,
    top_k_trials,
)
from glimr.keras import keras_losses, keras_metrics, keras_optimizer
from glimr.optimization import optimization_space
from glimr.search import Search
from glimr.utils import prune_search, sample_space

# list out things that are available for public use
__all__ = (
    # functions and classes of this package
    "default_checkpoints",
    "experiment_table",
    "keras_losses",
    "keras_metrics",
    "keras_optimizer",
    "optimization_space",
    "prune_search",
    "sample_space",
    "Search",
    "top_k_trials",
    "top_k_configs",
)
