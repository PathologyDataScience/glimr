from ray.tune.search import sample
import os
import pandas as pd
import numpy as np
import ray
from inspect import isfunction


def prune_search(space):
    """Prune constants and functions from a search space for pbt mutation.

    This function recurses through a nested dictionary defining a search space, removing
    any constant values and conditional functions (defined through tune.sample_from) from
    the space. Constant values and are defined as non-container types, non-callable objects,
    and non-ray.tune.search.sample.Domain or -Sampler objects. This allows the search
    space to be used as a the `hyperparam_mutations` argument for the
    ray.tune.schedulers.PopulationBasedTraining scheduler.

    Parameters
    ----------
    space : dict
        A configuration dictionary defining a search space.

    Returns
    -------
    pruned_space : dict
        A pruned space where constants and conditional search hyperparameters
        have been removed.
    """

    def _is_constant(value):
        return not (
            isinstance(value, (list, tuple, dict))
            or callable(value)
            or hasattr(value, "__call__")
            or isinstance(value, (sample.Domain, sample.Sampler))
        )

    def _is_function(value):
        return isinstance(value, ray.tune.search.sample.Function)

    pruned_space = {}
    for key, value in space.items():
        if isinstance(value, dict):
            pruned_value = prune_search(value)
            if pruned_value:  # don't add empty dicts
                pruned_space[key] = pruned_value
        elif not (_is_constant(value) or _is_function(value)):
            pruned_space[key] = value
    return pruned_space


def sample_space(space):
    """Sample a configuration from a Ray Tune search space.

    This function recurses through a nested dictionary defining a search space,
    sampling a value when a search space hyperparameter is encountered.

    Parameters
    ----------
    space : dict
        A nested dictionary containing fixed values and ray tune search space
        API objects.

    Returns
    -------
    config : dict
        A configuration where specific valuels have been sampleld from the tunable
        hyperparameters.
    """

    if isinstance(space, dict):
        config = {}
        for key in space:
            config[key] = sample_space(space[key])
        return config
    elif isinstance(space, (sample.Categorical, sample.Integer, sample.Float)):
        return sample_space(space.sample())
    else:  # non sampleable or nestable value
        return space
