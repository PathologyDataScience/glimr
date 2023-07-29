from ray.tune.search import sample


def prune_constants(space):
    """Prepares a search space for ray tune's PBT scheduler by pruning constants.

    This function recurses through a nested dictionary defining a search space, removing
    any constant values from the space, where constant values are defined as non-container
    types, non-callable objects, and non-ray.tune.search.sample.Domain or -Sampler objects.
    In this way, the returned search space is ready for use with ray tune's PBT scheduler.

    Parameters
    ----------
    space : dict
        A configuration dictionary defining a hyperparameter or model search space.

    Returns
    -------
    pruned_space : dict
        A pruned dictionary without any constants (defined as non-container types,
        non-callable objects, and non-ray.tune.search.sample.Domain or -Sampler objects) ready
        for use with ray tune's PBT scheduler.
    """

    def _is_constant(value):
        return not (
            isinstance(value, (list, tuple, dict))
            or callable(value)
            or hasattr(value, "__call__")
            or isinstance(value, (sample.Domain, sample.Sampler))
        )

    pruned_space = {}
    for key, value in space.items():
        if isinstance(value, dict):
            pruned_value = prune_constants(value)
            if pruned_value:  # don't add empty dicts
                pruned_space[key] = pruned_value
        elif not _is_constant(value):
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
