from ray.tune.search import sample


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
