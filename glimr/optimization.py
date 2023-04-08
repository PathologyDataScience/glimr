from ray import tune


def optimization_space(
    epochs=100,
    method=tune.choice(["rms", "sgd", "adadelta", "adagrad", "adam"]),
    learning_rate=tune.quniform(1e-5, 1e-2, 1e-5),
    rho=tune.quniform(0.5, 1.0, 1e-2),
    momentum=tune.quniform(0.0, 1e-1, 1e-2),
    beta_1=tune.quniform(0.5, 1.0, 1e-2),
    beta_2=tune.quniform(0.5, 1.0, 1e-2),
):
    """Generate a search space for a training and optimization parameters.
    This function is used to generate a search space for batching and gradient optimization
    parameters. This includes batch size, gradient optimization method, and method
    parameters like learning rate. The search space for each parameter
    is defined as: 1. A set defining specific values to search 2. A list specifying a search
    range [min, max, increment(optional)] 3. A specific value to assign to hyperparameter
    (no search).
    Parameters
    ----------
    epochs : int
        The maximum number of epochs to train a model for. This can be overrided by the
        scheduler or by the `stopper` attribute of a `Search` object.
    method : set[string]
        A list of strings encoding the gradient optimization method. The strings in `method`
        are converted to tf.keras.optimizer objects during training by
        search.utils.keras_optimizer. Default value is {"rms", "sgd", "adadelta", "adagrad",
        "adam"}.
    learning_rate : list[float]
        The range of learning rates for the gradient optimizer. Default value
        is [1e-5, 1e-2, 1e-5].
    rho : list[float]
        The range of learning rate decay values for the optimizers. Default value
        is [0.5, 1.0, 1e-2].
    momentum : list[float]
        The range of momentum values for the momentum optimizers. Default value
        is [0.0, 1e-1, 1e-2].
    beta_1 : list[float]
        The range of beta_1 values for the adam optimizer. Default value is
        [0.5, 1.0, 1e-2].
    beta_2 : list[float]
        The range of beta_2 values for the adam optimizer. Default value is
        [0.5, 1.0, 1e-2].
    Returns
    -------
    task : dict
        A ray optimization search space.
    """

    # convert to ray search space
    space = {
        "epochs": epochs,
        "method": method,
        "learning_rate": learning_rate,
        "rho": rho,
        "momentum": momentum,
        "beta_1": beta_1,
        "beta_2": beta_2,
    }

    return space
