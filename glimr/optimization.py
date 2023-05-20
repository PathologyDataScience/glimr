from ray import tune


# TODO: add support for other options, e.g., epsilon, centered, nesterov, initial_accumulator_value
# perhaps figure out conditional tuning
def optimization_space(
    epochs=100,
    method=tune.choice(["rms", "sgd", "adadelta", "adagrad", "adam"]),
    learning_rate=tune.quniform(1e-5, 1e-2, 1e-5),
    rho=tune.quniform(0.5, 1.0, 1e-2),
    momentum=tune.quniform(0.0, 1e-1, 1e-2),
    beta_1=tune.quniform(0.5, 1.0, 1e-2),
    beta_2=tune.quniform(0.5, 1.0, 1e-2),
    moving_average=tune.choice([True, False]),
):
    """Generate a search space for a training and optimization parameters.

    This function creates a search space for gradient optimization parameters
    including gradient optimization method, and method hyperparameters like
    learning rate or momentum. The search space for each hyperparameter is
    defined using the ray tune search space API.

    Parameters
    ----------
    epochs : int
        The maximum number of epochs to train a model for. Can be overrided
        by the scheduler or  the `stopper` attribute of a `Search` object.
    method : str, ray.tune.search.sample.Categorical
        A string or categorical search parameter defining the gradient
        optimization method. Strings are converted to tf.keras.optimizer objects
        during training by search.utils.keras_optimizer. Default value is
        tune.choice(["rms", "sgd", "adadelta", "adagrad", "adam"]).
    learning_rate : float, ray.tune.search.sample.Float
        The range of learning rates for the gradient optimizer. Default value
        is tune.quniform(1e-5, 1e-2, 1e-5).
    rho : list[float]
        The range of learning rate decay values for the optimizers. Default value
        is tune.quniform(0.5, 1.0, 1e-2).
    momentum : float, ray.tune.search.sample.Float
        The range of momentum values for the momentum optimizers. Default value
        is tune.quniform(0.0, 1e-1, 1e-2).
    beta_1 : float, ray.tune.search.sample.Float
        The range of beta_1 values for the adam optimizer. Default value is
        tune.quniform(0.5, 1.0, 1e-2).
    beta_2 : float, ray.tune.search.sample.Float
        The range of beta_2 values for the adam optimizer. Default value is
        tune.quniform(0.5, 1.0, 1e-2).
    moving_average : bool, ray.tune.search.sample.Categorical
        A bool indicating if the optimizer should take a moving average of
        the model throughout training. Default value is tune.choice([True, False])

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
        "moving_average": moving_average,
    }

    return space
