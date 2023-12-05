from ray import tune
import numpy as np


def optimization_space(
    epochs=100,
    method=tune.choice(["rms", "sgd", "adadelta", "adagrad", "adam"]),
    learning_rate=tune.quniform(1e-5, 1e-2, 1e-5),
    rho=tune.quniform(0.5, 1.0, 1e-2),
    momentum=tune.quniform(0.0, 1e-1, 1e-2),
    beta_1=tune.quniform(0.5, 1.0, 1e-2),
    beta_2=tune.quniform(0.5, 1.0, 1e-2),
    use_ema=tune.choice([True, False]),
    ema_momentum=tune.quniform(0.9, 0.99, 1e-2),
    ema_overwrite_frequency=tune.choice([None, 1, 2, 3, 4, 5]),
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
    use_ema : bool, ray.tune.search.sample.Categorical
        Whether to use moving averaging of weights during training. Default value
        is tune.choice([True, False]).
    ema_momentum : float, ray.tune.search.sample.Float
        How much to weight the average from prior iterations when calculating
        updated model weights. Higher values will weight the prior iterations
        more. Default value is tune.quniform(0.9, 0.99, 1e-2).
    ema_overwrite_frequency : int, ray.tune.search.sample.Categorical
        How often to overwrite the model weights with the calculated average.
        A value of None will overwrite once per epoch. Integer values will
        overwrite after the specified number of batches. Default value is
        tune.choice([None, 1, 2, 3, 4, 5]).

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
        "use_ema": use_ema,
        "ema_momentum": ema_momentum,
        "ema_overwrite_frequency": ema_overwrite_frequency,
    }

    return space


def data_space(batch_size=1, cv_folds=None, **kwargs):
    space = {}
    space["batch_size"] = batch_size

    for arg, val in zip(kwargs, kwargs.values()):
        space[arg] = val

    if cv_folds is not None:
        space["cv_fold_index"] = tune.grid_search(np.arange(cv_folds))
        space["cv_folds"] = cv_folds
    return space
