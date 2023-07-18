from functools import partial
import inspect
import tensorflow as tf


def keras_losses(config):
    """Builds loss and loss weight dictionaries for tf.keras.Model.compile.

    This creates loss class instances given class definitions and kwargs, and 
    formats loss names appropriately for keras.

    Parameters
    ----------
    config : dict
        A configuration containing tasks and their losses. Each task in config
        should have a "loss" key that links to a single dictionary defining the
        loss name, loss class or callable, and a dictionary of loss kwargs. 
        kwargs can inclulde tunable hyperparameters defined using the ray tune 
        search space API.

    Returns
    -------
    losses : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    weights : dict
        A dict of model output name : float key value pairs for use with
        tf.keras.Model.compile.
    """

    # check if kwargs in loss dictionary
    if "kwargs" in config["tasks"][task]["loss"]:
        kwargs = config["tasks"][task]["loss"]["kwargs"]
    else:
        kwargs = {}

    # create loss dictionary
    losses = {}
    for task in config["tasks"]:        
        if inspect.isfunction(config["tasks"][task]["loss"]["loss"]):
            loss = partial(config["tasks"][task]["loss"]["loss"], **kwargs)
        elif inspect.isclass(config["tasks"][task]["loss"]["loss"]):
            loss = config["tasks"][task]["loss"]["loss"](**kwargs)
        else:
            raise ValueError(
                "task 'loss' must be a function or class."
            )

    # create loss weight dictionary
    weights = {
        name: config["tasks"][name]["loss_weight"] for name in config["tasks"]
    }

    return losses, weights


def keras_metrics(config):
    """Builds metric dictionaries for tf.keras.Model.compile.

    This creates metric class instances given class definitions and kwargs, and 
    formats metric names appropriately for keras.

    Parameters
    ----------
    config : dict
        A configuration containing tasks and their metrics. Each task in config
        should have a "metrics" key that links to a list of dictionaries with
        each dictionary defining a metric name, metric class, and a dictionary
        of metric kwargs.

    Returns
    -------
    metrics : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    """

    # create a metric dictionary from the config
    metrics = {}
    for task in config["tasks"]:

        # get metric list for task
        task_metrics = config["tasks"][task]["metrics"]

        # get metric classes
        classes = [metric["metric"] for metric in task_metrics]

        # get metric names - user-defined
        names = [metric["name"] for metric in task_metrics]

        # get metric kwargs
        kwargs = [
            metric["kwargs"] if "kwargs" in metric else {}
            for metric in task_metrics
        ]

        # wrap metrics in a list if more than 1
        objects = [classes(name=n, **k) for n, k in zip(classes, names, kwargs)]

        # assign to metrics dict
        if len(names) > 1:
            metrics[task] = objects
        else:
            metrics[task] = objects[0]

    return metrics


def keras_optimizer(config):
    """Convert an optimizaiton configuration to a tf.keras.optimizers.Optimizer.

    Parameters
    ----------
    config : dict
        A configuration dictionary defining the optimization "method",
        and parameters including "learning_rate", "momentum", and other
        hyperparameters.

    Returns
    -------
    optimizer : tf.keras.optimizers.Optimizer
        An optimizer for keras model compilation.
    """

    def extract_args(config, kws):
        return {kw: config[kw] for kw in kws if kw in config.keys()}

    if config["method"] == "rms":
        kws = [
            "learning_rate",
            "rho",
            "momentum",
            "epsilon",
            "centered",
            "use_ema",
            "ema_momentum",
            "ema_overwrite_frequency",
        ]
        return tf.keras.optimizers.experimental.RMSprop(**extract_args(config, kws))
    elif config["method"] == "sgd":
        kws = [
            "learning_rate",
            "momentum",
            "nesterov",
            "use_ema",
            "ema_momentum",
            "ema_overwrite_frequency",
        ]
        return tf.keras.optimizers.experimental.SGD(**extract_args(config, kws))
    elif config["method"] == "adadelta":
        kws = [
            "learning_rate",
            "rho",
            "epsilon",
            "use_ema",
            "ema_momentum",
            "ema_overwrite_frequency",
        ]
        return tf.keras.optimizers.experimental.Adadelta(**extract_args(config, kws))
    elif config["method"] == "adagrad":
        kws = [
            "learning_rate",
            "initial_accumulator_value",
            "epsilon",
            "use_ema",
            "ema_momentum",
            "ema_overwrite_frequency",
        ]
        return tf.keras.optimizers.experimental.Adagrad(**extract_args(config, kws))
    elif config["method"] == "adam":
        kws = [
            "learning_rate",
            "beta_1",
            "beta_2",
            "epsilon",
            "amsgrad",
            "use_ema",
            "ema_momentum",
            "ema_overwrite_frequency",
        ]
        return tf.keras.optimizers.Adam(**extract_args(config, kws))
    else:
        raise ValueError(
            "config['method'] must be one of 'adadelta', 'adam', 'adagram', 'rms', or 'sgd'"
        )
