import tensorflow as tf
import types


def keras_losses(config, mapper):
    """A utility to build loss and loss weight dictionaries for
    tf.keras.Model.compile.

    Given a configuration where task losses are defined as strings, convert the
    strings to loss objects and format a dict for model compilation.

    Parameters
    ----------
    config : dict
        A configuration containing tasks and their losses. Each task in config
        should have a "loss" key that links to a loss dictionary. The loss
        dictionary contains a "name" that indexes a tf.keras.losses.Loss object
        or callable in `mapper`, and an optional "kwargs" dictionary defining
        keyword arguments for creating the loss. kwargs can inclulde tunable
        hyperparameters defined using the ray tune search space API.
    mapper : dict
        A dict mapping metric names to tf.keras.losses.Loss of function objects.

    Returns
    -------
    losses : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    weights : dict
        A dict of model output name : float key value pairs for use with
        tf.keras.Model.compile.
    """

    # create loss dictionary
    losses = {}
    for task in config["tasks"]:
        fn = mapper[config["tasks"][task]["loss"]["name"]]
        if isinstance(fn, types.FunctionType):
            losses[task] = fn
        else:
            if "kwargs" in config["tasks"][task]["loss"]:
                kwargs = config["tasks"][task]["loss"]["kwargs"]
            else:
                kwargs = {}
            losses[task] = fn(**kwargs)

    # create loss weight dictionary
    weights = {name: config["tasks"][name]["loss_weight"] for name in config["tasks"]}

    return losses, weights


def keras_metrics(config, mapper):
    """A utility to build metric dictionaries for tf.keras.Model.compile.

    Given a configuration where task metrics are defined as strings, convert
    the strings to tf.keras.metrics.Metric objects and format a dict for model
    compilation.

    Parameters
    ----------
    config : dict
        A configuration defining the metrics for each task. Each task in config
        should have a "metrics" key that links to a metrics dictionary. The
        metrics dictionary contains a "name" that indexes a tf.keras.metrics.Metric
        object in `mapper`, and an optional "kwargs" dictionary defining keyword
        arguments for creating the metric.
    mapper : dict
        A dict mapping metric names to tf.keras.metrics.Metric objects.

    Returns
    -------
    metrics : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    """

    # create a metric dictionary from the config
    metrics = {}
    for task in config["tasks"]:
        # get task metric dictionary
        task_metrics = config["tasks"][task]["metrics"]

        # get metric names for task
        names = [task_metrics[metric]["name"] for metric in task_metrics]

        # get kwargs for these metrics
        kwargs = [
            task_metrics[metric]["kwargs"] if "kwargs" in task_metrics[metric] else {}
            for metric in task_metrics
        ]

        # get user-defined display names for these metrics
        display = list(task_metrics.keys())

        # wrap metrics in a list if more than 1
        objects = [mapper[n](name=d, **k) for n, d, k in zip(names, display, kwargs)]

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
        kws = ["learning_rate", "rho", "momentum", "epsilon", "centered", "use_ema", "ema_momentum", "ema_overwrite_frequency"]
        return tf.keras.optimizers.experimental.RMSprop(**extract_args(config, kws))
    elif config["method"] == "sgd":
        kws = ["learning_rate", "momentum", "nesterov", "use_ema", "ema_momentum", "ema_overwrite_frequency"]
        return tf.keras.optimizers.experimental.SGD(**extract_args(config, kws))
    elif config["method"] == "adadelta":
        kws = ["learning_rate", "rho", "epsilon", "use_ema", "ema_momentum", "ema_overwrite_frequency"]
        return tf.keras.optimizers.experimental.Adadelta(**extract_args(config, kws))
    elif config["method"] == "adagrad":
        kws = ["learning_rate", "initial_accumulator_value", "epsilon", "use_ema", "ema_momentum", "ema_overwrite_frequency"]
        return tf.keras.optimizers.experimental.Adagrad(**extract_args(config, kws))
    elif config["method"] == "adam":
        kws = ["learning_rate", "beta_1", "beta_2", "epsilon", "amsgrad", "use_ema", "ema_momentum", "ema_overwrite_frequency"]
        return tf.keras.optimizers.Adam(**extract_args(config, kws))
    else:
        raise ValueError(
            "config['method'] must be one of 'adadelta', 'adam', 'adagram', 'rms', or 'sgd'"
        )
