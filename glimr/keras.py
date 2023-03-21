import tensorflow as tf


def keras_losses(config):
    """A utility to build loss and loss weight dictionaries for
    tf.keras.Model.compile.

    Given a configuration where task losses are defined as strings, convert the
    strings to loss objects and format a dict for model compilation.

    Parameters
    ----------
    config : dict
        A configuration defining the losses for each task. Config should have
        a "tasks" key that links to task dictionaries. Each task dictionary
        should define a "loss" key with a str value defining the name of the
        loss to apply to the task.

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
    losses = {
        name: string_to_loss(config["tasks"][name]["loss"]) for name in config["tasks"]
    }

    # create loss weight dictionary
    weights = {name: config["tasks"][name]["loss_weight"] for name in config["tasks"]}

    return losses, weights


def keras_metrics(config):
    """A utility to build metric dictionaries for tf.keras.Model.compile.

    Given a configuration where task metrics are defined as strings, convert
    the strings to tf.keras.metrics.Metric objects and format a dict for model
    compilation. This handles the expected naming of metrics which differs for
    single task and multiple task models.

    Parameters
    ----------
    config : dict
        A configuration defining the metrics for each task. Config should have
        a "tasks" key that links to task dictionaries. Each task dictionary
        should define a "metrics" key with a dict value that maps metric
        display names to str values that will be consumed by string_to_metric.

    Returns
    -------
    metrics : dict
        A dict of model output name : tf.keras.losses.Loss key value pairs
        for use with tf.keras.Model.compile.
    """

    # create a metric dictionary from the config
    metrics = {}
    for task in config["tasks"]:
        # get metric names for task
        names = list(config["tasks"][task]["metrics"].values())

        # get user-defined designations for these metrics
        display = list(config["tasks"][task]["metrics"].keys())

        # wrap metrics in a list if more than 1
        task_metrics = [string_to_metric(n, d) for n, d in zip(names, display)]

        # assign to metrics dict
        if len(names) > 1:
            metrics[task] = task_metrics
        else:
            metrics[task] = task_metrics[0]

    return metrics


def keras_optimizer(optimizer):
    """Convert config options to a tf.keras.optimizers.Optimizer that can be used in
    model compilation and training. Sets both the optimization method and hyperparameters.
    """

    def extract_args(optimizer, kws):
        return {kw: optimizer[kw] for kw in kws if kw in optimizer.keys()}

    if optimizer["method"] == "rms":
        kws = ["learning_rate", "rho", "momentum", "epsilon", "centered"]
        return tf.keras.optimizers.RMSprop(**extract_args(optimizer, kws))
    elif optimizer["method"] == "sgd":
        kws = ["learning_rate", "momentum", "nesterov"]
        return tf.keras.optimizers.SGD(**extract_args(optimizer, kws))
    elif optimizer["method"] == "adadelta":
        kws = ["learning_rate", "rho", "epsilon"]
        return tf.keras.optimizers.Adadelta(**extract_args(optimizer, kws))
    elif optimizer["method"] == "adagrad":
        kws = ["learning_rate", "initial_accumulator_value", "epsilon"]
        return tf.keras.optimizers.Adagrad(**extract_args(optimizer, kws))
    elif optimizer["method"] == "adam":
        kws = ["learning_rate", "beta_1", "beta_2", "epsilon", "amsgrad"]
        return tf.keras.optimizers.Adam(**extract_args(optimizer, kws))
    else:
        raise ValueError(
            "optimizer['method'] must be one of 'adadelta', 'adam', 'adagram', 'rms', or 'sgd'"
        )
