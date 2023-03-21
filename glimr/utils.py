from ray import tune
from survivalnet2.losses.cox import cox, efron
from survivalnet2.losses.parametric import Exponential, Weibull, Gompertz
from survivalnet2.metrics.brier import Brier
from survivalnet2.metrics.concordance import (
    HarrellsC,
    SomersD,
    GoodmanKruskalGamma,
    KendallTauA,
    KendallTauB,
)
from survivalnet2.metrics.dcal import Dcal
from survivalnet2.metrics.logrank import Logrank
import tensorflow as tf


def check_tunable(value, expected_type, arg_name):
    """A function for checking the values provided to define tunable objects.

    This function is used by configuration-generating functions to ensure that the values defining
    the range or categories used in search are valid. For sets and lists, each element is checked.

    Parameters
    ----------
    value : int, float, str or list of int, float, str, or set of int, float, str
        The value to check.
    expected_type : type
        The expected type. For sets or lists, we check that all elements are of this type.
    arg_name : str
        Name of argument in configuration. Used for producing informative error messages.
    """

    type_name = expected_type.__name__
    if not isinstance(value, expected_type):
        # set can be artibrary number of any item type, so long as consistent
        if isinstance(value, set):
            for v in value:
                if not isinstance(v, expected_type):
                    raise ValueError(
                        f"{arg_name} set expects elements of type {type_name}, received {type(v).__name__}"
                    )

        # list can be a 2- or 3-list of int or float defining a sampling interval
        elif isinstance(value, list):
            if expected_type not in [float, int]:
                raise ValueError(
                    f"list elements should be int or float, cannot define sampling interval from {type_name}"
                )
            for v in value:
                if not isinstance(v, expected_type):
                    raise ValueError(
                        f"{arg_name} list expects elements of type {type_name}, received {type(v).__name__}"
                    )
            if len(value) not in [2, 3]:
                raise ValueError(
                    f"{arg_name} list should be [min, max] or [min, max, increment], received {value}"
                )

        else:
            if expected_type in [float, int]:
                raise ValueError(
                    f"{arg_name} must be {type_name}, list[{type_name}], or set[{type_name}], received {type(value).__name__}"
                )
            else:
                raise ValueError(
                    f"{arg_name} must be {type_name} or set[{type_name}], received {type(value).__name__}"
                )


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


def string_to_loss(loss):
    """Converts a loss name to a loss object.

    Serialization of the loss name is preferred for hyperparameter tuning.
    Loss objects are not directly serializable in many cases."""

    # case insensitive comparison
    loss_i = loss.casefold()

    if loss_i == "cox":
        return cox
    elif loss_i == "efron":
        return efron
    elif loss_i == "exponential":
        return Exponential()
    elif loss_i == "weibull":
        return Weibull()
    elif loss_i == "gompertz":
        return Gompertz()
    else:
        raise ValueError(
            "loss must be one of 'cox', 'efron', 'exponential', 'weibull', or 'gompertz'"
        )


def string_to_metric(metric, name=None):
    """Converts a metric name to a metric object.

    Serialization of the metric name is preferred for hyperparameter tuning.
    Metric objects are not directly serializable in many cases."""

    # if name is None, default to metric value
    if name is None:
        name = metric

    # case insensitive comparison
    metric_i = metric.casefold()

    if metric_i == "brier":
        return Brier(name=name)
    elif metric_i == "harrellsc":
        return HarrellsC(name=name)
    elif metric_i == "somersd":
        return SomersD(name=name)
    elif metric_i == "goodmankruskalgamma":
        return GoodmanKruskalGamma(name=name)
    elif metric_i == "kendalltaua":
        return KendallTauA(name=name)
    elif metric_i == "kendalltaub":
        return KendallTauB(name=name)
    elif metric_i == "dcal":
        return Dcal(name=name)
    elif metric_i == "logrank":
        return Logrank(name=name)
    else:
        raise ValueError(
            "metric must be one of 'brier', 'harrellsc', 'somersd', 'goodmankruskalgamma', 'kendalltaua', 'kendalltaub', 'dcal', or 'logrank'"
        )


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


def set_hyperparameter(value):
    """Converts hyperparameter arguments to ray.tune search spaces.

    Recurses through nested dictionaries converting list and set values to ray.tune search
    spaces. All other values are set as specific hyperparameter choices.

    Parameters
    ----------
    value : dict, set, list, or other
        A nested dictionary of hyperparameters, containing list, set, and other entries.
    """

    # user provides set, list, or specific value like int, float, str, or dict
    if isinstance(value, set):
        output = tune.choice(list(value))

    elif isinstance(value, list):
        # determine increment
        if len(value) == 2:
            if isinstance(value[0], int):
                inc = 1
            elif isinstance(value[0], float):
                inc = (value[1] - value[0]) / 10.0
        elif len(value) == 3:
            inc = value[2]
        else:
            raise ValueError(
                f"value argument list should be [min, max] or [min, max, increment], received {value}"
            )

        # set min, max, and increment depending on type
        if isinstance(value[0], int):
            output = tune.qrandint(value[0], value[1], inc)
        elif isinstance(value[0], float):
            output = tune.quniform(value[0], value[1], inc)

    elif isinstance(value, dict):
        output = {name: set_hyperparameter(value[name]) for name in value.keys()}

    else:
        output = value

    return output
