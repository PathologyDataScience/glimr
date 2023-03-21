from ray import tune
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
