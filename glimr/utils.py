from ray.tune.search import sample
import os
import pandas as pd


def get_top_k_trials(
    exp_dir, metric=None, mode="max", k=10, drop_dups=True, config_filter=None
):
    """
    Returns the top k-many trials of a ray tune experiment as measured by a given metric.

    Given the directory path of a ray tune experiment as input, this function returns the top k-many
    trials of the experiment based on a specified metric, while also allowing for custom filtering options.

    Parameters
    ----------
    exp_dir : str
        The directory path of the ongoing or saved ray tune experiment. This path should contain
        subdirectories with the prefix "trainable" for each trial conducted in the experiment.
    metric : str
        Used to specify the column in the dataframe that will be used for sorting the trials.
        If `metric=None`, the metric will be taken as the first of the available metrics reported by
        ray tune. By default, `metric=None`.
    mode : str
        Specifies whether to report the k-most maximum or k-most minimum trials as measured by
        the provided `metric`. Must be one of "max" or "min". By default, `mode="max"`.
    k : int
        The number of trials to retrieve. If `k=None` all trials will be returned. By default, `k=10`.
    drop_dups : bool
        A boolean flag determining whether duplicate trials should be dropped from the final dataframe.
        If set to True, only the first occurrence of each trial_id will be kept in the dataframe, and
        therefore each trial will be represented by its best performing epoch as determined by `metric`.
        If set to False, then many epochs of a single trial could (in theory) be included in the final
        dataframe, depending on their performance. By default, `drop_dups=True`.
    config_filter : function
        An optional function that can be used to filter the rows of the dataframe based on the values in
        the config dictionary of a ray tune experiment trial. The given function must take a single
        argument, the dictionary representing the configuration of a trial, and must return a boolean
        value indicating whether the trial should be included or not. By default, `config_filter=None`.

    Returns
    -------
    final_df : pandas.DataFrame
        a pandas DataFrame containing performance metrics and metadata detailing the top k trials
        (as measued by the specified `metric`) of a ray tune experiment.
    """

    if mode not in ["max", "min"]:
        raise ValueError("Argument mode must be one of 'max' or 'min'.")

    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("Argument k must be a positive integer or None.")

    dataframes = []
    subdirs = os.listdir(exp_dir)
    for subdir in subdirs:
        if subdir.startswith("trainable") and os.path.isdir(
            os.path.join(exp_dir, subdir)
        ):
            json_path = os.path.join(exp_dir, subdir, "result.json")
            if os.path.exists(json_path):
                df = pd.read_json(json_path, dtype=False, lines=True)
                dataframes.append(df)

    final_df = pd.concat(dataframes, ignore_index=True)

    if metric is not None and metric not in final_df.columns:
        raise ValueError(
            f"Argument metric must be None or one of {final_df.columns.tolist()}."
        )

    if metric is None:
        metric = final_df.columns[0]

    final_df.sort_values(
        by=metric, ascending=(mode == "min"), inplace=True, ignore_index=True
    )

    if drop_dups:
        final_df.drop_duplicates(
            subset="trial_id", keep="first", inplace=True, ignore_index=True
        )

    if config_filter is not None:
        final_df = final_df[final_df["config"].apply(lambda c: config_filter(c))]

    if k is not None and k < final_df.shape[0]:
        final_df = final_df.head(k)

    def _get_chckpt_path(trial_id, training_iteration):
        subdir = [s for s in subdirs if s.startswith(f"trainable_{trial_id}")][0]
        chckpt_num = f"checkpoint_{str(training_iteration - 1).zfill(6)}"
        chckpt_path = os.path.join(exp_dir, subdir, chckpt_num, "")
        if not os.path.exists(chckpt_path):
            return None
        return chckpt_path

    final_df["checkpoint_path"] = [
        _get_chckpt_path(id, it)
        for id, it in zip(final_df["trial_id"], final_df["training_iteration"])
    ]
    metadata = ["trial_id", "training_iteration", "checkpoint_path", "config", metric]
    column_order = metadata + [col for col in df.columns if col not in metadata]

    return final_df[column_order]


def get_trial_info(exp_dir, metric=None):
    """
    Given the directory path of a ray tune experiment as input, this function returns data on each trial
    as specified by the given metric(es). If metric is `None`, a generic dataframe containing all
    trial information is returned.

    Parameters
    ----------
    exp_dir : str
        The directory path of the ongoing or saved ray tune experiment. This path should contain
        subdirectories with the prefix "trainable" for each trial conducted in the experiment.
    metric : list
        List of metric(es) used to query from all trial information. Loss value is a valid metric.
        If `None`, all trial information is returned
    Returns
    ----------
    queried: pandas.DataFrame
        A pandas DataFrame containing performance metrics of the all trials of a ray tune experiment.
    """
    dataframes = []
    subdirs = os.listdir(exp_dir)
    counter = 1
    for subdir in subdirs:
        if subdir.startswith("trainable") and os.path.isdir(
            os.path.join(exp_dir, subdir)
        ):
            result_path = os.path.join(exp_dir, subdir, "result.json")
            if os.path.exists(result_path):
                df = pd.read_json(result_path, lines=True)
                df.insert(0, "trial_#", counter)
                dataframes.append(df)
                counter += 1
    queried = pd.concat(dataframes, ignore_index=True)
    columns = queried.columns
    if metric is not None:
        bool_list = [True if i in columns else False for i in metric]
        if not all(bool_list):
            raise ValueError(f"{metric[bool_list.index(False)]} is not a valid metric")
        queried = queried.loc[:, metric]
    return queried


def prune_constants_functions(space):
    """Prepares a search space for ray tune's PBT scheduler by pruning constants and functions.

    This function recurses through a nested dictionary defining a search space, removing
    any constant values and conditional functions (defined through tune.sample_from) from
    the space, where constant values and functions are defined as non-container types,
    non-callable objects, and non-ray.tune.search.sample.Domain or -Sampler objects.
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

    def _is_function(value):
        return isinstance(value, ray.tune.search.sample.Function)

    pruned_space = {}
    for key, value in space.items():
        if isinstance(value, dict):
            pruned_value = prune_constants_functions(value)
            if pruned_value:  # don't add empty dicts
                pruned_space[key] = pruned_value
        elif not (_is_constant(value) or _is_function(value)):
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
