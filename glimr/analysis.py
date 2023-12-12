from ray.tune.search import sample
import os
import pandas as pd
import numpy as np
import ray
from inspect import isfunction


def _parse_experiment(exp_dir):
    """Parses an experiment directory, returning results as a pandas.Dataframe

    Parameters
    ----------
    exp_dir : str
        Path containing ray tune experiment output.

    Returns
    -------
    df : pandas.DataFrame
        A dataframe where each row represents a trial, and each column a metric.
    """

    dataframes = []
    subdirs = os.listdir(exp_dir)
    counter = 1
    for i, subdir in enumerate(subdirs):
        if subdir.startswith("trainable") and os.path.isdir(
            os.path.join(exp_dir, subdir)
        ):
            result_path = os.path.join(exp_dir, subdir, "result.json")
            if os.path.exists(result_path):
                df = pd.read_json(result_path, lines=True)
                df.insert(0, "trial_#", i)
                df.insert(1, "subdir", subdir)
                df.insert(2, "exp_dir", exp_dir)
                dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def _get_chckpt_path(training_iteration, subdir, exp_dir):
    """Recover checkpoint path given a trial_id and epoch"""
    chckpt_num = f"checkpoint_{str(training_iteration - 1).zfill(6)}"
    chckpt_path = os.path.join(exp_dir, subdir, chckpt_num, "")
    if not os.path.exists(chckpt_path):
        return None
    return chckpt_path


def _checkpoints(df):
    """Build list of checkpoints."""
    return [
        _get_chckpt_path(it, sb, dr)
        for it, sb, dr in zip(df["training_iteration"], df["subdir"], df["exp_dir"])
    ]


def get_top_k_trials(
    exp_dir, metric=None, mode="max", k=10, drop_dups=True, config_filter=None
):
    """Returns the top k trials of a ray tune experiment as measured by a given metric.

    Given an experiment output path, this function returns the top k trials based on a
    specified metric or custom filtering options.

    Parameters
    ----------
    exp_dir : str
        The directory path of the ongoing or saved ray tune experiment.
    metric : str
        The metric name used for sorting the trials. If `None`, the first metric
        reported by ray tune will be used. Default value is `None`.
    mode : str
        Sorting order to determine top trials. Must be one of "max" or "min". By default,
        `mode="max"` sorts trials in descending order.
    k : int
        The number of trials to retrieve. If `None` all trials will be returned. Default
        value is 10.
    drop_dups : bool
        If `True` duplicate trials will be eliminated from the output, retaining only
        the first / best trial as determined by `metric` and `mode`. If `False`,
        multiple epochs/checkpoints from a single trial may be included in the output.
        Default value is `True`.
    config_filter : function
        An optional function for filtering the rows of the dataframe based on values
        in the config dictionary of a trial. This function should accept the trial
        configuration dicationary as its only input, and should return a boolean
        indicating whether the trial should be included (True) or not (False). Default
        value is `None`.

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

    final_df = _parse_experiment(exp_dir)

    if metric is not None and metric not in final_df.columns:
        raise ValueError(
            f"Argument metric must be None or one of {final_df.columns.tolist()}."
        )

    if metric is None:
        metric = final_df.columns[1]

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

    final_df["checkpoint_path"] = _checkpoints(final_df)
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

    queried = _parse_experiment(exp_dir)
    columns = queried.columns
    if metric is not None:
        bool_list = [True if i in columns else False for i in metric]
        if not all(bool_list):
            raise ValueError(f"{metric[bool_list.index(False)]} is not a valid metric")
        queried = queried.loc[:, metric]

    return queried


def top_cv_trials(exp_dir, metric=None, mode="max", model_selection="fold_bests"):
    final_df = _parse_experiment(exp_dir)

    # drop duplicates
    final_df.drop_duplicates(
        subset="trial_id", keep="first", inplace=True, ignore_index=True
    )

    # fill None values with ''
    final_df.fillna("", inplace=True)

    # get fold indexes
    final_df["folds"] = [fold["data"]["cv_fold_index"] for fold in final_df["config"]]

    # check if metric(s) exists in trials
    if metric is not None:
        if isinstance(metric, dict):
            for key in metric:
                if key not in final_df.columns:
                    raise ValueError(
                        (
                            "Argument metric must be None or one or any combinations of "
                            f"{final_df.columns.tolist()}."
                        )
                    )
    if metric is None:
        metric = final_df.columns[1]

    # sort results based on metric
    if isinstance(metric, str):
        final_df.sort_values(
            by=metric, ascending=(mode == "min"), inplace=True, ignore_index=True
        )

    # define "combined metric" column if metric is a dict.
    # Example: metric={'softmax_auc': 0.7, 'softmax_balanced': 0.3}
    if isinstance(metric, dict):
        keys = list(metric.keys())
        final_df["combined_metric"] = np.sum(
            final_df.filter(keys).values
            * np.expand_dims(np.array(list(metric.values())), 0),
            -1,
        )
        model_selection_metric = "combined_metric"
    else:
        model_selection_metric = metric

    # build in functions
    def fold_bests(df, metric, mode):
        temp_df = df[df["checkpoint_path"] != ""]
        if mode == "max":
            idx = temp_df.groupby(["folds"])[metric].idxmax()
        else:
            idx = temp_df.groupby(["folds"])[metric].idxmin()
        out_df = temp_df.loc[idx]
        out_df = out_df.reset_index(drop=True)
        return out_df

    def global_best(df, metric, mode):
        temp_df = df[df["checkpoint_path"] != ""]
        if mode == "max":
            idx = temp_df[metric].idxmax()
        else:
            idx = temp_df[metric].idxmin()
        out_df = temp_df.loc[idx]
        out_df = out_df.reset_index(drop=True)
        return out_df

    # get checkpoint dirs
    final_df["checkpoint_path"] = _checkpoints(final_df)

    # drop unnecessary columns
    if isinstance(metric, dict):
        metric_names = list(metric.keys()) + ["combined_metric"]
    else:
        metric_names = [metric]
    metadata = [
        "folds",
        "trial_id",
        "training_iteration",
        "checkpoint_path",
        "config",
    ] + metric_names
    final_df = final_df[metadata]

    # model selection
    if model_selection is not None:
        if isinstance(model_selection, str):
            selected_configs = eval(model_selection)(
                final_df, model_selection_metric, mode
            )
        elif isfunction(model_selection):
            selected_configs = model_selection(final_df, model_selection_metric, mode)
        return selected_configs
    else:
        return final_df
