from ray.tune.search import sample
import os
import pandas as pd
import numpy as np
import ray
from inspect import isfunction
from collections import OrderedDict, defaultdict
import json


def _parse_experiment(exp_dir):
    """Parses an experiment directory returning results as a pandas.Dataframe

    Parameters
    ----------
    exp_dir : str
        Path containing ray tune experiment output.

    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame containing performance metrics for each epoch and trial of
        the experiment.
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


def _mean_config(serialized_config):
    """Average the numerical parameters of multiple models."""

    def _checkNumber(l):
        return all(isinstance(i, (int, float)) for i in l)

    merged_config = {
        key: [cfg[key] for cfg in serialized_config]
        for key in serialized_config[0].keys()
    }
    avg_config = {
        key: int(np.mean(merged_config[key]))
        if isinstance(merged_config[key][0], int)
        else np.mean(merged_config[key])
        if _checkNumber(merged_config[key])
        else merged_config[key][0]
        for key in merged_config
    }
    return avg_config


def _serialize_config(config, result=None, prefix=""):
    """Serialize a nested config dictionary for model parameter averaging."""

    if result is None:
        result = dict()
    for k, v in config.items():
        new_k = "__".join((prefix, k)) if prefix else k
        if not (isinstance(v, dict) or isinstance(v, OrderedDict)):
            result.update({new_k: v})
        else:
            _serialize_config(v, result, new_k)
    return result


def _deserialize_config(serialized_config, result=None):
    """Deserialize a nested config dictionary for model parameter averaging."""

    def tree():
        return defaultdict(tree)

    def rec(keys_iter, value):
        _r = tree()
        try:
            _k = next(keys_iter)
            _r[_k] = rec(keys_iter, value)
            return _r
        except StopIteration:
            return value

    if result is None:
        result = dict()
    for k, v in serialized_config.items():
        keys_nested_iter = iter(k.split("__"))
        cur_level_dict = result
        while True:
            try:
                k = next(keys_nested_iter)
                if k in cur_level_dict:
                    cur_level_dict = cur_level_dict[k]
                else:
                    cur_level_dict[k] = json.loads(json.dumps(rec(keys_nested_iter, v)))
            except StopIteration:
                break
    return result


def _filter_checkpoints(trial):
    """Remove non-checkpointed and last checkpoint rows from a trial DataFrame.
    Each row in the experiment DataFrame generated by _parse_experiment describes performance
    metrics for one epoch of one trial. Default checkpointing saves two checkpoints per
    trial: 1. The last epoch and 2. The best epoch. This function discards
    non-checkpointed rows and removes the row corresponding to the last epoch. If more
    than one row remains for a trial an error is thrown.
    """
    non_null_check_rows = trial.dropna(subset=["checkpoint_path"])
    if len(non_null_check_rows) == 1:
        return non_null_check_rows.iloc[0]
    elif len(non_null_check_rows) == 2:
        return non_null_check_rows.loc[
            non_null_check_rows["training_iteration"].idxmin()
        ]
    else:
        raise ValueError(
            f"There are more than two checkpoints for trial {non_null_check_rows.iloc[0]['trial']}"
        )


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
    """Create a DataFrame containing trial performance data.

    The dataframe describes the per-epoch performance of every trial in the experiment,
    along with information on configurations, folds (if cross validation used), and
    checkpoint locations.

    Parameters
    ----------
    exp_dir : str
        Path containing ray tune experiment output.
    metric : list
        List of metrics to collect. Loss value is a valid metric. If `None`, the table
        will contain all trial information.

    Returns
    -------
    queried: pandas.DataFrame
        A pandas DataFrame containing performance metrics for each epoch and trial of
        the experiment.
    """

    queried = _parse_experiment(exp_dir)
    columns = queried.columns
    if metric is not None:
        bool_list = [True if i in columns else False for i in metric]
        if not all(bool_list):
            raise ValueError(f"{metric[bool_list.index(False)]} is not a valid metric")
        queried = queried.loc[:, metric]

    return queried


def top_cv_trials(
    exp_dir, metric, mode="max", model_selection="fold_top_1", average_config=False
):
    """Returns the top k cross validation trials of a ray tune experiment.

    Given an experiment output path, this function returns the top k cross validation trials
    based on a specified metric and model selection criterion.

    Parameters
    ----------
    exp_dir : str
        The directory path of the ongoing or saved ray tune experiment.
    metric : [str, dict]
        The metric name used for sorting the trials. If `None`, the first metric
        reported by ray tune will be used. It can be also possible to use a combination
        of different metrics reported by ray to sort trials by defining the metric
        as a dictionary as below. Default value is `None`.
        metric = {"auc": 0.7, "f1-score": 0.3}
    mode : str
        Sorting order to determine top trials. Must be one of "max" or "min". By default,
        `mode="max"` sorts trials in descending order.
    model_selection : [str, function]
        The criterion used for selecting best models within folds or globally. The possible
        options for model selection criterion is as follows:
            1. "fold_top": it selects the best model among trials of each fold
            2. "global_top": it selects the best model among all trials
            3. "fold_top_<int>: it would select top k trials within each folds. For example,
                "fold_top_2" will select the two most top performing models within each fold.
            4. function: An optional criterion defined as a function outside "top_cv_trials"
            can be also used for model selection.
        Default value is "fold_top_1".
    average_config : bool
        If `True`, the average of top-performing configurations will be calculated.
        Default value is `False`.

    Returns
    -------
    df : pandas.DataFrame
        a pandas DataFrame containing performance metrics and metadata detailing the top k trials
        (as measued by the specified `metric`) of a ray tune experiment.
    """

    df = _parse_experiment(exp_dir)

    # fill None values with ''
    df.fillna("", inplace=True)

    # add fold indexes to table
    df["cv_index"] = [fold["data"]["cv_index"] for fold in df["config"]]

    # add checkpoint directories and remove non-checkpointed entries
    df["checkpoint_path"] = _checkpoints(df)
    df = df.groupby("trial_id").apply(_filter_checkpoints).reset_index(drop=True)

    # check metric formatting and presence in report
    if isinstance(metric, str) and metric not in df.columns:
        raise ValueError(f"metric {metric} not found in results.")
    elif isinstance(metric, dict):
        for k, v in metric.items():
            if k not in df.columns:
                raise ValueError(f"metric {k} not found in results.")
            if not isinstance(v, float):
                raise ValueError(
                    f'metric dict values must be float, received "{type(v)}.__name__"'
                )

    # calculate weighted metric if dict
    if isinstance(metric, dict):
        keys = list(metric.keys())
        df["weighted_metric"] = np.sum(
            df.filter(keys).values * np.expand_dims(np.array(list(metric.values())), 0),
            -1,
        )
        selection_metric = "weighted_metric"
    else:
        selection_metric = metric

    # build in functions
    def fold_top(df, metric, mode, k=1):
        temp_df = df[df["checkpoint_path"] != ""]
        if mode == "max":
            idx = [
                index[1]
                for index in temp_df.groupby("folds")[metric].nlargest(k).index.values
            ]
        else:
            idx = [
                index[1]
                for index in temp_df.groupby("folds")[metric].nsmallest(k).index.values
            ]
        out_df = temp_df.loc[idx]
        out_df = out_df.reset_index(drop=True)
        return out_df

    def global_top(df, metric, mode, k=1):
        temp_df = df[df["checkpoint_path"] != ""]
        if mode == "max":
            idx = temp_df[metric].nlargest(k).index.values
        else:
            idx = temp_df[metric].nsmallest(k).index.values
        out_df = temp_df.loc[idx]
        out_df = out_df.reset_index(drop=True)
        return out_df

    # drop unnecessary columns
    if isinstance(metric, dict):
        metric_names = list(metric.keys()) + ["weighted_metric"]
    else:
        metric_names = [metric]
    metadata = [
        "cv_index",
        "trial_id",
        "training_iteration",
        "checkpoint_path",
        "config",
    ] + metric_names
    df = df[metadata]

    # model selection
    if model_selection is not None:
        if isinstance(model_selection, str):
            if "top" in model_selection:
                s1, s2, k = model_selection.split("_")
                selected_configs = eval("_".join([s1, s2]))(
                    df, selection_metric, mode, int(k)
                )
            else:
                selected_configs = eval(model_selection)(df, selection_metric, mode)
        elif isfunction(model_selection):
            selected_configs = model_selection(df, selection_metric, mode)
        if average_config:
            cfgs = selected_configs["config"].values
            serialized_config = [_serialize_config(cfg) for cfg in cfgs]
            avg_config = _mean_config(serialized_config)
            selected_configs = _deserialize_config(avg_config)
        return selected_configs
    else:
        return df
