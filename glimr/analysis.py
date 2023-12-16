from collections import OrderedDict, defaultdict
from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
import ray
from ray.tune.search import sample
from inspect import isfunction


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


def _config_enum(configurations):
    """Enumerate configurations to link configuration across cv folds / trials"""

    def remove_function(config):
        # remove function keys
        clean = deepcopy(config)
        for k, v in config.items():
            if isinstance(v, str):
                if v.startswith("<function"):
                    del clean[k]
            elif isinstance(v, dict):
                clean[k] = remove_function(v)
        return clean

    cleaned = []
    for config in configurations:
        clean = deepcopy(config)
        clean = remove_function(clean)
        del clean["data"]["cv_index"]
        cleaned.append(clean)
    mapping = {}
    for clean in cleaned:
        if json.dumps(clean) not in mapping.keys():
            mapping[json.dumps(clean)] = len(mapping)
    enumerated = [mapping[json.dumps(clean)] for clean in cleaned]
    return enumerated


def experiment_table(exp_dir, checkpointed=True):
    """Parses an experiment directory returning results as a pandas.Dataframe.

    The Dataframe can be used to analyze performance statistics of individual trials
    and to identify checkpointed models of interest. Examples include identifying the
    top models in each cross validation fold, or the configuration with the best overall
    performance or highest median cross-validated performance. Each row of this table
    represents one epoch of one trial.

    Parameters
    ----------
    exp_dir : str
        Path containing ray tune experiment output.
    checkpointed : bool
        Retain only rows corresponding to checkpointed trials. Default value is True.

    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame containing performance, configuration, and cross-validation
        (if applicable) data for each trial.
    """

    # build dataframe
    trials = []
    subdirs = os.listdir(exp_dir)
    for i, subdir in enumerate(subdirs):
        if subdir.startswith("trainable") and os.path.isdir(
            os.path.join(exp_dir, subdir)
        ):
            result_path = os.path.join(exp_dir, subdir, "result.json")
            if os.path.exists(result_path):
                trial = pd.read_json(result_path, lines=True)
                trial.insert(0, "trial_#", i)
                trial.insert(1, "subdir", subdir)
                trial.insert(2, "exp_dir", exp_dir)
                trials.append(trial)
    df = pd.concat(trials, ignore_index=True)

    # fill None values with ''
    df.fillna("", inplace=True)

    # add checkpoint directories and remove non-checkpointed entries
    df["checkpoint_path"] = _checkpoints(df)
    if checkpointed:
        df = df.dropna(subset=["checkpoint_path"]).reset_index()

    # add cross-validation fold index
    if "cv_index" in df.iloc[0]["config"]["data"]:
        df["cv_index"] = [fold["data"]["cv_index"] for fold in df["config"]]

    # add enumerated configurations - used for cross validation
    if "cv_index" in df.loc[0, "config"]["data"]:
        df["config_enum"] = _config_enum(df["config"])

    return df


def default_checkpoints(df):
    """Selects the performance-criteria checkpoints for a set of trials.

    Each trial checkpoints: 1. The last epoch (to support trial resumption) and 2. An
    epoch defined by the performance criteria such as peak accuracy. Given an experiment
    dataframe, this function selects the rows corresponding to the performance
    criteria checkpoint for each trial.

    Parameters
    ----------
    df : pandas.DataFrame
        An experiment dataframe.

    Returns
    -------
    checkpointed : pandas.DataFrame
        A DataFrame containing the selected checkpoints for each trial.
    """

    def trial_checkpoint(trial):
        if len(trial) == 1:
            return trial.iloc[0]
        elif len(trial) == 2:
            return trial.loc[trial["training_iteration"].idxmin()]
        else:
            raise ValueError(
                (
                    "There are more than two checkpoints for trial "
                    f"{trial.iloc[0]['trial']}"
                )
            )
        return trial

    return df.groupby("trial_id").apply(trial_checkpoint).reset_index(drop=True)


def top_k_trials(df, metric, mode="max", k=10, fold=None, config_filter=None):
    """Filter the top k trials from an experiment DataFrame.

    This function can be applied to all trials within an experiment, or within
    a single cross-validation fold. An optional filter argument can be used to
    select configurations for analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame containing performance, configuration, and cross-validation
        data for each trial.
    metric : str
        The metric name used for sorting the trials.
    mode : str
        Sorting order to determine top trials. Must be one of "max" or "min". By 
        default, `mode="max"` sorts trials in descending order.
    k : int
        The number of trials to retrieve.
    fold : int
        Restrict analysis to a specific fold. If `None`, all folds will be included.
    config_filter : function
        An optional function for filtering based on trial configurations. This function 
        should accept the trial configuration dicationary as its only input, and should 
        return a boolean indicating whether the trial should be included (True) or not 
        (False). Default value is `None`.

    Returns
    -------
    final_df : pandas.DataFrame
        A pandas DataFrame containing performance metrics and metadata for the top
        k trials.
    """
    
    if metric not in df.columns:
        raise ValueError(
            f"Argument metric must be None or one of {df.columns.tolist()}."
        )

    if mode not in ["max", "min"]:
        raise ValueError("Argument mode must be one of 'max' or 'min'.")

    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("Argument k must be a positive integer or None.")

    if fold is not None and not isinstance(fold, int):
        raise ValueError("Argument fold must be integer or None.")

    if config_filter is not None:
        df = df[df["config"].apply(lambda c: config_filter(c))]

    if fold is not None:
        df = df.loc[df["cv_index"] == fold]

    df.sort_values(
        by=metric, ascending=(mode=="min"), inplace=True, ignore_index=True
    )

    return df.head(k)


def top_k_configs(df, metric, mode="max", k=10, statistic=np.median):
    """Filter the top k configurations from an experiment DataFrame.
    
    This function analyzes aggregate performance of a cross validation experiment
    to identify the top performing trials. This is 

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame containing performance, configuration, and cross-validation
        data for each trial.
    metric : str
        The metric name used for sorting the trials.
    mode : str
        Sorting order to determine top trials. Must be one of "max" or "min". By 
        default, `mode="max"` sorts trials in descending order.
    k : int
        The number of trials to retrieve.
    fold : int
        Restrict analysis to a specific fold. If `None`, all folds will be included.
    statistic : function
        The method of aggregation used to rank configuration. Default value is
        `np.median`.

    Returns
    -------
    final_df : pandas.DataFrame
        A pandas DataFrame containing performance metrics and metadata for the top
        k configurations.    
    """

    if metric not in df.columns:
        raise ValueError(
            f"Argument metric must be None or one of {df.columns.tolist()}."
        )

    if mode not in ["max", "min"]:
        raise ValueError("Argument mode must be one of 'max' or 'min'.")

    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("Argument k must be a positive integer or None.")

    ranking = np.argsort(df.groupby("config_enum")[metric].apply(statistic))
    if mode == "max":
        ranking = ranking[::-1]
    
    return df.loc[df["config_enum"].isin(ranking[0:k])]
