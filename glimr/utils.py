from ray.tune.search import sample
import os
import pandas as pd


def get_top_k_trials(
    dir, metric="softmax_balanced", k=10, drop_dups=True, config_filter=None
):
    """
    Returns the top k-many trials of a ray tune experiment as measured by a given metric.

    Given the directory path of a ray tune experiment as input, this function returns the top k-many
    trials of the experiment based on a specified metric, while also allowing for custom filtering options.

    Parameters
    ----------
    dir : str
        The directory path of the ongoing or saved ray tune experiment. This path should contain
        subdirectories with the prefix "trainable" for each trial conducted in the experiment.
    metric : str
        Used to specify the column in the dataframe that will be used for sorting the trials.
        If `metric=None`, no sorting will take place. By default, `metric="softmax_balanced"`.
    k : int
        The number of trials to retrieve. If `k=None`, all trials will be returned. By default, `k=10`.
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
    dataframes = []

    for subdir in os.listdir(dir):
        if subdir.startswith("trainable") and os.path.isdir(os.path.join(dir, subdir)):
            json_path = os.path.join(dir, subdir, "result.json")
            if os.path.exists(json_path):
                df = pd.read_json(json_path, lines=True)
                dataframes.append(df)

    final_df = pd.concat(dataframes, ignore_index=True)

    if metric is not None:
        final_df.sort_values(
            by=metric, ascending=False, inplace=True, ignore_index=True
        )

    if drop_dups:
        final_df.drop_duplicates(
            subset="trial_id", keep="first", inplace=True, ignore_index=True
        )

    if config_filter is not None:
        final_df = final_df[final_df["config"].apply(lambda c: config_filter(c))]

    if k is not None:
        final_df = final_df.head(k)

    metadata = ["trial_id", "training_iteration", "config"]
    column_order = metadata + [col for col in df.columns if col not in metadata]

    return final_df[column_order]


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
