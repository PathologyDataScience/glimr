import datetime
from ray import tune
from ray.air import CheckpointConfig, ScalingConfig
from ray.air.config import FailureConfig, RunConfig
from ray.tune import CLIReporter, JupyterNotebookReporter, SyncConfig
from ray.tune.integration.keras import TuneReportCheckpointCallback
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.tune_config import TuneConfig
from survivalnet2.search.utils import keras_optimizer


class Search(object):
    """A general-purpose class for automated hyperparameter tuning.

    This class provides hyperparameter tuning for any model that has a
    search space and model builder function. Initialization sets reasonable
    options for checkpointing, reporting, stopping criteria, and trial
    resources. Each of these is represented by an attribute and can be
    overrided using class methods. These methods provide a minimal interface
    for a subset of ray parameters, but additional parameters can be set
    using kwargs. Class attributes can also be overrided directly using
    ray library objects generated by the user.
    
    Parameters
    ----------
    space : dict
        A model search space that defines the range of possible model
        characteristics like architecture, activations, dropout, losses,
        and loss weights.
    builder : callable
        A function that returns a tf.keras.model given a configuration
        sampled from the search space.
    loader : callable
        A function that returns batched training and validation datasets
        of type tf.data.Dataset. This function should include a "batch"
        keyword argument and may include other keyword arguments to control
        data loading and preprocessing behavior.
    metric : str
        The name of the metric to optimize. This is in the form "task_name"
        where "task" is the task name and "name" is the key value of the
        metric to optimize. Default value of `None` selects the first
        metric of the first task.
    mode : {"max", "min"}
        Either "max" or "min" indicating whether to maximize or minimize
        the metric. Default value is "max".
    fit_kwargs : dict
        Keyword arguments for tf.keras.model.fit. Allows customization of
        keras model training options. Default value is None.
    
    Attributes
    ----------
    metric : str
        The name of the metric to optimize. This is in the form "task_name"
        where "task" is the task name and "name" is the key value of the
        metric to optimize.
    mode : str
        Either "max" or "min" indicating whether to maximize or minimize
        the metric.
    checkpoint_config : ray.air.CheckpointConfig
        Defines checkpointing preferences for saving the best model from each trial.
    failure_config : ray.air.FailureConfig
        Defines behavior for handling failed trials.
    reporter : ray.tune.CLIReporter or ray.tune.JupyterNotebookReporter
        An optional reporter for displaying experiment progress during tuning.
    scaling_config : ray.air.ScalingConfig
        Defines available compute resources for workers.
    stopper : ray.tune.Stopper
        Defines the stopping criteria for terminating trials. May be overrided
        by scheduler stopping criteria.
    sync_config : tune.SyncConfig
        Defines synchronization options for multi-machine experiments.
        
    Methods
    -------
    set_checkpoints(metric=None, mode="max", num_to_keep=1, **kwargs)
        Set checkpointing behavior by generating a ray.air.CheckpointConfig.
    set_reporter(metrics=None, parameters=None, jupyter=False, sort_by_metric=True,
        max_report_frequency=30, **kwargs)
        Set reporting behavior by generating a reporter object.
    set_scaling(num_workers=1, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 0},
        **kwargs)
        Set experiment resources.
    trainable(config)
        Static function for running a trial.
    experiment(local_dir, name=None, num_samples=100, max_concurrent_trials=8,
        scheduler=AsyncHyperBandScheduler, search_alg=None)
        Runs an experiment.
    """

    def __init__(
        self,
        space,
        builder,
        loader,
        metric=None,
        mode="max",
        loader_kwargs=None,
        fit_kwargs=None,
    ):

        # process kwarg parameters
        if fit_kwargs is None:
            fit_kwargs = {}
        if loader_kwargs is None:
            loader_kwargs = {}

        # add builder, loader, kwargs to space and capture as space member
        space["builder"] = builder
        space["fit_kwargs"] = fit_kwargs
        space["loader"] = loader
        self._space = space

        # extract default optimization metric - first task & first metric
        if metric is None:
            taskname = space["tasks"].keys()[0]
            metricname = space["tasks"][taskname].keys()[0]
            self.metric = f"{taskname}_{metricname}"
        else:
            self.metric = metric

        # capture optimization mode - one of {"min", "max"}
        if mode not in ["min", "max"]:
            raise ValueError("mode must be one of 'min', 'max'.")
        self.mode = mode

        # default CheckpointConfig
        self.set_checkpoints(self.metric, self.mode)

        # default FailureConfig
        self.failure_config = FailureConfig(max_failures=5)

        # default reporter
        self.set_reporter()

        # default ScalingConfig
        self.set_scaling()

        # default trial/experiment stopper
        self.stopper = TrialPlateauStopper(metric=self.metric)

        # default SyncConfig
        self.sync_config = SyncConfig(syncer=None)

    def set_checkpoints(self, metric=None, mode="max", num_to_keep=1, **kwargs):
        """Set checkpointing behavior for saving models.

        This sets the `ray.air.CheckpointConfig` object that is used by
        `ray.air.config.Runconfig`. This controls aspects including the number
        of checkpoints to retain from each trial, and the metric used to select
        the best checkpoint.

        Parameters
        ----------
        metric : str
            The name of the metric to optimize. This is in the form "task_name"
            where "task" is the task name and "name" is the key value of the
            metric to optimize. Default value of `None` selects the first
            metric of the first task.
        mode : {"max", "min"}
            Either "max" or "min" indicating whether to maximize or minimize
            the metric. Default value is "max".
        num_to_keep : int
            The number of checkpoints to retain from each trial. Default value
            is 1.

        Notes
        -----
        See https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.CheckpointConfig.html
        for additional details.
        """

        if metric is None:
            metric = self.metric
        self.checkpoint_config = CheckpointConfig(
            checkpoint_score_attribute=metric,
            checkpoint_score_order=mode,
            num_to_keep=num_to_keep,  # saved checkpoints per trial
            **kwargs,
        )

    def set_reporter(
        self,
        metrics=None,
        parameters=None,
        jupyter=False,
        sort_by_metric=True,
        max_report_frequency=30,
        **kwargs,
    ):
        """Creates a reporter to display trial results and progress during tuning.

        Parameters
        ----------
        metrics : list(string)
            A list of metrics to display during tuning. Metrics have the form
            `task_metric` where `task` is the task name and `metric` is the metric
            name from the model configuration.
        parameters : dict
            A dictionary of configuration parameters to display during tuning.
            Each key is an index into the configuration dictionary, and each
            value is the name this parameter will be displayed as. Nested
            parameters are indicated using a `/`. Default value of
            `{"optimization/method": "method", "optimization/learning_rate": "learning rate"}`
            will display the `config["optimization"]["method"]` as "method" and
            `config["optimization"]["learning_rate"]` as "learning_rate".
        jupyter : bool
            If running trials in Jupyter, selecting `True` will report updates
            in-place. If `False` reports will be periodically appended to
            command line output. Default value is False.
        sort_by_metric : bool
            If `True`, trials will be sorted by the single metric used to rank
            experiments. Default value is True.
        max_report_frequency : int
            The number of seconds between updates. Default value is 30.

        Notes
        -----
        See https://docs.ray.io/en/latest/tune/api/reporters.html for details
        on reporting in ray tune.
        """

        # set metrics, parameters if None
        if metrics is None:
            metrics = [
                f"{t}_{m}"
                for t in self._space["tasks"]
                for m in self._space["tasks"][t]["metrics"]
            ]
        if parameters is None:
            parameters = {
                "optimization/method": "method",
                "optimization/learning_rate": "learning rate",
            }

        # set reporter kwargs
        reporter_kwargs = {
            "metric_columns": metrics,
            "parameter_columns": parameters,
            "sort_by_metric": sort_by_metric,
            "max_report_frequency": max_report_frequency,
            **kwargs,
        }

        # select juptyer or cli output
        if jupyter:
            self.reporter = JupyterNotebookReporter(**reporter_kwargs)
        else:
            self.reporter = CLIReporter(**reporter_kwargs)

    def set_scaling(
        self,
        num_workers=1,
        use_gpu=False,
        resources_per_worker={"CPU": 1, "GPU": 0},
        **kwargs,
    ):
        """Sets cpu and gpu resources used for training.

        This sets the `ray.air.ScalingConfig` object used by `ray.tune`.
        Resources set by in this configuration are assigned to the experiment
        using `ray.tune.with_resources()`. The number of concurrent trials
        that are run is set by `max_concurrent_trials` in `Search.experiment()`.

        Parameters
        ----------
        num_workers : int
            The number of workers to launch. Each worker receives 1 CPU by default.
            Default value is 1.
        use_gpu : bool
            Whether workers should be able to access GPUs. Default value is False.
        resources_per_worker : dict
            The cpu and gpu resources assigned to each worker. These values can
            be fractional. Default value is `{"CPU": 1, "GPU": 0}`.

        Notes
        -----
        See https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.ScalingConfig.html
        for details on the `ScalingConfig` object.

        See https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html for
        details on parallelism in ray tune and how resources are assigned to
        trials in tune experiments.
        """

        self.scaling_config = ScalingConfig(
            num_workers=num_workers,
            use_gpu=use_gpu,
            resources_per_worker=resources_per_worker,
            **kwargs,
        )

    @staticmethod
    def trainable(config):
        """Trains a model from a hyperparameter configuration.

        This function compiles a model from the config and trains using general
        parameters found in self.options. Communication of results is performed
        using the TuneReportCheckpointCallback from ray.tune.integration.keras.

        Parameters
        ----------
        config : dict
            A configuration describing optimization and model hyperparameters
            as well as functions for data loading and model building.
        """

        # create the model from the config
        model, losses, loss_weights, metrics = config["builder"](config)

        # build optimizer
        optimizer = keras_optimizer(config["optimization"])

        # compile the model with the optimizer, losses, and metrics
        model.compile(
            optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics
        )

        # load example data, generate random train/test split
        train_dataset, validation_dataset = config["loader"](
            **config["data"]
        )

        # epoch reporting of performance metrics - link keras metric names
        # to names displayed by ray in reporting
        report = {
            f"{t}_{m}": f"val_{t}_{m}" if len(config["tasks"]) > 1 else f"val_{m}"
            for t in config["tasks"]
            for m in config["tasks"][t]["metrics"]
        }

        callback = TuneReportCheckpointCallback(report)

        # train the model for the desired epochs using the call back
        model.fit(
            train_dataset,
            epochs=config["optimization"]["epochs"],
            validation_data=validation_dataset,
            callbacks=[callback],
            verbose=0,
            **config["fit_kwargs"],
        )

    def experiment(
        self,
        local_dir,
        name=None,
        num_samples=100,
        max_concurrent_trials=8,
        scheduler=AsyncHyperBandScheduler(
            time_attr="training_iteration",
            max_t=100,
            grace_period=10,
            stop_last_trials=False,
        ),
        search_alg=None,
    ):
        """Run hyperparameter tuning experiment trials.

        This can be called multiple times with different search spaces
        definitions, model building functions, schedulers, and search
        algorithms to perform experiments under different conditions.

        Parameters
        ----------
        local_dir : str
            The path to store experiment results including logs and model
            checkpoints.
        name : str
            The name for the experiment. Used by developers to organize
            experimental results. Default value of None uses datetime
            year_month_day_hour_minute_second.
        num_samples : int
            The number of trials to run when tuning a model. Default value 100.
        max_concurrent_trials : int
            The maximum number of trials to run concurrently. Default value 8.
        scheduler : object
            A scheduling algorithm from ray.tune.schedulers that can be used
            to terminate poorly performing trials, to pause trials, to clone
            trials, and to alter hyperparameters of a running trial. Some
            search algorithms do not require a scheduler. Default value is the
            AsyncHyperBandScheduler.
        search_alg : object
            A search algorithm from ray.tune.search for adaptive hyperparameter
            selection. Default value of None results in a random search with
            the AsyncHyperBandScheduler.

        Returns
        -------
        analysis : ray.tune.ResultGrid
            A dictionary describing the tuning experiment outcomes. See the
            documentation of ray.tune.Tuner.fit() and ray.tune.ResultGrid for
            more details.
        """

        # set config name
        if name is None:
            name = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

        # create tune config
        tune_kwargs = {}
        tune_kwargs["max_concurrent_trials"] = max_concurrent_trials
        tune_kwargs["metric"] = self.metric
        tune_kwargs["mode"] = "max"
        tune_kwargs["num_samples"] = num_samples
        tune_kwargs["scheduler"] = scheduler
        if search_alg is not None:
            tune_kwargs["search_alg"] = search_alg
        tune_config = TuneConfig(**tune_kwargs)

        # create run config
        run_kwargs = {}
        run_kwargs["checkpoint_config"] = self.checkpoint_config
        run_kwargs["failure_config"] = self.failure_config
        run_kwargs["local_dir"] = local_dir
        run_kwargs["log_to_file"] = True
        run_kwargs["name"] = name
        run_kwargs["sync_config"] = self.sync_config
        if hasattr(self, "reporter"):
            run_kwargs["progress_reporter"] = self.reporter
        if hasattr(self, "stopper"):
            run_kwargs["stop"] = self.stopper
        run_config = RunConfig(**run_kwargs)

        # add scaling config to search space if
        if hasattr(self, "scaling_config"):
            trainable = tune.with_resources(self.trainable, self.scaling_config)
        else:
            trainable = self.trainable

        # run experiment
        tuner = tune.Tuner(
            trainable,
            param_space=self._space,
            tune_config=tune_config,
            run_config=run_config,
        )
        analysis = tuner.fit()

        return analysis
