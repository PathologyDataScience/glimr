import numpy as np
import os
import pandas as pd
import pytest
import ray
from survivalnet2.layers.advmtl import advmtl_model, advmtl_space, task_space
from survivalnet2.losses.cox import cox, efron
from survivalnet2.losses.parametric import Exponential, Weibull, Gompertz
from survivalnet2.search.search import Search
from survivalnet2.search.utils import (
    check_tunable,
    set_hyperparameter,
    string_to_loss,
    string_to_metric,
    keras_optimizer,
)
from survivalnet2.metrics.concordance import (
    HarrellsC,
    SomersD,
    GoodmanKruskalGamma,
    KendallTauA,
    KendallTauB,
)
from survivalnet2.metrics.logrank import Logrank
import tensorflow as tf
from tensorflow.keras.optimizers import Adadelta, Adam, Adagrad, RMSprop, SGD


ROOT_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_set_hyperparameter():
    # check set specific value - no hyperparameter search
    s = 1.0
    result = set_hyperparameter(s)
    assert result == s

    # check set input - one element
    s = {1}
    result = set_hyperparameter(s)
    assert result.categories == list(s)

    # check set input - multiple elements
    s = {1, 2}
    result = set_hyperparameter(s)
    assert result.categories == list(s)

    # check list input - two int elements (range)
    s = [1, 3]
    result = set_hyperparameter(s)
    assert result.lower == 1
    assert result.upper == 3
    assert result.sampler.q == 1

    # check list input - three int elements (range+increment)
    s = [1, 5, 2]
    result = set_hyperparameter(s)
    assert result.lower == 1
    assert result.upper == 5
    assert result.sampler.q == 2

    # check list input - two float elements (range)
    s = [0.0, 1.0]
    result = set_hyperparameter(s)
    assert result.lower == 0.0
    assert result.upper == 1.0
    assert result.sampler.q == 0.1

    # check list input - three float elements (range+increment)
    s = [0.0, 1.0, 0.1]
    result = set_hyperparameter(s)
    assert result.lower == 0.0
    assert result.upper == 1.0
    assert result.sampler.q == 0.1

    # check that other list input lengths raises ValueError
    s = [0.0]
    with pytest.raises(Exception) as exc:
        set_hyperparameter(s)
    assert (
        str(exc.value)
        == "value argument list should be [min, max] or [min, max, increment], received [0.0]"
    )


def test_check_tunable():
    # check scalar inputs
    with pytest.raises(Exception) as exc:
        check_tunable(1, float, "test")
    assert (
        str(exc.value) == "test must be float, list[float], or set[float], received int"
    )
    with pytest.raises(Exception) as exc:
        check_tunable(1.0, int, "test")
    assert str(exc.value) == "test must be int, list[int], or set[int], received float"
    with pytest.raises(Exception) as exc:
        check_tunable(1.0, str, "test")
    assert str(exc.value) == "test must be str or set[str], received float"

    # check list input - mixed types int
    s = [1, 5.0, 2]
    with pytest.raises(Exception) as exc:
        check_tunable(s, int, "test")
    assert str(exc.value) == "test list expects elements of type int, received float"

    # check list input - mixed types float
    s = [1.0, 5, 2.0]
    with pytest.raises(Exception) as exc:
        check_tunable(s, float, "test")
    assert str(exc.value) == "test list expects elements of type float, received int"

    # check invalid length
    s = [1.0]
    with pytest.raises(ValueError) as exc:
        check_tunable(s, float, "test")
    assert (
        str(exc.value)
        == "test list should be [min, max] or [min, max, increment], received [1.0]"
    )

    # check invalid type
    s = ["a", "c"]
    with pytest.raises(Exception) as exc:
        check_tunable(s, str, "test")
    assert (
        str(exc.value)
        == "list elements should be int or float, cannot define sampling interval from str"
    )


def test_string_to_loss():
    assert string_to_loss("cox") == cox
    assert string_to_loss("Cox") == cox
    assert string_to_loss("efron") == efron
    assert string_to_loss("Efron") == efron
    assert isinstance(string_to_loss("exponential"), Exponential)
    assert isinstance(string_to_loss("Exponential"), Exponential)
    assert isinstance(string_to_loss("weibull"), Weibull)
    assert isinstance(string_to_loss("Weibull"), Weibull)
    assert isinstance(string_to_loss("gompertz"), Gompertz)
    assert isinstance(string_to_loss("Gompertz"), Gompertz)
    with pytest.raises(ValueError) as exc:
        string_to_loss("effron")
    assert (
        str(exc.value)
        == "loss must be one of 'cox', 'efron', 'exponential', 'weibull', or 'gompertz'"
    )


def test_string_to_metric():
    metric = string_to_metric("harrellsc")
    assert isinstance(metric, HarrellsC)
    assert metric.name == "harrellsc"
    assert isinstance(string_to_metric("somersd"), SomersD)
    assert isinstance(string_to_metric("goodmankruskalgamma"), GoodmanKruskalGamma)
    assert isinstance(string_to_metric("kendalltaua"), KendallTauA)
    assert isinstance(string_to_metric("kendalltaub"), KendallTauB)
    assert isinstance(string_to_metric("logrank"), Logrank)
    with pytest.raises(ValueError) as exc:
        string_to_metric("sommersd")
    assert (
        str(exc.value)
        == "metric must be one of 'brier', 'harrellsc', 'somersd', 'goodmankruskalgamma', 'kendalltaua', 'kendalltaub', 'dcal', or 'logrank'"
    )


def optimization_sample():
    return {
        "batch": np.random.choice([32, 64, 128]),
        "method": np.random.choice(["rms", "sgd", "adadelta", "adagrad", "adam"]),
        "learning_rate": np.random.uniform(1e-5, 1e-2),
        "rho": np.random.uniform(0.5, 1.0),
        "momentum": np.random.uniform(0.0, 1e-1),
        "beta_1": np.random.uniform(0.5, 1.0),
        "beta_2": np.random.uniform(0.5, 1.0),
    }


def test_keras_optimizer():
    config = optimization_sample()
    config["method"] = "adam"
    assert isinstance(keras_optimizer(config), Adam)
    config = optimization_sample()
    config["method"] = "adadelta"
    assert isinstance(keras_optimizer(config), Adadelta)
    config = optimization_sample()
    config["method"] = "adagrad"
    assert isinstance(keras_optimizer(config), Adagrad)
    config = optimization_sample()
    config["method"] = "rms"
    assert isinstance(keras_optimizer(config), RMSprop)
    config = optimization_sample()
    config["method"] = "sgd"
    assert isinstance(keras_optimizer(config), SGD)
    config["method"] = "missing"
    with pytest.raises(ValueError) as exc:
        keras_optimizer(config)
    assert (
        str(exc.value)
        == "optimizer['method'] must be one of 'adadelta', 'adam', 'adagram', 'rms', or 'sgd'"
    )


def load_example(batch_size):
    # load example data, generate random train/test split
    filepath = os.path.join(ROOT_TEST_DIR, "examples", "TCGA_glioma.csv")
    data = pd.read_csv(filepath, index_col=0)

    # retrieve protein expression features
    features = data.iloc[13:, :].to_numpy().T

    # get outcomes
    osr = data.iloc[[6, 5], :].to_numpy().T
    pfi = data.iloc[[12, 11], :].to_numpy().T

    # convert types
    features = features.astype(np.float32)
    osr = osr.astype(np.float32)
    pfi = pfi.astype(np.float32)

    # create a train/validation split
    np.random.seed(0)
    index = np.argsort(np.random.rand(data.shape[0]))
    train = np.zeros(features.shape[0], np.bool_)
    train[index[0 : int(0.8 * data.shape[0])].astype(np.int32)] = True

    # create tf.data.Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (features[train, :], {"osr": osr[train, :], "pfi": osr[train, :]})
    )
    train_dataset = train_dataset.shuffle(sum(train), reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(batch_size)
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (features[~train, :], {"osr": osr[~train, :], "pfi": pfi[~train, :]})
    )
    validation_dataset = validation_dataset.shuffle(
        train.size - sum(train), reshuffle_each_iteration=True
    )
    validation_dataset = validation_dataset.batch(batch_size)

    return train_dataset, validation_dataset


def test_search():
    # test experiment
    space = advmtl_space(
        412,
        tasks={
            "osr": task_space(loss={"efron", "cox"}),
            "pfi": task_space(loss={"efron", "cox"}),
        },
    )
    space["epochs"] = 10
    tuner = Search(space, advmtl_model, load_example, "osr_harrellsc")
    results = tuner.experiment(
        local_dir=ROOT_TEST_DIR, num_samples=1, max_concurrent_trials=1
    )

    # build a specific config
    def sample(config):
        for k in config.keys():
            if isinstance(config[k], dict):
                config[k] = sample(config[k])
            elif isinstance(
                config[k],
                (
                    ray.tune.search.sample.Integer,
                    ray.tune.search.sample.Float,
                    ray.tune.search.sample.Categorical,
                ),
            ):
                config[k] = config[k].sample()
            else:
                config[k] = config[k]
        return config

    # test trial
    space["builder"] = advmtl_model
    space["loader"] = load_example
    space["fit_kwargs"] = {}
    Search.trainable(sample(space))
