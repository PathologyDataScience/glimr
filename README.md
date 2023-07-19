# glimr
A simplified wrapper for hyperparameter search with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).

## Overview

Glimr was developed to provide hyperparameter tuning capabilities for [survivalnet](https://github.com/PathologyDataScience/survivalnet2), [mil](https://github.com/PathologyDataScience/mil), and other TensorFlow/keras-based machine learning packages. It simplifies the complexities of Ray Tune without compromising the ability of advanced users to control details of the tuning process. Glimr can be integrated into any application with three simple steps:

1. Define the search space of possible model configurations using the glimr hyperparameter notation
2. Implement a model-building function that creates a model, losses, and metrics from a configuration
3. Provide a dataloader function

## Installation

glimr is pip installable

```
pip install glimr
```

# User guide <a name="user-guide"></a>

## Contents

- [Terminology](#terminology)
- [Module overview](#overview)
- [The ray.tune.search hyperparameter API](#ray-api)
- [Creating a search space](#search-space)
- [The model-building function](#builder)
- [The data loader](#dataloader)
- [Next steps](#next-steps)


## Terminology <a name="terminology"></a>

Ray Tune is a hyperparameter search library for building highly optmized models. A *hyperparameter* is any selected parameter used in the design or training of a model including but not limited to network architecture (depth, width, activations, topology), optimization (gradient algorithm, learning rate, scheduling), and other training parameters like losses and loss parameters. 

A *search space* is the range of allowable hyperparameter selections that lead to a trained model. Hyperparameters can be drawn from an interval of values, or selected from a discrete set. For example, a model can may be allowed to have between two and six layers with learning rate ranging from 1e-5 to 1e-3. Glimr provides a simple way to define search spaces by creating a nested dictionary that describes the choices required to build their model. Ray samples this space to generate *model configurations* that correspond to different models to evaluate for fitness (see below).

Each configuration sampled from the search space defines a *trial*. During the trial, the model is built and trained to specification, and the performance of this model is evaluated and recorded. A collection of these trials is called an *experiment*. A *search algorithm* is a strategy for selecting trials to run. This could be a random or grid search where each trial is independent and highly parallelizable, or a more intelligent approach like *Bayesian optimization* that tries to sequentially sample the best configurations based on the outcome of previous trials. A *scheduler* allows early termination of less promising trials so that resources can be devoted to exploring more promising configurations.

Additional discussion of these concepts can be found in the [key concepts](https://docs.ray.io/en/latest/tune/key-concepts.html) documentation of Ray Tune.

## Module overview <a name="overview"></a>

Glimr consists of four modules:
1. `search` implements the `Search` class that is used to run trials and experiments. This class contains all the details of configuring Ray Tune checkpointing, reporting, and model training behavior. 
2. `utils` contains functions to help users build and work with search spaces. This module will be expanded in the future.
3. `keras` contains functions to help with model building logic, and simplify the process of building data structures that keras needs to compile models with losses, metrics, and optimization algorithms.
4. `optimization` implements a default search space for gradient optimizers.

## The ray.tune.search hyperparameter API]<a name="ray-api"></a>

Glimr uses the [Ray Tune search space API](https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space) to represent hyperparameters. This API provides a set of functions for defining range/interval hyperparameters for things like learning rate or layers/units as well as discrete choice parameters for things like gradient optimization. Commonly used functions for `float` or `int` range parameters include [`ray.tune.uniform`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.uniform.html#ray.tune.uniform)/[`ray.tune.quniform`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.quniform.html#ray.tune.quniform) or [`ray.tune.randint`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.randint.html#ray.tune.randint)/[`ray.tune.qrandint`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.qrandint.html#ray.tune.qrandint). 

```python
# a uniformly distributed float hyperparameter in the range [0., 1.]
hp1 = ray.tune.uniform(0., 1.)

# a uniformly distributed float hyperparameter in the range [0., 1.] with quantization 0.1
hp2 = ray.tune.quniform(0., 1., 0.1)

# a uniformly distributed int hyperparameter in the range [0, 10] (quantization 1)
hp3 = ray.tune.randint(0, 10, 1)

#  a uniformly distributed int hyperparameter in the range [0, 10] with quantization 2
hp4 = ray.tune.qrandint(0, 10, 2)
```

Other options for range parameters include logarithmic interval spacing.

For discrete hyperparameter [`ray.tune.choice`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.choice.html#ray.tune.choice) provides random sampling.

```python
# choose between stochastic-gradient descent and adam
gradient_alg = tune.choice(["sgd", "adam"])
```

## Creating a search space <a name="search-space"></a>

A search space is simply a dictionary of hyperparameters and other fixed values. Dictionaries can be nested to improve organiziation by separating elements of the space. For example, here is a hypothetical search space for a simple two-layer network that searches the units, activation functions, and dropout for each layer, along with training hyperparameters:

```python
search = {
  layer1: {
    activation: tune.choice(["relu", "gelu"]),
    dropout: tune.quniform(0.0, 0.5, 0.05),
    units: tune.choice([64, 48, 32, 16]),
  }
  layer2: {
    activation: tune.choice(["relu", "gelu"]),
    dropout: tune.quniform(0.0, 0.5, 0.05),
    units: tune.choice([64, 48, 32, 16]),
  }
  optimization: {
    batch= tune.choice([32, 64, 128]),
    method= tune.choice(["rms", "sgd"]),
    learning_rate= tune.quniform(1e-5, 1e-2, 1e-5),
    momentum= tune.quniform(0.0, 1e-1, 1e-2)
  }
  data: {
    "batch_size": tune.choice([32, 64, 128])
  }
}
```

To set a fixed hyperparameter simply assign this value without the search API function:

```python
layer2: {
  activation: "relu",
  dropout: 0.5,
  units: 32,
}
```

**Note:**
>Configuration values must be pickleable for Ray Tune to pass configurations to its workers. For this reason, callables like losses or metrics that are not pickleable have to be encoded as strings and decoded in the model building function inside the worker process. In the simple example above, that leads us to use `sgd` instead of `tf.keras.optimizers.experimental.SGD`. When we write the model builder function, we can decode the string to produce the optmizer object using a dict to map between strings and their objects.

### Required elements

Glimre is flexible and can be used with a wide variety of models, due to the user-defined configurations and model builder functions. There are some required elements of the search space, however. These constraints are implemented in [`glimr.Search.trainiable()`](https://github.com/PathologyDataScience/survivalnet2/blob/1b3c2ac4d6866e3eabdbc85063cd20df62aed292/survivalnet2/search/search.py#L297) which is used to build run each trial.

#### Tasks
Terminal model outputs/layers must be captured as values in a `tasks` key that is located at the top level of the space. This is required to support multi-task models and to correctly assign metrics and losses to model outputs. Each key in `tasks` defines a task name, and each task requires a loss, loss weight, and metric. For example, from our model above, we can define the second layer as a task:

```python

search = {
  layer1: {
    activation: tune.choice(["relu", "gelu"]),
    dropout: tune.quniform(0.0, 0.5, 0.05),
    units: tune.choice([64, 48, 32, 16]),
  }
  tasks: {
    task1: {
      activation: tune.choice(["relu", "gelu"]),
      dropout: tune.quniform(0.0, 0.5, 0.05),
      units: tune.choice([64, 48, 32, 16]),
      loss={
          "name": "binary_crossentropy",
          "loss": tf.keras.losses.BinaryCrossentropy
      },
      loss_weight=1.0,
      metrics=[{
          "name": "f1",
          "metric": tf.keras.metrics.F1Score
          "kwargs": {"threshold": 0.25},
        }]
      }
    }
  }
  optimization: {
    epochs = 100,
    method= tune.choice(["rms", "sgd"]),
    learning_rate= tune.quniform(1e-5, 1e-2, 1e-5),
    momentum= tune.quniform(0.0, 1e-1, 1e-2)
  }
  data: {
    "batch_size": tune.choice([32, 64, 128])
  }
}
```

Here, `metrics=[{"name": "f1",...` defines a metric that will be registered and displayed as `f1` during model compilation, and the metric `tf.keras.metrics.F1Score` will be used with a object for compilation. The `kwargs` allows customization of things like thresholds for metrics and losses that are classes. Keyword arguments cannot be used with non-class metrics or losses.

#### Data

The `data` value contains keyword arguments that are passed to the dataloader. At a minimum this must include `batch_size`, but can also include parameters to control data loading (e.g. prefetch) and preprocessing steps like training data augmentation. A typical data search space will combine constants like data paths with hyperparameters for things like data preprocessing steps and parameters.

#### Optimization

The `optimization` value represents gradient optimizaton hyperparameters including `method`, `learning_rate`, and `batch` size. The optimization configuration is decoded by [`keras_optimizer`](https://github.com/cooperlab/glimr/blob/47ca4e58da1296805557947f78afd8acf533005d/glimr/search/utils.py#L210) to produce an optimizer object for model compilaton.

The maximum number of epochs is encoded as `epochs` in `optimization`. The maximum number of epochs is also controllable through the class attribute `Search.stopper`. A trial may terminate early if the model converges to the stopping criteria, or if the scheduler determines it is performing poorly.

## The model-building function <a name="builder"></a>

The model builder is a user-defined function that takes a configuration as a single argument. This configuration is a sample from the search space, and represents specific choices of hyperparameter values to use in building and training a model. The builder should return `model`, `losses`, `loss_weights`, and `metrics` values for use with `tf.keras.Model.compile`.

### Helper functions

We provide helper functions in `glimr.utils` to aid in the model building process. [`utils.keras_losses()`](https://github.com/cooperlab/glimr/blob/4217444a61381decbf14fc7bb81c5056a0f1f4e5/glimr/search/utils.py#L70) can be used to generate a dictionary of output name / loss pairs consistent with the naming expected by `tf.keras.Model.compile()`. [`utils.keras_metrics()`](https://github.com/cooperlab/glimr/blob/4217444a61381decbf14fc7bb81c5056a0f1f4e5/glimr/search/utils.py#L166) performs a similar function for metrics.

### Naming model outputs

Since tasks represent model outputs, each task has to provide a named output that can be used for registering metrics and losses in model compilation:

```python
task_layer = tf.keras.layers.Dense(units, activation=activation, name="task1")
```

Furthermore, these names have to be used when creating the `tf.keras.Model` object

```python
named = {f"{name}": task for (name, task) in zip(config["tasks"].keys(), tasks)}
model = tf.keras.Model(inputs=inputs, outputs=named)
```

See the examples for more details.

## The data loader <a name="dataloader"></a>

The dataloader is a simple function that generates two `tf.data.Dataset` objects used for training and validation. The training object is used to build models and the valdiation object is used to report accuracy to Ray Tune for search, checkpointing, and early termination.

### Label names

Labels must be structured in a dictionary with task names for correct linking to the model tasks:

```python
train_dataset = tf.data.Dataset.from_tensor_slices((features[train, :], {'task1': labels[train]}))
```

## Next steps <a name="next-steps"></a>

The `Search` class provides the interface for running experiments, reporting outcomes, and saving models. Check out the example notebooks for using this class with a concrete example of the concepts described above.
