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

## Contents

- [User guide](#user-guide)
- [Terminology](#terminology)
- [Hyperparater notation](#hyperparameter-notation)
- [Creating a search space](#search-space)
- [The builder function](#builder)
- [The data loader](#dataloader)

# User guide <a name="user-guide"></a>
## Terminology <a name="terminology"></a>

Ray Tune is a hyperparameter search library for building highly optmized models. A *hyperparameter* is any selected parameter used in the design or training of a model including but not limited to network architecture (depth, width, activations, topology), optimization (gradient algorithm, learning rate, scheduling), and other training parameters like losses and loss parameters. 

A *search space* is the range of allowable hyperparameter selections that lead to a trained model. Hyperparameters can be drawn from an interval of values, or selected from a discrete set. For example, a model can may be allowed to have between two and six layers with learning rate ranging from 1e-5 to 1e-3. Glimr provides a simple way to define search spaces using basic `list` and `set` python types. To create a search space, users define a nested dictionary that describes the choices required to build their model. Ray samples this space to generate *model configurations* that correspond to different models to evaluate for fitness (see below).

Each configuration sampled from the search space defines a *trial*. During the trial, the model is built and trained to specification, and the performance of this model is evaluated and recorded. A collection of these trials is called an *experiment*. A *search algorithm* is a strategy for selecting trials to run. This could be a random or grid search where each trial is independent and highly parallelizable, or a more intelligent approach like *Bayesian optimization* that tries to sequentially sample the best configurations based on the outcome of previous trials. A *scheduler* allows early termination of less promising trials so that resources can be devoted to exploring more promising configurations.

Additional discussion of these concepts can be found in the [key concepts](https://docs.ray.io/en/latest/tune/key-concepts.html) documentation of Ray Tune.

## Hyperparameter notation <a name="hyperparameter-notation"></a>

Glimr uses a simplified convention to represent the range of possible search spaces [provided by Ray Tune](https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space). Two options are offered:
1. `list` defines a uniformly-sampled interval of numerics like `int` or `float`.
2. `set` defines a random choice among discrete options like gradient descent algorithm.

The `list` notation defines the lower and upper interval range, with an optional quantization. Depending on whether `float` or `int` arguments are provided, this will map to the Ray Tune search space API functions [`ray.tune.uniform`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.uniform.html#ray.tune.uniform)/[`ray.tune.quniform`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.quniform.html#ray.tune.quniform) or [`ray.tune.randint`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.randint.html#ray.tune.randint)/[`ray.tune.qrandint`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.qrandint.html#ray.tune.qrandint). If no quantization is provided, the interval will be divided in ten segments.

```
# defines a parameter interval from 0. to 20. with quantization 2.
{"parameter": [0., 20.]}

# defines a parameter with the same interval but with quantization 1.
{"parameter": [0., 20., 1.]}
```

The `set` notation can be used with any types, with each option being equally likely to be selected. This maps to the [`ray.tune.choice`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.choice.html#ray.tune.choice) function from the Ray Tune search space API.

```
# choose between stochastic-gradient descent and adam
{"gradient_alg": {"sgd", "adam"}}

# select among log-spaced values for learning rate
{"learning_rate": {1e-5, 1e-4, 1e-3, 1e-2}}
```

**Notes:**
> Advanced users can override this notation and instead directly use the [Ray Tune search space API functions](https://docs.ray.io/en/latest/tune/api/search_space.html#tune-search-space). Mapping from `list` and `set` is performed by the [`set_hyperparameter()`](https://github.com/cooperlab/glimr/blob/abefc5820a873691d396001d43d883ce416d429b/glimr/search/utils.py#L239) function.

## Creating a search space <a name="search-space"></a>

## The builder function <a name="builder"></a>

## The data loader <a name="dataloader"></a>
