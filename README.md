# pycalibration

Estimation and hypothesis tests of calibration in Python using CalibrationErrors.jl and CalibrationTests.jl.

[![Stable](https://img.shields.io/badge/Julia%20docs-stable-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/stable)
[![Dev](https://img.shields.io/badge/Julia%20docs-dev-blue.svg)](https://devmotion.github.io/CalibrationErrors.jl/dev)
[![Status](https://github.com/devmotion/pycalibration/workflows/CI/badge.svg?branch=main)](https://github.com/devmotion/pycalibration/actions?query=workflow%3ACI+branch%3Amain)
[![codecov](https://codecov.io/gh/devmotion/pycalibration/branch/main/graph/badge.svg?token=URGY32W5EB)](https://codecov.io/gh/devmotion/pycalibration)
[![CalibrationAnalysis.jl Status](https://img.shields.io/github/workflow/status/devmotion/CalibrationAnalysis.jl/CI/main?label=CalibrationAnalysis.jl)](https://github.com/devmotion/CalibrationAnalysis.jl/actions?query=workflow%3ACI+branch%3Amain)
[![CalibrationErrors.jl Status](https://img.shields.io/github/workflow/status/devmotion/CalibrationErrors.jl/CI/main?label=CalibrationErrors.jl)](https://github.com/devmotion/CalibrationErrors.jl/actions?query=workflow%3ACI+branch%3Amain)
[![CalibrationTests.jl Status](https://img.shields.io/github/workflow/status/devmotion/CalibrationTests.jl/CI/main?label=CalibrationTests.jl)](https://github.com/devmotion/CalibrationTests.jl/actions?query=workflow%3ACI+branch%3Amain)

pycalibration is a package for estimating calibration of probabilistic models in Python.
It is a Python interface for the [CalibrationAnalysis.jl](https://github.com/devmotion/CalibrationAnalysis.jl)
suite and uses [CalibrationErrors.jl](https://github.com/devmotion/CalibrationErrors.jl) and
[CalibrationTests.jl](https://github.com/devmotion/CalibrationTests.jl) for its
computations. As such, the package allows the estimation of calibration errors (ECE and
SKCE) and statistical testing of the null hypothesis that a model is calibrated.

## Installation

To install `pycalibration`, run

```shell
python -m pip install git+https://github.com/devmotion/pycalibration.git
```

The use of `pycalibration` requires that its dependency
[`pyjulia`](https://github.com/JuliaPy/pyjulia) (installed automatically)
and itself are configured correctly.

For `pyjulia`, you have to
[install Julia](https://pyjulia.readthedocs.io/en/latest/installation.html#step-1-install-julia) (at least version 1.6 is required)
and the
[Julia dependencies of `pyjulia`](https://pyjulia.readthedocs.io/en/latest/installation.html#step-3-install-julia-packages-required-by-pyjulia).
The configuration process is described in detail in the
[`pyjulia` documentation](https://pyjulia.readthedocs.io/en/latest/installation.html).

When `pyjulia` is configured correctly, you can install the Julia packages required by
`pycalibration` in the Python interpreter:

```pycon
>>> import pycalibration
>>> pycalibration.install()
```

### Custom Julia environment

With the default settings, `pyjulia` and `pycalibration` install all Julia dependencies
in the default environment. In particular, if you use Julia for other projects as well,
a separate [project environment](https://pkgdocs.julialang.org/v1/environments/) can
simplify package management and ensure that the state of the Julia dependencies is
reproducible. In `pyjulia` and `pycalibration`, a custom project environment is used if
you set the environment variable `JULIA_PROJECT`:

```shell
export JULIA_PROJECT="path/to/the/environment/"
```

## Usage

Import and setup calibration analysis tools from CalibrationAnalysis.jl with
```pycon
>>> from pycalibration import ca
```

You can then do the same as would be done in Julia, except you have to add
`ca.` in front for functionality from CalibrationAnalysis.jl.
Most of the commands will work without any modification. Thus the documentation
of the Julia package is the main in-depth documentation for this package.

### Valid identifiers

Not all valid Julia identifiers are valid Python identifiers. This is an inherent
limitation of [Python and `pyjulia`](https://pyjulia.readthedocs.io/en/latest/limitations.html#mismatch-in-valid-set-of-identifiers). In particular, it is a common idiom in Julia to
append `!` to functions that mutate their arguments but it is not possible to use
`!` in function names in Python. `pyjulia` renames these functions by substituting
`!` with `_b`, e.g., you can call the Julia function `copy!` with `copy_b` in Python.

### Calibration errors

Let us estimate the squared kernel calibration error (SKCE) with the tensor
product kernel
```math
k((p, y), (p̃, ỹ)) = exp(-|p - p̃|) δ(y - ỹ)
```
from a set of predictions and corresponding observed outcomes.

```pycon
>>> skce = ca.SKCE(ca.tensor(ca.ExponentialKernel(), ca.WhiteKernel()))
```

Other estimators of the SKCE and estimators of other calibration errors such
as the expected calibration error (ECE) are available as well. The Julia package
[KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl)
supports a variety of kernels, all compositions and transformations of
[kernels available there](https://juliagaussianprocesses.github.io/KernelFunctions.jl/stable/kernels/)
can be used.

#### Sequences of probabilities

Predictions can be provided as sequences of probabilities. In this case, the
predictions correspond to Bernoulli distributions with these parameters and the
targets are boolean values.

```pycon
>>> import random
>>> random.seed(1234)
>>> predictions = [random.random() for _ in range(100)]
>>> outcomes = [bool(random.getrandbits(1)) for _ in range(100)]
>>> skce(predictions, outcomes)
0.028399084017414655
```

NumPy arrays are supported as well.

```pycon
>>> import numpy as np
>>> rng = np.random.default_rng(1234)
>>> predictions = rng.random(100)
>>> outcomes = rng.choice([True, False], 100)
>>> skce(predictions, outcomes)
0.03320398246523166
```

#### Sequences of probability vectors

Predictions can be provided as sequences of probability vectors (i.e., vectors
in the probability simplex) as well. In this case, the predictions correspond to categorical
distributions with these class probabilities and the targets are integers in `{1,...,n}`.

```pycon
>>> import numpy as np
>>> rng = np.random.default_rng(1234)
>>> predictions = [rng.dirichlet((3, 2, 5)) for _ in range(100)]
>>> outcomes = rng.integers(low=1, high=4, size=100)
>>> skce(predictions, outcomes)
0.02015240706950358
```

Sequences of probability vectors can also be provided as NumPy matrices. However, it is
required to specify if the probability vectors correspond to rows or columns of the matrix
by wrapping them in `ca.RowVecs` and `ca.ColVecs`, respectively. These wrappers are defined
in [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl).

```pycon
>>> import numpy as np
>>> rng = np.random.default_rng(1234)
>>> predictions = rng.dirichlet((3, 2, 5), 100)
>>> outcomes = rng.integers(low=1, high=4, size=100)
>>> skce(ca.RowVecs(predictions), outcomes)
0.02015240706950358
```

The wrappers have to be used also for, e.g., lists of lists since `pyjulia` converts them
to matrices automatically.

```pycon
>>> predictions = [[0.1, 0.8, 0.1], [0.2, 0.5, 0.3]]
>>> outcomes = [2, 3]
>>> skce(ca.RowVecs(predictions), outcomes)
-0.10317943453412069
```

#### Sequences of probability distributions

Predictions can also be provided as sequences of probability distributions defined in the
Julia package [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). Currently,
analytical formulas for the estimators of the SKCE and unnormalized calibration mean embedding
(UCME) are implemented for uni- and multivariate normal distributions `ca.Normal` and
`ca.MvNormal` with squared exponential kernels on the target space and Laplace distributions
`ca.Laplace` with exponential kernels on the target spaca.

In this example we use the tensor product kernel
```math
k((p, y), (p̃, ỹ)) = exp(-W₂(p, p̃)) exp(-(y - ỹ)²/2),
```
where `W₂(p, p̃)` is the 2-Wasserstein distance of the two normal distributions `p` and `p̃`.
It is given by
```math
W₂(p, p̃) = √((μ - μ̃)² + (σ - σ̃)²),
```
where `p = N(μ, σ)` and `p̃ = N(μ̃, σ̃)`.

```pycon
>>> import random
>>> random.seed(1234)
>>> predictions = [ca.Normal(random.gauss(0, 1), random.random()) for _ in range(100)]
>>> outcomes = [random.gauss(0, 1) for _ in range(100)]
>>> skce = ca.SKCE(ca.tensor(ca.ExponentialKernel(metric=ca.Wasserstein()), ca.SqExponentialKernel()))
>>> skce(predictions, outcomes)
0.02203618235964146
```

### Calibration tests

`pycalibration` provides different calibration tests that estimate the p-value of the null hypothesis
that a model is calibrated, based on a set of predictions and outcomes:
- `ca.ConsistencyTest` estimates the p-value with consistency resampling for a given calibration error estimator
- `ca.DistributionFreeSKCETest` computes distribution-free (and therefore usually quite weak) upper bounds of the p-value for different estimators of the SKCE
- `ca.AsymptoticBlockSKCETest` estimates the p-value based on the asymptotic distribution of the unbiased block estimator of the SKCE
- `ca.AsymptoticSKCETest` estimates the p-value based on the asymptotic distribution of the unbiased estimator of the SKCE
- `ca.AsymptoticCMETest` estimates the p-value based on the asymptotic distribution of the UCME

```pycon
>>> from pycalibration import caltests as ct
>>> import numpy as np
>>> rng = np.random.default_rng(1234)
>>> predictions = rng.dirichlet((3, 2, 5), 100)
>>> outcomes = rng.integers(low=1, high=4, size=100)
>>> kernel = ca.tensor(ca.ExponentialKernel(metric=ca.TotalVariation()), ca.WhiteKernel())
>>> test = ca.AsymptoticSKCETest(kernel, predictions, outcomes)
>>> print(test)
<PyCall.jlwrap Asymptotic SKCE test
--------------------
Population details:
    parameter of interest:   SKCE
    value under h_0:         0.0
    point estimate:          6.07887e-5

Test summary:
    outcome with 95% confidence: fail to reject h_0
    one-sided p-value:           0.4330

Details:
    test statistic: -4.955380469272125
>>> ca.pvalue(test)
0.435
```

## Citing

If you use pycalibration as part of your research, teaching, or other activities, please consider citing the following publications:

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html). In
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)* (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021).
[Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx).
*International Conference on Learning Representations (ICLR 2021)*.

## Acknowledgements

This work was financially supported by the Swedish Research Council via the projects *Learning of Large-Scale Probabilistic Dynamical Models* (contract number: 2016-04278), *Counterfactual Prediction Methods for Heterogeneous Populations* (contract number: 2018-05040), and *Handling Uncertainty in Machine Learning Systems* (contract number: 2020-04122), by the Swedish Foundation for Strategic Research via the project *Probabilistic Modeling and Inference for Machine Learning* (contract number: ICA16-0015), by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation, and by ELLIIT.
