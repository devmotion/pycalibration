# pycalibration

Estimation and hypothesis tests of calibration in Python using CalibrationErrors.jl,
CalibrationErrorsDistributions.jl, and CalibrationTests.jl.

![Python test](https://github.com/devmotion/pycalibration/workflows/Python%20test/badge.svg?branch=master)

pycalibration is a package for estimating calibration of probabilistic models in Python.
It uses [CalibrationErrors.jl](https://github.com/devmotion/CalibrationErrors.jl),
[CalibrationErrorsDistributions.jl](https://github.com/devmotion/CalibrationErrorsDistributions.jl),
and [CalibrationTests.jl](https://github.com/devmotion/CalibrationTests.jl) for its
computations. As such, the package allows the estimation of calibration errors (ECE and
SKCE) and statistical testing of the null hypothesis that a model is calibrated.

## Installation

To install `pycalibration`, run

```shell
python -m pip install git+https://github.com/devmotion/pycalibration.git
```

The use of `pycalibration` requires that Julia is installed and in the path, along
with CalibrationErrors.jl, CalibrationErrorsDistributions.jl, CalibrationTests.jl, and
PyCall.jl. To install Julia, [download a generic binary](https://julialang.org/downloads/)
and add it to your path. Then open up a Python interpreter and install the Julia packages
required by `pycalibration`:

```pycon
>>> import pycalibration
>>> pycalibration.install()
```
