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

## Usage

Import and setup

- calibration errors for classification models with
  ```pycon
  >>> from pycalibration import calerrors
  ```
- additional support for general predictive probabilistic models with
  ```pycon
  >>> from pycalibration import calerrorsdists
  ```
- statistical hypothesis tests for calibration with
  ```pycon
  >>> from pycalibration import caltests
  ```

You can then do the same as would be done in Julia, except you have to add
`calerrors.`, `calerrorsdists.`, or `caltests.` in front for functionality
from CalibrationErrors.jl, CalibrationErrorsDistributions.jl, or
CalibrationTests.jl, respectively. Most of the commands will work without
any modification. Thus the documentaton of the Julia packages are the main
in-depth documentation for this package.

## References

If you use pycalibration as part of your research, teaching, or other activities,
please consider citing the following publications:

Widmann, D., Lindsten, F., & Zachariah, D. (2019). [Calibration tests in multi-class
classification: A unifying framework](https://proceedings.neurips.cc/paper/2019/hash/1c336b8080f82bcc2cd2499b4c57261d-Abstract.html). In
*Advances in Neural Information Processing Systems 32 (NeurIPS 2019)* (pp. 12257–12267).

Widmann, D., Lindsten, F., & Zachariah, D. (2021).
[Calibration tests beyond classification](https://openreview.net/forum?id=-bxf89v3Nx).
To be presented at *ICLR 2021*.
