import math
import numpy as np
import pytest

from .. import calerrors


def test_ece():
    ece = calerrors.ECE(calerrors.UniformBinning(10))

    # simple calls
    assert calerrors.calibrationerror(ece, [[1, 0], [0, 1]], [1, 2]) == 0
    assert calerrors.calibrationerror(ece, ([[1, 0], [0, 1]], [1, 2])) == 0
    assert calerrors.calibrationerror(
        ece, [[0, 0.5, 0.5, 1], [1, 0.5, 0.5, 0]], [2, 2, 1, 1]) == 0
    assert calerrors.calibrationerror(
        ece, ([[0, 0.5, 0.5, 1], [1, 0.5, 0.5, 0]], [2, 2, 1, 1])) == 0

    # three-dimensional data
    rng = np.random.default_rng()
    predictions = rng.dirichlet((1, 5, 0.5), 1000).transpose()
    targets = rng.integers(1, 3, 1000)

    x = calerrors.calibrationerror(ece, predictions, targets)
    assert 0 < x < 1

    ece = calerrors.ECE(calerrors.MedianVarianceBinning(10))
    x = calerrors.calibrationerror(ece, predictions, targets)
    assert 0 < x < 1


def test_biasedskce():
    skce = calerrors.BiasedSKCE(
        calerrors.SqExponentialKernel(),
        calerrors.WhiteKernel())

    assert calerrors.calibrationerror(
        skce, ([[1, 0], [0, 1]], [1, 2])) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce, [[1, 0], [0, 1]], [1, 1]) == pytest.approx(0.5)
    assert calerrors.calibrationerror(
        skce, [[1, 0], [0, 1]], [2, 1]) == pytest.approx(1 - math.exp(-1))
    assert calerrors.calibrationerror(
        skce, ([[1, 0], [0, 1]], [2, 2])) == pytest.approx(0.5)

    # multi-dimensional data
    skce = calerrors.BiasedSKCE(
        calerrors.transform(calerrors.ExponentialKernel(), 0.1),
        calerrors.WhiteKernel())
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = rng.dirichlet(np.ones(nclasses), 20).transpose()
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(calerrors.calibrationerror(
                skce, predictions, targets))

        assert min(estimates) > 0


def test_unbiasedskce():
    skce = calerrors.UnbiasedSKCE(
        calerrors.SqExponentialKernel(),
        calerrors.WhiteKernel())

    assert calerrors.calibrationerror(
        skce, ([[1, 0], [0, 1]], [1, 2])) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce, [[1, 0], [0, 1]], [1, 1]) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce, [[1, 0], [0, 1]], [2, 1]) == pytest.approx(- 2 * math.exp(-1))
    assert calerrors.calibrationerror(
        skce, ([[1, 0], [0, 1]], [2, 2])) == pytest.approx(0.0)

    # average for multi-dimensional data
    skce = calerrors.UnbiasedSKCE(
        calerrors.transform(calerrors.ExponentialKernel(), 0.1),
        calerrors.WhiteKernel())
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = rng.dirichlet(np.ones(nclasses), 20).transpose()
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(calerrors.calibrationerror(
                skce, predictions, targets))

        assert np.mean(estimates) == pytest.approx(0.0, abs=1e-2)


def test_blockunbiasedskce():
    # blocks of two samples
    skce = calerrors.BlockUnbiasedSKCE(
        calerrors.SqExponentialKernel(),
        calerrors.WhiteKernel())

    # only two predictions, i.e., one term in the estimator
    assert calerrors.calibrationerror(
        skce, ([[1, 0], [0, 1]], [1, 2])) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce, [[1, 0], [0, 1]], [1, 1]) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce, [[1, 0], [0, 1]], [2, 1]) == pytest.approx(- 2 * math.exp(-1))
    assert calerrors.calibrationerror(
        skce, ([[1, 0], [0, 1]], [2, 2])) == pytest.approx(0.0)

    # two predictions, ten times replicated
    assert calerrors.calibrationerror(
        skce,
        (np.tile([[1, 0], [0, 1]], (1, 10)),
         np.tile([1, 2], 10))) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce,
        np.tile([[1, 0], [0, 1]], (1, 10)),
        np.tile([1, 1], 10)) == pytest.approx(0.0)
    assert calerrors.calibrationerror(
        skce,
        np.tile([[1, 0], [0, 1]], (1, 10)),
        np.tile([2, 1], 10)) == pytest.approx(- 2 * math.exp(-1))
    assert calerrors.calibrationerror(
        skce,
        (np.tile([[1, 0], [0, 1]], (1, 10)),
         np.tile([2, 2], 10))) == pytest.approx(0.0)

    # average for multi-dimensional data
    skce = calerrors.BlockUnbiasedSKCE(
        calerrors.transform(calerrors.ExponentialKernel(), 0.1),
        calerrors.WhiteKernel())
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = rng.dirichlet(np.ones(nclasses), 20).transpose()
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(calerrors.calibrationerror(
                skce, predictions, targets))

        assert np.mean(estimates) == pytest.approx(0.0, abs=1e-2)
