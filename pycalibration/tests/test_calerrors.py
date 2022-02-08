import math
import numpy as np
import pytest

from .. import ca


def test_ece():
    ece = ca.ECE(ca.UniformBinning(10))

    # simple calls (list of lists is converted to Matrix!)
    for predictions in (ca.RowVecs([[1, 0], [0, 1]]), ca.RowVecs(np.eye(2))):
        assert ece(predictions, [1, 2]) == 0
    assert ece(ca.ColVecs([[0, 0.5, 0.5, 1], [1, 0.5, 0.5, 0]]), [
               2, 2, 1, 1]) == 0

    # three-dimensional data
    rng = np.random.default_rng()
    predictions = ca.RowVecs(rng.dirichlet((5, 0.5), 1000))
    targets = rng.integers(1, 3, 1000)

    x = ece(predictions, targets)
    assert 0 < x < 1

    ece = ca.ECE(ca.MedianVarianceBinning(10))
    x = ece(predictions, targets)
    assert 0 < x < 1


def test_biasedskce():
    # binary example

    # categorical distributions (list of lists is converted to Matrix!)
    skce = ca.SKCE(ca.tensor(ca.SqExponentialKernel(), ca.WhiteKernel()), unbiased=False)
    for predictions in (ca.RowVecs([[1, 0], [0, 1]]), ca.RowVecs(np.eye(2))):
        assert skce(predictions, [1, 2]) == pytest.approx(0.0)
        assert skce(predictions, [1, 1]) == pytest.approx(0.5)
        assert skce(predictions, [2, 1]) == pytest.approx(1 - math.exp(-1))
        assert skce(predictions, [2, 2]) == pytest.approx(0.5)

    # probabilities
    skce = ca.SKCE(ca.tensor(ca.compose(ca.SqExponentialKernel(), ca.ScaleTransform(math.sqrt(2))), ca.WhiteKernel()), unbiased=False)
    assert skce([1, 0], [True, False]) == pytest.approx(0.0)
    assert skce([1, 0], [True, True]) == pytest.approx(0.5)
    assert skce([1, 0], [False, True]) == pytest.approx(1 - math.exp(-1))
    assert skce([1, 0], [False, False]) == pytest.approx(0.5)

    # multi-dimensional data
    skce = ca.SKCE(
        ca.tensor(
            ca.compose(ca.ExponentialKernel(), ca.ScaleTransform(0.1)),
            ca.WhiteKernel()
        ),
        unbiased=False
    )
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = ca.RowVecs(rng.dirichlet(np.ones(nclasses), 20))
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(skce(predictions, targets))

        assert min(estimates) > 0


def test_unbiasedskce():
    # binary example

    # categorical distributions (list of lists is converted to Matrix!)
    skce = ca.SKCE(
        ca.tensor(ca.SqExponentialKernel(), ca.WhiteKernel()),
    )
    for predictions in (ca.RowVecs([[1, 0], [0, 1]]), ca.RowVecs(np.eye(2))):
        assert skce(predictions, [1, 2]) == pytest.approx(0.0)
        assert skce(predictions, [1, 1]) == pytest.approx(0.0)
        assert skce(predictions, [2, 1]) == pytest.approx(- 2 * math.exp(-1))
        assert skce(predictions, [2, 2]) == pytest.approx(0.0)

    # probabilities
    skce = ca.SKCE(
        ca.tensor(
            ca.compose(ca.SqExponentialKernel(), ca.ScaleTransform(math.sqrt(2))),
            ca.WhiteKernel()
        )
    )
    assert skce([1, 0], [True, False]) == pytest.approx(0.0)
    assert skce([1, 0], [True, True]) == pytest.approx(0.0)
    assert skce([1, 0], [False, True]) == pytest.approx(- 2 * math.exp(-1))
    assert skce([1, 0], [False, False]) == pytest.approx(0.0)

    # average for multi-dimensional data
    skce = ca.SKCE(
        ca.tensor(
            ca.compose(ca.ExponentialKernel(), ca.ScaleTransform(0.1)),
            ca.WhiteKernel()
        )
    )
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = ca.RowVecs(rng.dirichlet(np.ones(nclasses), 20))
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(skce(predictions, targets))

        assert np.mean(estimates) == pytest.approx(0.0, abs=1e-2)


def test_blockunbiasedskce():
    # blocks of two samples
    skce = ca.SKCE(
        ca.tensor(ca.SqExponentialKernel(), ca.WhiteKernel()),
        blocksize=2
    )

    # categorical distributions (list of lists is converted to Matrix!)
    for predictions in (ca.RowVecs([[1, 0], [0, 1]]), ca.RowVecs(np.eye(2))):
        assert skce(predictions, [1, 2]) == pytest.approx(0.0)
        assert skce(predictions, [1, 1]) == pytest.approx(0.0)
        assert skce(predictions, [2, 1]) == pytest.approx(- 2 * math.exp(-1))
        assert skce(predictions, [2, 2]) == pytest.approx(0.0)

    # two predictions, ten times replicated
    predictions = ca.ColVecs(np.tile([[1, 0], [0, 1]], (1, 10)))
    assert skce(predictions, np.tile([1, 2], 10)) == pytest.approx(0.0)
    assert skce(predictions, np.tile([1, 1], 10)) == pytest.approx(0.0)
    assert skce(predictions, np.tile([2, 1], 10)
                ) == pytest.approx(- 2 * math.exp(-1))
    assert skce(predictions, np.tile([2, 2], 10)) == pytest.approx(0.0)

    # probabilities
    skce = ca.SKCE(
        ca.tensor(
            ca.compose(
                ca.SqExponentialKernel(), ca.ScaleTransform(math.sqrt(2)),
            ),
            ca.WhiteKernel()
        ),
        blocksize=2
    )
    assert skce([1, 0], [True, True]) == pytest.approx(0.0)
    assert skce([1, 0], [True, False]) == pytest.approx(0.0)
    assert skce([1, 0], [False, True]) == pytest.approx(- 2 * math.exp(-1))
    assert skce([1, 0], [False, False]) == pytest.approx(0.0)

    # two predictions, ten times replicated
    predictions = np.tile([1, 0], 10)
    assert skce(predictions, np.tile([True, True], 10)) == pytest.approx(0.0)
    assert skce(predictions, np.tile([True, False], 10)) == pytest.approx(0.0)
    assert skce(predictions, np.tile(
        [False, True], 10)) == pytest.approx(- 2 * math.exp(-1))
    assert skce(predictions, np.tile([False, False], 10)) == pytest.approx(0.0)

    # average for multi-dimensional data
    skce = ca.SKCE(
        ca.tensor(
            ca.compose(ca.ExponentialKernel(), ca.ScaleTransform(0.1)),
            ca.WhiteKernel()
        ),
        blocksize=2,
    )
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = ca.RowVecs(rng.dirichlet(np.ones(nclasses), 20))
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(skce(predictions, targets))

        assert np.mean(estimates) == pytest.approx(0.0, abs=1e-2)
