import math
import numpy as np
import pytest

from .. import calerrors as ce


def test_ece():
    ece = ce.ECE(ce.UniformBinning(10))

    # simple calls (list of lists is converted to Matrix!)
    for predictions in (ce.RowVecs([[1, 0], [0, 1]]), ce.RowVecs(np.eye(2))):
        assert ece(predictions, [1, 2]) == 0
    assert ece(ce.ColVecs([[0, 0.5, 0.5, 1], [1, 0.5, 0.5, 0]]), [
               2, 2, 1, 1]) == 0

    # three-dimensional data
    rng = np.random.default_rng()
    predictions = ce.RowVecs(rng.dirichlet((5, 0.5), 1000))
    targets = rng.integers(1, 3, 1000)

    x = ece(predictions, targets)
    assert 0 < x < 1

    ece = ce.ECE(ce.MedianVarianceBinning(10))
    x = ece(predictions, targets)
    assert 0 < x < 1


def test_biasedskce():
    # binary example

    # categorical distributions (list of lists is converted to Matrix!)
    skce = ce.SKCE(ce.tensor(ce.SqExponentialKernel(), ce.WhiteKernel()), unbiased=False)
    for predictions in (ce.RowVecs([[1, 0], [0, 1]]), ce.RowVecs(np.eye(2))):
        assert skce(predictions, [1, 2]) == pytest.approx(0.0)
        assert skce(predictions, [1, 1]) == pytest.approx(0.5)
        assert skce(predictions, [2, 1]) == pytest.approx(1 - math.exp(-1))
        assert skce(predictions, [2, 2]) == pytest.approx(0.5)

    # probabilities
    skce = ce.SKCE(ce.tensor(ce.compose(ce.SqExponentialKernel(), ce.ScaleTransform(math.sqrt(2))), ce.WhiteKernel()), unbiased=False)
    assert skce([1, 0], [True, False]) == pytest.approx(0.0)
    assert skce([1, 0], [True, True]) == pytest.approx(0.5)
    assert skce([1, 0], [False, True]) == pytest.approx(1 - math.exp(-1))
    assert skce([1, 0], [False, False]) == pytest.approx(0.5)

    # multi-dimensional data
    skce = ce.SKCE(
        ce.tensor(
            ce.compose(ce.ExponentialKernel(), ce.ScaleTransform(0.1)),
            ce.WhiteKernel()
        ),
        unbiased=False
    )
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = ce.RowVecs(rng.dirichlet(np.ones(nclasses), 20))
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(skce(predictions, targets))

        assert min(estimates) > 0


def test_unbiasedskce():
    # binary example

    # categorical distributions (list of lists is converted to Matrix!)
    skce = ce.SKCE(
        ce.tensor(ce.SqExponentialKernel(), ce.WhiteKernel()),
    )
    for predictions in (ce.RowVecs([[1, 0], [0, 1]]), ce.RowVecs(np.eye(2))):
        assert skce(predictions, [1, 2]) == pytest.approx(0.0)
        assert skce(predictions, [1, 1]) == pytest.approx(0.0)
        assert skce(predictions, [2, 1]) == pytest.approx(- 2 * math.exp(-1))
        assert skce(predictions, [2, 2]) == pytest.approx(0.0)

    # probabilities
    skce = ce.SKCE(
        ce.tensor(
            ce.compose(ce.SqExponentialKernel(), ce.ScaleTransform(math.sqrt(2))),
            ce.WhiteKernel()
        )
    )
    assert skce([1, 0], [True, False]) == pytest.approx(0.0)
    assert skce([1, 0], [True, True]) == pytest.approx(0.0)
    assert skce([1, 0], [False, True]) == pytest.approx(- 2 * math.exp(-1))
    assert skce([1, 0], [False, False]) == pytest.approx(0.0)

    # average for multi-dimensional data
    skce = ce.SKCE(
        ce.tensor(
            ce.compose(ce.ExponentialKernel(), ce.ScaleTransform(0.1)),
            ce.WhiteKernel()
        )
    )
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = ce.RowVecs(rng.dirichlet(np.ones(nclasses), 20))
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(skce(predictions, targets))

        assert np.mean(estimates) == pytest.approx(0.0, abs=1e-2)


def test_blockunbiasedskce():
    # blocks of two samples
    skce = ce.SKCE(
        ce.tensor(ce.SqExponentialKernel(), ce.WhiteKernel()),
        blocksize=2
    )

    # categorical distributions (list of lists is converted to Matrix!)
    for predictions in (ce.RowVecs([[1, 0], [0, 1]]), ce.RowVecs(np.eye(2))):
        assert skce(predictions, [1, 2]) == pytest.approx(0.0)
        assert skce(predictions, [1, 1]) == pytest.approx(0.0)
        assert skce(predictions, [2, 1]) == pytest.approx(- 2 * math.exp(-1))
        assert skce(predictions, [2, 2]) == pytest.approx(0.0)

    # two predictions, ten times replicated
    predictions = ce.ColVecs(np.tile([[1, 0], [0, 1]], (1, 10)))
    assert skce(predictions, np.tile([1, 2], 10)) == pytest.approx(0.0)
    assert skce(predictions, np.tile([1, 1], 10)) == pytest.approx(0.0)
    assert skce(predictions, np.tile([2, 1], 10)
                ) == pytest.approx(- 2 * math.exp(-1))
    assert skce(predictions, np.tile([2, 2], 10)) == pytest.approx(0.0)

    # probabilities
    skce = ce.SKCE(
        ce.tensor(
            ce.compose(
                ce.SqExponentialKernel(), ce.ScaleTransform(math.sqrt(2)),
            ),
            ce.WhiteKernel()
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
    skce = ce.SKCE(
        ce.tensor(
            ce.compose(ce.ExponentialKernel(), ce.ScaleTransform(0.1)),
            ce.WhiteKernel()
        ),
        blocksize=2,
    )
    rng = np.random.default_rng(1234)

    for nclasses in (2, 10, 100):
        estimates = []

        for _ in range(1_000):
            predictions = ce.RowVecs(rng.dirichlet(np.ones(nclasses), 20))
            targets = rng.integers(1, nclasses + 1, 20)
            estimates.append(skce(predictions, targets))

        assert np.mean(estimates) == pytest.approx(0.0, abs=1e-2)
