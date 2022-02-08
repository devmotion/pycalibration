import numpy as np

from .. import ca


def test_consistency():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ca.tensor(ca.compose(ca.ExponentialKernel(),
                                  ca.ScaleTransform(3)), ca.WhiteKernel())
    estimators = [ca.SKCE(kernel), ca.SKCE(kernel, unbiased=False)]

    for estimator in estimators:
        test = ca.ConsistencyTest(estimator, ca.RowVecs(
            predictions), targets_consistent)
        assert ca.pvalue(test) > 0.7
        print(test)

        test = ca.ConsistencyTest(
            estimator, ca.RowVecs(predictions), targets_onlytwo)
        assert ca.pvalue(test) < 1e-6
        print(test)


def test_distributionfree():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ca.tensor(ca.compose(ca.ExponentialKernel(),
                                  ca.ScaleTransform(3)), ca.WhiteKernel())
    estimators = [ca.SKCE(kernel, unbiased=False), ca.SKCE(
        kernel), ca.SKCE(kernel, blocksize=2)]

    for i, estimator in enumerate(estimators):
        test = ca.DistributionFreeSKCETest(estimator, ca.RowVecs(
            predictions), targets_consistent)
        assert ca.pvalue(test) > 0.7
        print(test)

        test = ca.DistributionFreeSKCETest(
            estimator, ca.RowVecs(predictions), targets_onlytwo)
        if i == 0:
            assert ca.pvalue(test) < 1e-6
        else:
            assert ca.pvalue(test) < 0.4
        print(test)


def test_asymptoticblockskce():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ca.tensor(ca.compose(ca.ExponentialKernel(),
                                  ca.ScaleTransform(3)), ca.WhiteKernel())

    for blocksize in (2, 10):
        test = ca.AsymptoticBlockSKCETest(kernel, blocksize, ca.RowVecs(
            predictions), targets_consistent)
        assert ca.pvalue(test) > 0.7
        print(test)

        test = ca.AsymptoticBlockSKCETest(
            kernel, blocksize, ca.RowVecs(predictions), targets_onlytwo)
        assert ca.pvalue(test) < 1e-6
        print(test)


def test_asymptoticskce():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ca.tensor(ca.compose(ca.ExponentialKernel(),
                                  ca.ScaleTransform(3)), ca.WhiteKernel())

    test = ca.AsymptoticSKCETest(
        kernel, ca.RowVecs(predictions), targets_consistent)
    assert ca.pvalue(test) > 0.7
    print(test)

    test = ca.AsymptoticSKCETest(
        kernel, ca.RowVecs(predictions), targets_onlytwo)
    assert ca.pvalue(test) < 1e-6
    print(test)


def test_asymptoticcme():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    testpredictions = rng.dirichlet((1, 1), 5)
    testtargets = rng.integers(low=1, high=3, size=5)

    kernel = ca.tensor(ca.compose(ca.ExponentialKernel(),
                                  ca.ScaleTransform(3)), ca.WhiteKernel())
    estimator = ca.UCME(kernel, ca.RowVecs(testpredictions), testtargets)

    test = ca.AsymptoticCMETest(
        estimator, ca.RowVecs(predictions), targets_consistent)
    assert ca.pvalue(test) > 0.7
    print(test)

    test = ca.AsymptoticCMETest(
        estimator, ca.RowVecs(predictions), targets_onlytwo)
    assert ca.pvalue(test) < 1e-6
    print(test)
