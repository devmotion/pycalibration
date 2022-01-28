import numpy as np

from .. import caltests as ct


def test_consistency():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ct.tensor(ct.compose(ct.ExponentialKernel(),
                                  ct.ScaleTransform(3)), ct.WhiteKernel())
    estimators = [ct.SKCE(kernel), ct.SKCE(kernel, unbiased=False)]

    for estimator in estimators:
        test = ct.ConsistencyTest(estimator, ct.RowVecs(
            predictions), targets_consistent)
        assert ct.pvalue(test) > 0.7
        print(test)

        test = ct.ConsistencyTest(
            estimator, ct.RowVecs(predictions), targets_onlytwo)
        assert ct.pvalue(test) < 1e-6
        print(test)


def test_distributionfree():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ct.tensor(ct.compose(ct.ExponentialKernel(),
                                  ct.ScaleTransform(3)), ct.WhiteKernel())
    estimators = [ct.SKCE(kernel, unbiased=False), ct.SKCE(
        kernel), ct.SKCE(kernel, blocksize=2)]

    for i, estimator in enumerate(estimators):
        test = ct.DistributionFreeSKCETest(estimator, ct.RowVecs(
            predictions), targets_consistent)
        assert ct.pvalue(test) > 0.7
        print(test)

        test = ct.DistributionFreeSKCETest(
            estimator, ct.RowVecs(predictions), targets_onlytwo)
        if i == 0:
            assert ct.pvalue(test) < 1e-6
        else:
            assert ct.pvalue(test) < 0.4
        print(test)


def test_asymptoticblockskce():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ct.tensor(ct.compose(ct.ExponentialKernel(),
                                  ct.ScaleTransform(3)), ct.WhiteKernel())

    for blocksize in (2, 10):
        test = ct.AsymptoticBlockSKCETest(kernel, blocksize, ct.RowVecs(
            predictions), targets_consistent)
        assert ct.pvalue(test) > 0.7
        print(test)

        test = ct.AsymptoticBlockSKCETest(
            kernel, blocksize, ct.RowVecs(predictions), targets_onlytwo)
        assert ct.pvalue(test) < 1e-6
        print(test)


def test_asymptoticskce():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    kernel = ct.tensor(ct.compose(ct.ExponentialKernel(),
                                  ct.ScaleTransform(3)), ct.WhiteKernel())

    test = ct.AsymptoticSKCETest(
        kernel, ct.RowVecs(predictions), targets_consistent)
    assert ct.pvalue(test) > 0.7
    print(test)

    test = ct.AsymptoticSKCETest(
        kernel, ct.RowVecs(predictions), targets_onlytwo)
    assert ct.pvalue(test) < 1e-6
    print(test)


def test_asymptoticcme():
    rng = np.random.default_rng(1234)
    predictions = rng.dirichlet((1, 1), 500)
    targets_consistent = 2 - (rng.random(500) < predictions[:, 0])
    targets_onlytwo = np.full(500, 2)

    testpredictions = rng.dirichlet((1, 1), 5)
    testtargets = rng.integers(low=1, high=3, size=5)

    kernel = ct.tensor(ct.compose(ct.ExponentialKernel(),
                                  ct.ScaleTransform(3)), ct.WhiteKernel())
    estimator = ct.UCME(kernel, ct.RowVecs(testpredictions), testtargets)

    test = ct.AsymptoticCMETest(
        estimator, ct.RowVecs(predictions), targets_consistent)
    assert ct.pvalue(test) > 0.7
    print(test)

    test = ct.AsymptoticCMETest(
        estimator, ct.RowVecs(predictions), targets_onlytwo)
    assert ct.pvalue(test) < 1e-6
    print(test)
