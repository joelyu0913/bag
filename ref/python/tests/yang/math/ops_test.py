import numpy as np
import math.ops


def test_rank():
    a = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]).astype("float32")
    expected = [
        [0, 0.25, 0.5, 0.75, 1],
        [0, 0.25, 0.5, 0.75, 1],
    ]
    assert np.array_equal(yang.math.ops.rank(a), expected)
    assert np.array_equal(yang.math.ops.rank(a, out=a), expected)


def test_demean():
    a = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]).astype("float32")
    expected = [
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
    ]
    assert np.array_equal(yang.math.ops.demean(a), expected)
    assert np.array_equal(yang.math.ops.demean(a, out=a), expected)


def test_group_demean():
    a = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]).astype("float32")
    g = np.array([
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ])
    expected = [
        [-1, 0, 1, -0.5, 0.5],
        [-1, 0, 1, -0.5, 0.5],
    ]
    assert np.array_equal(yang.math.ops.group_demean(a, g), expected)
    assert np.array_equal(yang.math.ops.group_demean(a, g, out=a), expected)


def test_filter():
    a = np.array([
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
    ]).astype("float32")
    m = np.array([
        [False, False, False, True, True],
        [False, False, False, True, True],
    ])
    expected = [
        [np.nan, np.nan, np.nan, 3, 4],
        [np.nan, np.nan, np.nan, 8, 9],
    ]
    assert np.array_equal(yang.math.ops.filter(a, m), expected, equal_nan=True)
    assert np.array_equal(yang.math.ops.filter(a, m, out=a), expected, equal_nan=True)


def test_hedge():
    a = np.array([
        [0, 1, 2, -3, -4, -5],
        [0, 1, 2, -3, -4, -5],
    ]).astype("float32")
    expected = [
        [0, -3, 2, 0, 0, -5],
        [0, -3, 2, 0, 0, -5],
    ]
    assert np.array_equal(yang.math.ops.hedge(a, 5, 1), expected)
    assert np.array_equal(yang.math.ops.hedge(a, 5, 1, out=a), expected)


def test_scale():
    a = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]).astype("float32")
    expected = [
        [0, 2, 4, 6, 8],
        [0, 2, 4, 6, 8],
    ]
    assert np.array_equal(yang.math.ops.scale(a, 20), expected)
    assert np.array_equal(yang.math.ops.scale(a, 20, out=a), expected)


def test_mean():
    a = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]).astype("float32")
    expected = [2, 2]
    assert np.array_equal(yang.math.ops.mean(a), expected)


def test_variance():
    a = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]).astype("float32")
    expected = [2, 2]
    assert np.array_equal(yang.math.ops.variance(a), expected)


def test_stdev():
    a = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ]).astype("float32")
    expected = np.sqrt([2, 2]).astype("float32")
    assert np.array_equal(yang.math.ops.stdev(a), expected)
