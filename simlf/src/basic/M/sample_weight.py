import numpy as np


def equal_weight(seq_len):
    return np.ones(seq_len)


def shock_weight(seq_len):
    wt = np.ones(seq_len)
    wt[-60:] = 5
    return wt


def shock_weight2(seq_len):
    wt = np.ones(seq_len)
    wt[-20:] = 10
    wt[-40:-20] = 5
    wt[-80:-40] = 3
    wt[-160:-80] = 2
    return wt


def shock_weight3(seq_len):
    wt = np.ones(seq_len)
    wt[-60:] = 10
    return wt
