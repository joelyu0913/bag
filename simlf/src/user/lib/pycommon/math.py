import numpy as np
from numba import guvectorize, njit


@njit()
def compute_cos(sig: np.array, ret: np.array) -> float:
    sig = sig.ravel()
    ret = ret.ravel()
    sum_xy = 0
    sum_x2 = 0
    sum_y2 = 0
    for i in range(len(sig)):
        x = sig[i]
        y = ret[i]
        if np.isfinite(x) and np.isfinite(y):
            sum_xy += x * y
            sum_x2 += x * x
            sum_y2 += y * y
    if sum_x2 == 0 or sum_y2 == 0:
        return np.nan
    return sum_xy / np.sqrt(sum_x2 * sum_y2)


def linreg(X, y, W=None, I=None):
    if W:
        U = X.T @ W @ X
        V = X.T @ W @ y
    else:
        U = X.T @ X
        V = X.T @ y
    if I:
        U += I
    return np.linalg.inv(U) @ V


@njit()
def ts_reg(start_di, end_di, univ, sig_out, sig1, sig2, window):
    for di in range(max(start_di, window), end_di):
        for ii in np.where(univ[di])[0]:
            x = sig1[di - window + 1 : di + 1, ii]
            y = sig2[di - window + 1 : di + 1, ii]
            mask = np.logical_and(np.isfinite(x), np.isfinite(y))
            if np.sum(mask) < window // 2:
                continue

            X = x[mask].copy().reshape(-1, 1)
            y = y[mask].copy()

            U = X.T @ X
            sig_out[di, ii] = (np.linalg.inv(U) @ X.T @ y)[0]


def similar(x_, y_):
    flag_neg = 1
    x = np.cumsum(x_)
    y = np.cumsum(y_)
    x -= x[0]
    y -= y[0]

    if (x[-1] < 0) ^ (y[-1] < 0):
        flag_neg *= -1

    x /= x[-1]
    y /= y[-1]
    delta_x = x[1:] - x[:-1]
    delta_y = y[1:] - y[:-1]
    return 1 / (flag_neg * np.mean(np.abs(delta_y - delta_x)) * np.sqrt(len(delta_x)))
