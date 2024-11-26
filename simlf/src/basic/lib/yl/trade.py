import sys

import numpy as np


def trade_ts(ts_sig, mp, algo=0):
    if algo == 0:
        thd = mp["thd"]
        ts_sig[np.abs(ts_sig) < thd] = 0.0
        ts_sig[ts_sig >= thd] = 1.0
        ts_sig[ts_sig <= -thd] = -1.0
    elif algo == 1:
        thd = mp["thd"]
        ts_sig /= thd
        ts_sig[ts_sig >= 1.0] = 1.0
        ts_sig[ts_sig <= -1.0] = -1.0
    elif algo == 2:
        thd = mp["thd"]
        thd_hold = mp["thd_hold"]
        for i in range(len(ts_sig)):
            if ~np.isfinite(ts_sig[i]):
                continue
            if ts_sig[i] <= -thd:
                ts_sig[i] = -1.0
            elif ts_sig[i] >= thd:
                ts_sig[i] = 1.0
            elif i > 0 and ts_sig[i - 1] > 0.0 and ts_sig[i] > thd_hold:
                ts_sig[i] = ts_sig[i - 1]
            elif i > 0 and ts_sig[i - 1] < 0.0 and ts_sig[i] < -thd_hold:
                ts_sig[i] = ts_sig[i - 1]
            else:
                ts_sig[i] = 0.0
    elif algo == 3:
        thd = mp["thd"]
        thd_hold = mp["thd_hold"]
        for i in range(len(ts_sig)):
            if ~np.isfinite(ts_sig[i]):
                continue
            if ts_sig[i] <= -thd:
                ts_sig[i] = -1.0
            elif ts_sig[i] >= thd:
                ts_sig[i] = 1.0
            elif i > 0 and ts_sig[i - 1] > 0.0 and ts_sig[i] > thd_hold:
                ts_sig[i] = ts_sig[i - 1]
            elif i > 0 and ts_sig[i - 1] < 0.0 and ts_sig[i] < -thd_hold:
                ts_sig[i] = ts_sig[i - 1]
            elif i > 0 and ts_sig[i - 1] > 0.0 and ts_sig[i] > 0.0:
                ts_sig[i] = ts_sig[i - 1] / 2.0
            elif i > 0 and ts_sig[i - 1] < 0.0 and ts_sig[i] < 0.0:
                ts_sig[i] = ts_sig[i - 1] / 2.0
            else:
                ts_sig[i] = 0.0


def trade(sim, sid, path, mp, algo=0):
    if isinstance(path, str):
        sig = sim.load_mod(path).copy()
    else:
        sig = path.copy()

    if isinstance(sid, str):
        sids = [sid]
    else:
        sids = sid

    for sid_ in sids:
        ii = sim.univ.find(sid_.upper())
        ts_sig = sig[:, ii]
        trade_ts(ts_sig, mp, algo)

    return sig
