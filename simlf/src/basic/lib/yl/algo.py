import os
import sys

import numba
import numpy as np
import pandas as pd
import tsfresh.feature_extraction.feature_calculators as tsc
from numba import njit
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.distribution import MultiprocessingDistributor

from yao.lib.pycommon.math import compute_cos

tsf_ftypes = [
    "agg_autocorrelation",
    "agg_linear_trend",
    "ar_coefficient",
    "augmented_dickey_fuller",
    "cwt_coefficients",
    "energy_ratio_by_chunks",
    "fft_aggregated",
    "fft_coefficient",
    "friedrich_coefficients",
    "index_mass_quantile",
    "linear_trend",
    "linear_trend_timewise",
    "matrix_profile",
    "partial_autocorrelation",
    "query_similarity_count",
    "spkt_welch_density",
    "symmetry_looking",
]


def tsf(ftype, sim, sig, isig, start_di, end_di, window, pars={}, sids=[], fillna=False):
    func = getattr(tsc, ftype)
    if len(sids) == 0:
        iis = range(sim.univ_size)
    else:
        iis = [sim.univ.find(sid) for sid in sids]

    if fillna:
        isig = isig.copy()
        isig[start_di:end_di, : sim.univ_size][
            ~np.isfinite(isig[start_di:end_di, : sim.univ_size])
        ] = 0.0

    if ftype in tsf_ftypes:
        for ii in iis:
            for di in range(max(window - 1, start_di), end_di):
                sig[di, ii] = list(func(isig[di - window + 1 : di + 1, ii], [pars]))[0][1]
    else:
        for ii in iis:
            for di in range(max(window - 1, start_di), end_di):
                sig[di, ii] = func(isig[di - window + 1 : di + 1, ii], **pars)


def tsf_univ(ftype, sim, sig, isig, start_di, end_di, window, pars={}, univ=None, fillna=False):
    func = getattr(tsc, ftype)
    if univ is None:
        univ = sim.load_base("univ_all")
        iis = range(sim.univ_size)

    if fillna:
        isig = isig.copy()
        isig[start_di:end_di, : sim.univ_size][
            ~np.isfinite(isig[start_di:end_di, : sim.univ_size])
        ] = 0.0

    if ftype in tsf_ftypes:
        for ii in iis:
            for di in range(max(window - 1, start_di), end_di):
                if univ[di, ii]:
                    sig[di, ii] = list(func(isig[di - window + 1 : di + 1, ii], [pars]))[0][1]
    else:
        for ii in iis:
            for di in range(max(window - 1, start_di), end_di):
                if univ[di, ii]:
                    sig[di, ii] = func(isig[di - window + 1 : di + 1, ii], **pars)


def tsf_features(sid, ts, window, mp):
    df = pd.DataFrame({"id": [sid] * len(ts), "ts": ts})
    df_rolled = roll_time_series(
        df, column_id="id", max_timeshift=window - 1, min_timeshift=window - 1, n_jobs=5
    )
    df_rolled = df_rolled[["id", "ts"]]

    return extract_features(df_rolled, column_id="id", n_jobs=5, default_fc_parameters=mp)


@njit()
def _pick_corr(list_ts, thd):
    ids = [0]
    for i in range(1, len(list_ts)):
        ts_ = list_ts[i]
        flag_add = True

        for j in ids:
            if compute_cos(ts_, list_ts[j]) > thd:
                flag_add = False
                break

        if flag_add:
            ids.append(i)
    return ids


def pick_corr(list_ts, thd):
    return _pick_corr(numba.typed.List(list_ts), thd)
