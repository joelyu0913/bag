import logging
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from data import Array, BlockMat, RefStructArray
from sim import Env#, ExprRunner

# from basic.lib.pnl import OperationManager
from basic.lib.pycommon.data import format_date


def mat_diff(a, b):

    mask_ = np.logical_and(np.isfinite(a), np.isfinite(b))
    diff_ = np.logical_and(a != b, mask_)
    nan_ = np.logical_xor(np.isfinite(a), np.isfinite(b))
    return (
        np.sum(diff_),
        np.sum(nan_),
        {"diff": np.where(diff_), "nan": np.where(np.logical_xor(np.isfinite(a), np.isfinite(b)))},
    )


def get_months(dates):
    tmp = [dates[0] // 1000000]

    for date in dates[1:]:
        yyyymm = date // 1000000
        if yyyymm != tmp[-1]:
            tmp.append(yyyymm)
    return tmp


def apply_ops(module_, sim, alpha):
    if "ops" in module_.config and module_.config["ops"] != "":
        alpha_op = sim.write_mod(f"{module_.config['name']}/b_sig_op")
        alpha_op[sim.start_di : sim.end_di, :] = sim.apply_ops(
            alpha, ops=module_.config["ops"], start_di=sim.start_di, end_di=sim.end_di
        )[sim.start_di : sim.end_di, :]


class SimPlot:
    def __init__(self):
        pass

    def set(self, sim, start_date=-1, end_date=-1, y_stride=0.1):
        x = [str(_) for _ in sim.dates.items]

        if start_date == -1:
            self.start_date = sim.dates[0]
        else:
            self.start_date = start_date

        if end_date == -1:
            self.end_date = sim.dates[-1]
        else:
            self.end_date = end_date

        start_di = sim.dates.lower_bound(self.start_date)
        end_di = sim.dates.upper_bound(self.end_date)

        self.x_dates = x[start_di:end_di]

        plt.rc("font", size=25)
        plt.figure(figsize=(24, 16))

        x_stride = len(self.x_dates) // 30
        plt.xticks(ticks=range(0, len(self.x_dates), x_stride), rotation=90)
        self.ax = plt.subplot(111)
        self.ax.spines.right.set_visible(False)
        self.ax.spines.top.set_visible(False)
        self.y_stride = y_stride
        self.max_range = 0

    def plot(self, sim, mod_path, **kwargs):
        start_di = sim.dates.lower_bound(self.start_date)
        end_di = sim.dates.upper_bound(self.end_date)
        # kwargs["start_di"] = start_di
        # kwargs["end_di"] = end_di
        kwargs["start_date"] = self.start_date
        kwargs["end_date"] = self.end_date
        if "color" in kwargs:
            color = kwargs["color"]
            kwargs.pop("color")
        else:
            color = "red"

        _ = sim.pnl3.compute_pnls(mod_path, **kwargs)

        y = np.cumsum(_[0]) / sim.pnl3.default_info["booksize"]

        pnl = y[0 : end_di - start_di]

        self.max_range = max(self.max_range, int(pnl[end_di - start_di - 1] / self.y_stride) + 2)
        self.ax.plot(self.x_dates, pnl, color=color, linewidth=2)
        self.update_y()

    def update_y(self):
        range_in = range(-1, self.max_range)
        for t in np.array(range_in) * self.y_stride:
            self.ax.axhline(y=t, color="black", linewidth=0.3)
        plt.yticks(ticks=np.array(range_in) * self.y_stride)


class Sim:
    def __init__(self, config):

        if "cache_dir" in config:
            config["sys_cache"] = os.path.join(config["cache_dir"], "sys")
            config["user_cache"] = os.path.join(config["cache_dir"], "user")

        self.sys_cache = config["sys_cache"]

        if "user_cache" not in config:
            self.user_cache = self.sys_cache
            config["user_cache"] = self.user_cache
        else:
            self.user_cache = config["user_cache"]
        self.env = Env(config, False)
        self.univ = self.env.univ
        self.dates = self.env.dates
        self.dates_size = self.env.dates_size
        self.univ_size = self.env.univ_size
        self.max_univ_size = self.env.max_univ_size
        self.max_dates_size = self.env.max_dates_size
        self.start_di = self.env.start_di
        self.end_di = self.env.end_di
        # add env
        # self.taq_tiems

        self.base_dirs = ["base", "sup_univ", "ibase"]
        # rewrite dates.find

    def _full_path(self, mod_path):
        return self.env.cache_dir.get_path(mod_path)

    def get_cache_path(self, mod_path, cache_type="both"):
        if cache_type == "sys":
            p = os.path.join(self.sys_cache, mod_path)
            if os.path.exists(p):
                return p
            else:
                return None
        elif cache_type == "user":
            p = os.path.join(self.user_cache, mod_path)
            if os.path.exists(p):
                return p
            else:
                return None
        else:
            p = os.path.join(self.sys_cache, mod_path)
            if os.path.exists(p):
                return p
            p = os.path.join(self.user_cache, mod_path)
            if os.path.exists(p):
                return p
            return None

    def _load_base(self, mod_name):
        for base_dir in self.base_dirs:
            mod_path = self.get_cache_path(f"{base_dir}/{mod_name}", "sys")
            if mod_path:
                setattr(self, f"b_{mod_name}", Array.mmap(mod_path).data[: self.env.dates_size, :])
                return
        logging.error(f"[Error] Missing mod_name = {mod_name}")

    def list_base(self, base_dir=None):
        if base_dir:
            return os.listdir(self.get_cache_path(base_dir, "sys"))
        else:
            return self.base_dirs

    def load_base(self, mod_names):
        if isinstance(mod_names, str):
            self._load_base(mod_names)
        else:
            for mod_name in mod_names:
                self._load_base(mod_name)

    def write_mod(self, mod_path="", shape=2, dtype=np.float32):
        if mod_path == "":
            if shape == 2:
                return np.full((self.env.dates_size, self.env.max_univ_size), np.nan, dtype=dtype)
            elif shape == 3:
                return np.full(
                    (self.env.dates_size, len(self.taq_times), self.max_univ_size),
                    np.nan,
                    dtype=dtype,
                )

        else:
            mod_path = self._full_path(mod_path)
            if shape == 2:
                shape = (self.max_dates_size, self.max_univ_size)
            elif shape == 3:
                shape = (self.max_dates_size, len(self.taq_times), self.max_univ_size)
            elif isinstance(shape, dict):
                if shape["type"] == 2:
                    shape = (self.max_dates_size, self.max_univ_size, shape["len"])

            return Array.mmap(
                mod_path,
                writable=True,
                shape=shape,
                dtype=dtype,
            ).data[: self.env.dates_size]

    def load_mod(
        self, mod_path, writable=False, dtype=np.float32, mod_shape=False, force_sys=False
    ):
        if not isinstance(mod_path, str):
            return mod_path

        if force_sys:
            full_mod_path = os.path.join(self.sys_cache, mod_path)
        elif mod_path[0] == "/":
            full_mod_path = mod_path
        else:
            full_mod_path = self._full_path(mod_path)

        if not os.path.exists(full_mod_path):
            print("Missing mod ", mod_path)
            return None

        if mod_shape:
            return Array.mmap(
                full_mod_path,
                writable=writable,
                dtype=dtype,
            ).data
        else:
            return Array.mmap(
                full_mod_path,
                writable=writable,
                shape=(self.max_dates_size, self.max_univ_size),
                dtype=dtype,
            ).data[: self.env.dates_size]

    def eval(self, str_):

        return eval(str_, {key: val for key, val in self.env.pnl_dict.items()})

    def load_pos_file(
        self, file_fmt, start_date, end_date, index_col, col, delimiter, verbose=True, batch=False
    ):
        sig = np.full([self.env.dates_size, self.env.max_univ_size], np.nan, dtype=np.float32)

        start_di = self.env.dates.lower_bound(start_date)
        end_di = self.env.dates.upper_bound(end_date)

        for di in range(start_di, end_di):
            date = self.env.dates[di]
            file_path = format_date(file_fmt, date)
            if not os.path.exists(file_path):
                if verbose:
                    print("Missing ", file_path)
                continue
            else:
                df = pd.read_csv(file_path, delimiter=delimiter)
                if batch:
                    if verbose:
                        arr_index = df[index_col].to_numpy()
                        index_size = arr_index.shape[0]
                        s1, s2, s3 = 0, index_size // 2, index_size - 1
                        if not (
                            arr_index[s1] == self.env.univ[s1]
                            and arr_index[s2] == self.env.univ[s2]
                            and arr_index[s3] == self.env.univ[s3]
                        ):
                            print("Error in batch mode!")
                            return None

                    arr = df[col].to_numpy()
                    sig[di, : arr.shape[0]] = arr
                else:
                    for idx, row in df.iterrows():
                        sid = row[index_col]
                        ii = self.univ.find(sid)
                        sig[di, ii] = row[col]

        return sig

    def dump_pos_file(
        self,
        sig,
        file_fmt,
        start_date,
        end_date,
        index_col="sid",
        col="weight",
        delimiter="|",
        skip_nan=True,
    ):
        if isinstance(sig, str):
            sig = self.load_mod(sig)
        start_di = self.dates.lower_bound(start_date)
        end_di = self.dates.upper_bound(end_date)
        if "{dd}" not in file_fmt or "{mm}" not in file_fmt or "{yyyy}" not in file_fmt:
            print("Please input valid file_fmt: {yyyy}/{mm}/{yyyy}{mm}{dd}.txt")
            return
        # os.makedirs(file_fmt[: file_fmt.rfind("/")], exist_ok=True)
        for di in range(start_di, end_di):
            date = self.dates[di]
            file_path = format_date(file_fmt, date)
            os.makedirs(file_path[: file_path.rfind("/")], exist_ok=True)
            with open(file_path, "w") as f:
                f.write(f"{index_col}{delimiter}{col}\n")

                for ii in range(self.env.univ_size):
                    if not np.isfinite(sig[di, ii]) and skip_nan:
                        continue
                    f.write(f"{self.univ[ii]}{delimiter}{sig[di, ii]}\n")

    def combo(self, alphas):
        sig = np.full((self.env.dates_size, self.env.max_univ_size), 0.0, dtype=np.float32)
        sum_wt = 0.0
        for alpha_ in alphas:
            if isinstance(alpha_, tuple):
                alpha = alpha_[0]
                wt = alpha_[1]
            else:
                alpha = alpha_
                wt = 1.0
            if isinstance(alpha, str):
                xx = self.load_mod(alpha).copy()
            else:
                xx = alpha.copy()

            xx[~np.isfinite(xx)] = 0.0
            sig += xx * wt
            sum_wt += wt
        sig /= sum_wt
        return sig

    def apply_ops(self, sig, ops, start_di=-1, end_di=-1):
        if not hasattr(self.env, "op_manager"):
            self.env.op_manager = OperationManager(self.env)

        sig_op = np.empty_like(sig)
        sig_op.fill(np.nan)
        if start_di == -1:
            start_di = self.env.start_di
        if end_di == -1:
            end_di = self.env.end_di
        self.env.op_manager.apply(sig, sig_op, start_di, end_di, ops)
        return sig_op

    def apply_ops_inplace(self, sig, sig_op, ops, start_di=-1, end_di=-1):
        if not hasattr(self.env, "op_manager"):
            self.env.op_manager = OperationManager(self.env)

        if start_di == -1:
            start_di = self.env.start_di
        if end_di == -1:
            end_di = self.env.end_di
        self.env.op_manager.apply(sig, sig_op, start_di, end_di, ops)

    def get_iis(self, sig, di=0):
        if isinstance(sig, str):
            sig = self.load_mod(sig)

        return np.where(sig[di, : self.env.univ_size])[0]

    def get_univ(self, sig, di=0):

        iis = self.get_iis(sig)
        return sorted([self.univ[ii] for ii in iis])

    def cmp_cache(self, sig, sig2, **kwargs):
        from basic.lib.pycommon.data import cmp_cache

        return cmp_cache(sig, sig2, **kwargs)

    def search_file(self, path, pattern):
        import subprocess

        if path[-1] != "/":
            path += "/"
        paths = (
            subprocess.run(f'find {path} -name "{pattern}"', shell=True, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
            .split("\n")
        )
        return [p.replace(path, "") for p in paths]

    def get_max_date(pattern, func_extract, offset=0):
        import glob

        files = glob.glob(pattern, recursive=True)
        dates = []
        for file in files:
            date = func_extract(file)
            if date not in dates:
                dates.append(date)

        idx = max(0, len(dates) - 1 - offset)
        return sorted(dates)[idx]

    def sub_run(cmd, shell=True, stdout=subprocess.PIPE):
        return subprocess.run(cmd, shell=shell, stdout=stdout)

    def plt_vline(self, start_date, *end_dates):
        start_di = self.dates.find(start_date)
        for end_date in end_dates:
            end_di = self.dates.find(end_date)
            if start_di != -1 and end_di != -1:
                plt.axvline(end_di - start_di)
            else:
                plt.axvline(self.dates.upper_bound(end_date) - self.dates.lower_bound(start_date))

    def pool(f, list_pars, process=5):
        import multiprocessing as mpro

        pool = mpro.Pool(process)
        return pool.starmap(f, list_pars)

    def dump_insample_cache(self, cache_path, end_date, to_dir="dump_research"):
        end_di = self.dates.less_equal_than(end_date)
        os.makedirs(self.env.cache_dir.get_path(to_dir), exist_ok=True)

        if isinstance(cache_path, str):
            cache_paths = [cache_path]
        else:
            cache_paths = cache_path

        for cache_path in cache_paths:
            if cache_path == "sys/env":
                import shutil

                import yaml

                env_dir = self.env.cache_dir.get_path(f"{to_dir}/env")
                os.makedirs(env_dir, exist_ok=True)
                to_file_datetimes = f"{env_dir}/datetimes"
                np.savetxt(to_file_datetimes, self.dates[:end_di], "%d")
                for file in ["listing", "listing.meta", "univ"]:
                    shutil.copy(
                        self.env.cache_dir.get_path(f"env/{file}"),
                        self.env.cache_dir.get_path(f"{to_dir}/env/{file}"),
                    )

                in_file = yaml.safe_load(open(self.env.cache_dir.get_path("env/meta.yml")))
                in_file["univ_end_datetime"] = end_date
                yaml.safe_dump(
                    in_file,
                    open(self.env.cache_dir.get_path(f"{to_dir}/env/meta.yml"), "w"),
                    sort_keys=False,
                )
            else:
                if cache_path.split("/")[0] == "sys":
                    cache_path = cache_path[cache_path.find("/") + 1 :]
                    whole_cache_path = self.env.cache_dir.sys_dir + "/" + cache_path
                    force_sys = True
                else:
                    whole_cache_path = self.env.cache_dir.get_path(cache_path)
                    force_sys = False

                if os.path.isdir(whole_cache_path):
                    alphas = [
                        f"{cache_path}/{_}"
                        for _ in os.listdir(whole_cache_path)
                        if ".meta" not in _ and os.path.exists(f"{whole_cache_path}/{_}.meta")
                    ]
                else:
                    alphas = [whole_cache_path]

                for alpha_name in alphas:

                    x = self.load_mod(alpha_name, force_sys=force_sys)
                    if np.issubdtype(x.dtype, np.bool_):
                        fill_num = False
                    elif np.issubdtype(x.dtype, np.integer):
                        fill_num = -1
                    else:
                        fill_num = np.nan
                    if x.shape == (self.env.dates_size, self.env.max_univ_size):
                        print(alpha_name, x.dtype)
                        a = self.write_mod(f"{to_dir}/{alpha_name}", dtype=x.dtype)
                        a[:end_di] = x[:end_di]
                        try:
                            a[end_di:] = fill_num
                        except:
                            print("Error:", x.dtype)
                            sys.exit(-1)

    ############ register ############
    def register_cov_mat(self, mat_path):
        class _BMat:
            def __init__(self, mat_path):
                self.bmat = BlockMat.mmap(mat_path)

            def load(self, cur_di, ret_type="cov"):
                for di in range(cur_di, -1, -1):
                    if not self.bmat.load_mat(di):
                        continue
                    cov_stocks = []
                    univ_sz = self.bmat.mat_size(di)
                    for idx_ii in range(univ_sz):
                        cov_stocks.append(self.bmat.get_id(di, idx_ii))

                    mat_corr = np.full([univ_sz, univ_sz], 0.0)
                    mat_corr[:, :] = self.bmat[:, :]
                    if ret_type == "cov":
                        mat_diag = np.diag(np.diag(mat_corr))
                        mat_cov = mat_diag @ mat_corr @ mat_diag
                        for id_ii in range(univ_sz):
                            mat_cov[id_ii, id_ii] = mat_cov[id_ii, id_ii] ** (2 / 3)

                        return cov_stocks, mat_cov
                    elif ret_type == "corr":
                        return cov_stocks, mat_corr
                return [], None

        mat_path = self.env.cache_dir.get_path(mat_path)
        self.cov_mat = _BMat(mat_path)
        return self

    def register_risk(self, risk_data, fields):
        class _RiskMod:
            def __init__(self):
                pass

        self.risk = _RiskMod()

        for field in fields:
            setattr(
                self.risk, field, self.load_mod(self.env.cache_dir.get_path(f"{risk_data}/{field}"))
            )
            setattr(self.risk, field.lower(), getattr(self.risk, field))
        return self

    def register_index(self):
        class _Index:
            def __init__(self, sim):
                self.sim = sim
                self.member_strs = ["csi300_member", "csi500_member", "csi1000_member"]
                self.sim.load_base(self.member_strs)
                self.members = [getattr(self.sim, f"b_{_}") for _ in self.member_strs]

            def ana_index_wgt(self, sig, start_date, end_date):
                mp = {}
                start_di, end_di = self.sim.dates.lower_bound(
                    start_date
                ), self.sim.dates.upper_bound(end_date)
                for di in range(start_di, end_di):
                    date = self.sim.dates[di]
                    mp[date] = {}
                    for idx, member in enumerate(self.members):
                        member_str = self.member_strs[idx]
                        mp[date][member_str] = np.nansum(sig[di][np.isfinite(member[di])])
                return mp

        self.index = _Index(self)
        return self

    def register_expr(self):
        from basic.B.basic.expr.expr import expr_macro

        class _ExprRunner(ExprRunner):
            def __init__(self, sim):
                self.sim = sim
                super().__init__(sim.env)

            def add_extra_data(self, input_extra_data):
                extra_data = []

                for tuple in input_extra_data:
                    expr_name = tuple[0]
                    expr_mixed = self.sim._full_path(tuple[1])
                    if len(tuple) > 2:
                        expr_eod = self.sim._full_path(tuple[2])
                    else:
                        expr_eod = expr_mixed
                    # to be done
                    if expr_mixed[0] == "/":
                        expr_mixed = "/" + expr_mixed
                    if expr_eod[0] == "/":
                        expr_eod = "/" + expr_eod

                    extra_data.append((expr_name, expr_mixed, expr_eod))
                super().add_extra_data(extra_data)

            def run(self, ss):
                return super().run(expr_macro(ss))

        self.expr_runner = _ExprRunner(self)
        return self

    def register_crypto(self, flag_print=True, freq=24):
        from basic.lib.pycommon.crypto import Crypto

        self.crypto = Crypto(self, flag_print, freq)
        return self

    def register_extra(self):
        self.months = get_months(self.dates[: self.end_di])
        return self

    def register_fe(self, alpha_store_dir):
        from basic.lib.pycommon.fe import Fe

        self.fe = Fe(self, alpha_store_dir)
        return self

    def register_fdm(self, field, flag_incremental):
        import basic.B.data.in_fdm as in_fdm

        class FdmLoader:
            def __init__(self, cache_path, fdm_field, flag_incremental=False):
                self.fdm_data = RefStructArray.mmap(f"{cache_path}/B_in_fdm/{fdm_field}/data")

                self.date_ts = self.fdm_data.field("dates")
                self.value_ts = self.fdm_data.field("values")
                self.flag_incremental = flag_incremental

            def __call__(self, di, ii):
                values = self.value_ts[di, ii]
                if len(values) == 0:
                    return ([], [])
                return in_fdm.preprocess_fdm(
                    self.date_ts[di, ii].astype(np.int32), values, self.flag_incremental
                )

        self.fdm = FdmLoader(self.sys_cache, field, flag_incremental)
        return self

    def register_equity(self):
        class Equity:
            def __init__(self):
                pass

            def get_equity_names(self, data_path, sid, name, delimiter="|"):
                df = pd.read_csv(data_path, delimiter=delimiter)[[sid, name]]
                return dict(zip(df[sid], df[name]))

        self.equity = Equity()
        return self

    def register_ops(self, ops_str):
        if ops_str == "derisk":
            import basic.B.basic.sup.operations.op_derisk as op_derisk
            from yang.sim.operation import register_operation

            register_operation("derisk", op_derisk.OpDerisk)
        return self

    def register_pycommon(self):
        from basic.lib.pycommon.data import format_date

        class Pycommon:
            def __init__(self):
                pass

        self.pycommon = Pycommon()

        return self

    def register_store(self):
        from basic.lib.pycommon.store import Store

        self.store = Store(self)
        return self

    def register_pnl3(self):
        from basic.lib.pnl3 import Pnl3

        self.pnl3 = Pnl3(self)
        return self

    def extend_cfg(proj_dir, cfg_path, new_data_fmt, data_fmt_parser, start_date, cmds, offset=0):
        # get max_date
        os.chdir(proj_dir)
        max_date = Sim.get_max_date(new_data_fmt, data_fmt_parser, offset)

        f = open(cfg_path)
        cfg_mp = yaml.safe_load(f)
        f.close()

        if "univ_start_datetime" in cfg_mp:
            cfg_mp["univ_end_datetime"] = max_date
            cfg_mp["sim_end_datetime"] = max_date
            cfg_mp["sim_start_datetime"] = start_date
        elif "univ_start_date" in cfg_mp:
            cfg_mp["univ_end_date"] = max_date
            cfg_mp["sim_end_date"] = max_date
            cfg_mp["sim_start_date"] = start_date
        else:
            raise Exception("univ_start_date or univ_start_datetime must be in cfg")

        yaml.safe_dump(cfg_mp, open(cfg_path, "w"), sort_keys=False)
        print(f"-- Extend to {max_date}")

        for cmd in cmds:
            print(f"-- running {cmd}")
            result = Sim.sub_run(cmd)
            output = result.stdout.decode("utf-8")

            for check_word in ["error", "fail"]:
                if check_word in output:
                    print(f"{check_word} in output")

        if "univ_start_datetime" in cfg_mp:
            print(f'-- Fill back sim_start_datetime to {cfg_mp["univ_start_datetime"]}')
            cfg_mp["sim_start_datetime"] = cfg_mp["univ_start_datetime"]
        elif "univ_start_date" in cfg_mp:
            print(f'-- Fill back sim_start_date to {cfg_mp["univ_start_date"]}')
            cfg_mp["sim_start_date"] = cfg_mp["univ_start_date"]

        yaml.safe_dump(cfg_mp, open(cfg_path, "w"), sort_keys=False)
