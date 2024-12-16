import logging
import os
import sys

import numpy as np
import pandas as pd

from sim import Module
from basic.lib.simm import Sim
from basic.lib.pycommon.data import read_header, read_line


def write_member(sim, gz_univ, gz_member, date, di, path_fmt, file_type):
    date = str(date)
    yyyy, mm, dd = date[:4], date[4:6], date[6:]
    file_path = path_fmt.format(yyyy=yyyy, mm=mm, dd=dd)
    if os.path.exists(file_path):
        gz_univ[di, :] = False
        gz_member[di, :] = np.nan
        sum_wt = 0.0
        if file_type == "csv":
            with open(file_path, 'rt') as f:
                header_mp = read_header(f, sep=',')

                for ln in f:
                    row = read_line(ln, header_mp, sep=',', float_fields = ['weight'])
                    sid = row["code"]
                    wt = row["weight"]
                    ii = sim.univ.find(sid)
                    gz_univ[di, ii] = True
                    gz_member[di, ii] = wt
                    sum_wt += wt

        elif file_type == "feather":
            df = pd.read_feather(file_path)
            for row in df.iterrows():
                row = row[1]
                sid = f'{row["CONSTITUENTCODE"]}.{row["EXCHANGE"][-2:]}'
                wt = row["WEIGHT"]
                ii = sim.univ.find(sid)
                gz_univ[di, ii] = True
                gz_member[di, ii] = wt
                sum_wt += wt

        gz_member[di] /= sum_wt
    else:
        if di > 0:
            gz_member[di] = gz_member[di - 1]
            gz_univ[di] = gz_univ[di - 1]

        logging.warning(f"Missing index file: {file_path} !")


class SupUnivIndex(Module):
    def run_impl(self):
        path_fmt = self.config["path_fmt"]
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})

        gz_univ = sim.write_mod(self.config["output_univ"], dtype=bool)
        gz_member = sim.write_mod(self.config["output_member"], dtype=np.float32)
        backfill_date = int(self.config["backfill_date"])
        file_type = self.config.get("file_type", "csv")
        for di in range(self.start_di, self.end_di):
            date = sim.dates[di]
            if date < backfill_date:
                date = backfill_date
            write_member(sim, gz_univ, gz_member, date, di, path_fmt, file_type)
