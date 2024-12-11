import logging
import numpy as np
from sim import Module

class SupUnivCn(Module):
    def run_impl(self):
        #hash_int_set = yang::unordered_set<int>;
        univ = self.config["par_univ"]
        b_univ_custom = self.write_array(univ, dtype=bool)
        b_sup_passed = self.write_array(univ + "_passed", dtype=bool)
        univ_size_ = int(self.config["par_univ_size"])
        lookback_ = int(self.config["par_univ_lookback"])
        sticky_ = int(self.config["par_univ_sticky_days"])
        vol_window_ = int(self.config["par_univ_vol_window"])
        pass_rate_ = float(self.config["par_univ_pass_rate"])
        min_cap_ = float(self.config["par_univ_min_cap"])
        min_dvol_ = float(self.config["par_univ_min_dvol"])
        min_prc_ = float(self.config["par_univ_min_prc"])
        max_prc_ = float(self.config["par_univ_max_prc"])
        vol_mode_ = "mean"

        b_univ_parent = self.read_array(self.config.get("par_univ_parent", "base/univ_all"))
        candidates = {}
        b_close = self.read_array("base/close")
        b_vol = self.read_array("base/vol")
        b_cap = self.read_array("base/cap")
        b_st = self.read_array("base/st")
        get_dollar_vol = lambda di, ii: b_close[di, ii] * b_vol[di, ii]

        for di in range(self.start_di - lookback_ if self.start_di > lookback_ else 1, self.end_di):
            sorted_vol = {}
            for ii in range(self.univ_size):
                fx = 1.
                if b_univ_parent[di, ii] and b_st[di, ii] == 0 and b_cap[di - 1, ii] / fx >= min_cap_ and b_close[di - 1, ii] / fx >= min_prc_ and b_close[di - 1, ii] / fx <= max_prc_:
                    dvols = []
                    for w in range(vol_window_):
                        if w > di - 1:
                            break
                        dvol = get_dollar_vol(di - 1 - w, ii)
                        if dvol >= 0 and np.isfinite(dvol):
                            dvols.append(dvol)
                    if dvols:
                        est_dvol = self.CalcEstimatedVol(dvols, vol_mode_)
                        if est_dvol >= min_dvol_:
                            sorted_vol[est_dvol] = ii

#       // select the top (univ_size) stocks as candidates
#       // traversing sorted_vol backwards gives us the avg volumes in
#       // decreasing order
            candidates[di] = set()
            result = candidates[di]
            for rcurr in sorted(sorted_vol, reverse=True):
                if len(result) < univ_size_:
                    result.add(sorted_vol[rcurr])
            if not result:
                logging.error(f"[{self.name}] no stocks matched universe volume criteria on {self.dates[di]}")
            else:
                logging.info(f"[{self.name}] Selected {len(result)} candidates on {self.dates[di]} in the first pass")

#     // 2nd pass
        pick_cnt = [0] * self.univ_size
        last_good = [-1] * self.univ_size
        for ii in range(self.univ_size):
            di = self.start_di - 1
            if di >= 0:
                lb1 = lookback_ if di > lookback_ else di
                lb2 = sticky_ if di > sticky_ else di
                for dl in range(lb1):
                    if ii in candidates[di - dl]:
                        pick_cnt[ii] += 1
                for dl in range(lb2):
                    if b_sup_passed(di - dl, ii):
                        last_good[ii] = di - dl
                        break

        for di in range(self.start_di, self.end_di):
            if di < 1:  # this selection has to always be delay 1
                continue
            lb1 = lookback_ if di > lookback_ else di
            lb2 = sticky_ if di > sticky_ else di
            dl = 0
#       // error check: make sure all necessary lookback dates were covered
#       // in 1st pass
            for dl in range(lb1):
                if di - dl not in candidates:
                    logging.warning(f"[{self.name}] universe: candidate universe was not computed for {di - 1 - dl}")

            activated = added = removed = 0
            for ii in range(self.univ_size):
#         // master universe gets priority
                good = b_univ_parent[di, ii]
                if di > lookback_ and pick_cnt[ii] > 0:
                    pick_cnt[ii] -= ii in candidates[di - lookback_]
                if good:
#           // check the last 'lookback' days
#           // candidates was computed with delay 1 info, so we can
#           // start with today
                    if ii in candidates[di]:
                        pick_cnt[ii] += 1
                    assert pick_cnt[ii] <= lb1
                    if lb1 > 0 and pick_cnt[ii] / lb1 * 100 < pass_rate_:
                        good = False
#           // record whether it passed_ today
                    b_sup_passed[di, ii] = good
                    if good:
                        last_good[ii] = di
                    else:
#             // must be bad for 'sticky' days in order to deactivate
                        good = last_good[ii] >= di - lb2
                if good:
                    activated += 1
                    b_univ_custom[di, ii] = True
                    if di == 0 or not b_univ_custom[di - 1, ii]:
                        added += 1
                else:
                    b_univ_custom[di, ii] = False
                    if di == 0 or b_univ_custom[di - 1, ii]:
                        removed += 1

            logging.info(f"[{self.name}] Activated {activated} instruments on {self.dates[di]} (+{added} -{removed})")

    def CalcEstimatedVol(self, vols, vol_mode):
        est_vol = 0.0
        def find_median():
            n = len(vols) // 2
            vols.sort()
            if len(vols) % 2 == 0:
                return (vols[n] + max(vols[:n])) / 2
            else:
                return vols[n]
        def find_mean():
            sum_ = 0
            for v in vols:
                sum_ += v
            return sum_ / len(vols)

        if vol_mode == "median":
            est_vol = find_median()
        elif vol_mode == "min_median_mean":
            est_vol = min(find_median(), find_mean())
        else:
            est_vol = find_mean()

        return est_vol

