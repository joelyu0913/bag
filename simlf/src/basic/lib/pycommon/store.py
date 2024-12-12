import logging
import os

import numpy as np
import yaml


class Store:
    def __init__(self, sim):
        self.sim = sim

    def merge_StoreMP(
        self,
        store_mp_ymls,
        start_date,
        end_date,
        output_dir,
        dtype=np.float32,
        verbose=True,
        missing_alert_num=1,
    ):
        mp_inds = {}
        for store_mp_yml in store_mp_ymls:
            mp_inds = {**mp_inds, **yaml.safe_load(open(store_mp_yml))}

        meta = {
            "shape": [self.sim.univ_size, len(mp_inds)],
            "univ": list(self.sim.univ),
            "inds": list(mp_inds.keys()),
        }

        sigs = []
        for store_ind in meta["inds"]:
            ind = mp_inds[store_ind]
            sigs.append(self.sim.load_mod(ind))

        start_di = self.sim.dates.lower_bound(start_date)
        end_di = self.sim.dates.upper_bound(end_date)

        os.makedirs(output_dir, exist_ok=True)

        # dump_data
        for di in range(start_di, end_di):
            output = np.empty(meta["shape"], dtype)

            for idx in range(meta["shape"][1]):
                for ii in range(meta["shape"][0]):
                    output[ii, idx] = sigs[idx][di, ii]

            if verbose:
                yy = np.nansum(np.abs(output), axis=0)
                missing_column = np.sum(yy < 1e-10)

                if missing_column > missing_alert_num:
                    logging.warning(f"{self.sim.dates[di]} data missing inds {missing_column}!")

            path = f"{output_dir}/{self.sim.dates[di]}"
            output.tofile(f"{path}.sig")

            # dump_meta
            yaml.safe_dump(meta, open(f"{path}.meta", "w"), sort_keys=False)
            if verbose:
                # logging.info(f"Finish {self.sim.dates[di]} !")
                print(f"Finish {self.sim.dates[di]}")

    def load_merged_cache(cache_path):
        meta = yaml.safe_load(open(f"{cache_path}.meta"))
        sig = np.memmap(f"{cache_path}.sig", mode="r", shape=tuple(meta["shape"]), dtype=np.float32)
        return sig, meta
