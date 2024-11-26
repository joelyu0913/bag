import os

import numpy as np
import yaml


def fe_down_corr(valid_list, metric_mp, corr_func, corr_thd):
    final_list = []
    for alpha_name in valid_list:
        final_flag = True
        coss1 = metric_mp[alpha_name]

        for final_name in final_list:
            coss2 = metric_mp[final_name]
            corr_ = corr_func(np.array(coss1), np.array(coss2))

            if corr_ > corr_thd:
                final_flag = False
                break
        if final_flag:
            final_list.append(alpha_name)
    return final_list


def ops_to_str(ops):
    return ops.replace("|", "__").replace(".", "_")


class FeMeta:
    def __init__(self, pnls=None, stats_all=None, stats_freq=None):
        self.pnls = pnls
        self.stats_all = stats_all
        self.stats_freq = stats_freq

    def exist(path):
        return os.path.exists(path + ".npy") and os.path.exists(path + ".yaml")

    def save(self, path):
        import yaml

        np.save(path + ".npy", self.pnls)
        with open(path + ".yaml", "w") as f:
            yaml.safe_dump({"stats_all": self.stats_all, "stats_freq": self.stats_freq}, f)

    def load(self, path):
        import yaml

        self.pnls = np.load(path + ".npy")
        meta_ = yaml.safe_load(open(path + ".yaml"))
        self.stats_all = meta_["stats_all"]
        self.stats_freq = meta_["stats_freq"]
        return self


class Fe:
    def __init__(self, sim, alpha_store_dir):
        self.sim = sim
        self.alpha_store_dir = alpha_store_dir
        self.lab_dir = self.sim.cache_dir.get_path("lab")

    def open_lab(self, lab_dir):
        self.lab_dir = self.sim.cache_dir.get_path(lab_dir)

    def load_lab_cache(self, cache_name):
        if not os.path.exists(f"{self.lab_dir}/{cache_name}"):
            return None
        else:
            return self.sim.load_mod(f"{self.lab_dir}/{cache_name}")

    def get_lab_path(self, path=""):
        return f"{self.lab_dir}/{path}"

    def search_file(self, pattern, sub_dir=""):
        return self.sim.search_file(f"{self.alpha_store_dir}/{sub_dir}", pattern)

    def get_path(self, path=""):
        return f"{self.alpha_store_dir}/{path}"

    def load_yaml(self, path):
        return yaml.safe_load(open(self.get_path(path)))

    def insert_mp2(
        self,
        mp_init,
        ind_name,
        ind_path,
        ops,
        start_date=-1,
        end_date=-1,
        sids=[],
        flag_cache=False,
    ):

        mod_name = f"{self.get_lab_path(ind_name)}.{ops_to_str(ops)}"

        sids_str = "".join([str(sid) for sid in sids])
        if start_date == -1:
            start_date = self.sim.dates[0]

        if end_date == -1:
            end_date = self.sim.dates[-1]

        pnl_path = self.get_lab_path(f"{mod_name}.{sids_str}.{start_date}.{end_date}")
        if not FeMeta.exist(pnl_path):
            sig = self.load_lab_cache(mod_name)
            if sig is None:
                sig = self.sim.write_mod(self.get_lab_path(mod_name) if flag_cache else "")
                isig = self.sim.load_mod(ind_path).copy()
                sig[:, :] = self.sim.apply_ops(isig, ops)

            info = self.sim.pnl2.compute_pnls(
                sig, sids=sids, start_date=start_date, end_date=end_date, fee_rate=0.0
            )
            FeMeta(*info).save(pnl_path)

            if info[1]["total_pnl"] >= 0.0:
                mp_init[f"{mod_name}:{ind_path}:{ops}"] = FeMeta(*info)
            else:
                if ops == "":
                    ops_neg = "neg"
                else:
                    ops_neg = "neg|" + ops

                mod_name_neg = f"{ind_name}.{ops_to_str(ops_neg)}"
                sig_neg = self.load_lab_cache(mod_name_neg)
                if sig_neg is None:
                    sig_neg = self.sim.write_mod(
                        self.get_lab_path(mod_name_neg) if flag_cache else ""
                    )
                    isig = self.sim.load_mod(ind_path).copy()
                    sig_neg[:, :] = self.sim.apply_ops(isig, ops_neg)

                info_neg = self.sim.pnl2.compute_pnls(
                    sig_neg, sids=sids, start_date=start_date, end_date=end_date, fee_rate=0.0
                )
                pnl_path_neg = self.get_lab_path(
                    f"{mod_name_neg}.{sids_str}.{start_date}.{end_date}"
                )
                FeMeta(*info_neg).save(pnl_path_neg)
                mp_init[f"{mod_name_neg}:{ind_path}:{ops_neg}"] = FeMeta(*info_neg)
        else:
            femeta = FeMeta().load(pnl_path)
            if femeta.stats_all["total_pnl"] >= 0.0:
                mp_init[f"{mod_name}:{ind_path}:{ops}"] = femeta
            else:
                if ops == "":
                    ops_neg = "neg"
                else:
                    ops_neg = "neg|" + ops
                mod_name_neg = f"{ind_name}.{ops_to_str(ops_neg)}"
                pnl_path_neg = self.get_lab_path(
                    f"{mod_name_neg}.{sids_str}.{start_date}.{end_date}"
                )
                mp_init[f"{mod_name_neg}:{ind_path}:{ops_neg}"] = FeMeta().load(pnl_path_neg)

    def insert_mp(self, mp_init, feature, sig_ori, start_date, end_date=-1, sids=[], ops=""):
        if end_date == -1:
            end_date = self.sim.dates[-1]

        if "default" in self.sim.pnl2.cost:
            ori_cost = self.sim.pnl2.cost["default"]
        else:
            ori_cost = 0.0

        self.sim.pnl2.set_cost(0e-4)
        if ops != "":
            sig = self.sim.apply_ops(sig_ori, ops)
        else:
            sig = sig_ori
        info = self.sim.pnl2.compute_pnls(sig, sids=sids, start_date=start_date, end_date=end_date)
        if np.nansum(info[0]) < 0:
            if ops == "":
                ops_ = "neg"
            else:
                ops_ = "neg|" + ops

            sig = self.sim.apply_ops(sig_ori, ops_)
            info = self.sim.pnl2.compute_pnls(
                sig, sids=sids, start_date=start_date, end_date=end_date
            )
            mp_init[f"{feature}:{ops_}"] = info[:2]
        else:
            mp_init[f"{feature}:{ops}"] = info[:2]

        self.sim.pnl2.set_cost(ori_cost)

    def get_mp_crypto(self, features, start_date, end_date=-1, sids=[], ops=""):
        mp_init = {}
        for feature_meta in features:
            if isinstance(feature_meta, str):
                feature = feature_meta
                sig = self.sim.load_mod(feature)
            elif isinstance(feature_meta, tuple):
                feature, sig = feature_meta
            else:
                print("Error: wrong features type!")
                return {}

            self.insert_mp(mp_init, feature, sig, start_date, end_date, sids, ops)

        return mp_init

    def down_corr(self, valid_list, metric_mp, corr_func, corr_thd):
        return fe_down_corr(valid_list, metric_mp, corr_func, corr_thd)

    def dump_FI(self, FI_group, FI_inds, FI_file):
        yaml.dump(
            [
                dict(zip(["fi", "ind", "ops"], f"{FI_group}/{FI_ind}".split(":")))
                for FI_ind in FI_inds
            ],
            open(FI_file, "w"),
            sort_keys=False,
        )

    def create_FI_ycfg(self, output_file, FI_name):
        with open(self.get_path(output_file), "w") as f:
            f.write(
                """fi_name = "{FI_name}"

fi_mods, fi_outputs = py_fi(resolve_relative_path(f'{{fi_name}}.yml'))

export(f'{{fi_name}}_mods', fi_mods)
export(f'{{fi_name}}_outputs', fi_outputs)
""".format(
                    FI_name=FI_name
                )
            )

    def create_MPred_ycfg(
        self, output_file, model_name, model_path, FI_path, univ, ops1=None, ops2=None
    ):
        with open(output_file, "w") as f:
            if ops1 is not None:
                ops1 = '"' + ops1 + '"'
            if ops2 is not None:
                ops2 = '"' + ops2 + '"'
            f.write(
                """
require("yao/A/_common/M/ycfg/combo_model.ycfg")
require("yao/A/crypto/_common/data/sup_univ/crypto.ycfg")

combo_model2(
    mod_name="MPred_{model_name}",
    model_dir=f"$model_dir/{model_path}",
    inds_path=f"{{alpha_store_dir}}/{FI_path}",
    univ="{univ}",
    ops1={ops1},
    ops2={ops2},
    deps=[PV_mods, "sup_univ_xx"],
)

""".format(
                    model_name=model_name,
                    model_path=model_path,
                    FI_path=FI_path,
                    univ=univ,
                    ops1=ops1,
                    ops2=ops2,
                )
            )

    def create_train_sh(self, output_file, model_path, MP_paths, FI_path, univ):
        with open(output_file, "w") as f:
            MP_str = ""
            for MP_path in MP_paths:
                MP_str += f"    --config ${{alpha_store_dir}}/{MP_path} \\"
                if MP_path != MP_paths[-1]:
                    MP_str += "\n"
            f.write(
                """
#!/bin/bash
set -e

if [[ $# != 2 ]]; then
   exit 1;
fi

sys_cache=$1
user_cache=$2
model_dir=$3
alpha_store_dir={alpha_store_dir_}

for i in {{0..9}}; do
  ./scripts/prophet/train.py --workdir ${{model_dir}}/{model_path}/$i \\
{MP_str}
    --inds ${{alpha_store_dir}}/{FI_path} \\
    --data-dir ${{cache_dir}} \\
    --random-state $i \\
    --univ {univ}

done
""".format(
                    model_path=model_path,
                    MP_str=MP_str,
                    FI_path=FI_path,
                    univ=univ,
                    fe_dir_=self.alpha_store_dir,
                )
            )
