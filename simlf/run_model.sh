set -e
source .env

python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg src/basic/sup/fret/ff.ycfg $*
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/kdr_py/ff_ml.ycfg -t 16  $*

sys_cache=tmp/cache/cn/sys
user_cache=tmp/cache/cn/user
model_dir=tmp/cache/cn/model

for i in {0..0}; do
  ./scripts/prophet/train.py --workdir $model_dir/lgbm/grp_lgbm_kdr/$i \
    --config scripts/prophet/config/train_lgbm_addlimit.yml  \
    --inds src/user/kdr_py/inds_200.yml \
    --sys-data-dir $sys_cache \
    --user-data-dir $user_cache \
    --random-state $i
done

python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/M/M_lgbm.ycfg -t 16 $*
