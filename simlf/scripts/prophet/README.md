Usage examples
----
```bash
./scripts/prophet/train.py --workdir local/lgbm_ta_374_addlimit \
  --config scripts/prophet/config/train_lgbm_addlimit.yml \
  --inds local/inds_ta_374.yml \
  --data-dir tmp/cache_os \
  --predict-config scripts/prophet/config/predict.yml

./scripts/prophet/predict.py --workdir local/lstm_ta_374_addlimit \
  --config scripts/prophet/config/predict.yml \
  --data-dir tmp/cache_os
```
