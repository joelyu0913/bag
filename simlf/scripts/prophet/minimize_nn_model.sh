#!/usr/bin/bash

set -e

if [[ $# != 1 ]]; then
  echo "Usage: $0 <grp-dir>"
  exit 1
fi

grp_dir=$1
rm -f $grp_dir/*/roll_*/data/states_*.pkl
rsync -a -r $grp_dir/ $grp_dir.bak/
echo "back up $grp_dir.bak"
for model_dir in $grp_dir/*; do
  echo "dir: $model_dir"

  for roll_dir in $model_dir/roll_*; do
    if [[ -e "$roll_dir/data/model_0.pkl" ]]; then
      echo "roll_dir: $roll_dir"
      cp -L "$roll_dir/data/model_best.pkl" "$roll_dir/data/best.pkl"
      rm -f $roll_dir/data/model_*.pkl
      mv "$roll_dir/data/best.pkl" "$roll_dir/data/model_best.pkl"
    fi
  done
done
