set -e
source .env

# build base data
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg src/basic/cn_base/ff.ycfg  $*
# build user alpha
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/kdr_py/ff.ycfg  $*

