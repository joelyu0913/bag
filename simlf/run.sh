set -e
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg src/user/test/long_one/ff.ycfg $*
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg src/basic/cn_base/ff.ycfg $*
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/test/kdr_py/ff.ycfg $*

