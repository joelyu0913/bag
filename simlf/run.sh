set -e
source .env


#### build base data
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg src/basic/cn_base/ff.ycfg  $*
#### build user alpha
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/kdr_py/ff.ycfg --always-run 'kdr*'
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/kdr_py/ff_test.ycfg --always-run 'kdr*'

# python src/cli/run.py --cfg config/jp/region/daily.ycfg --cfg config/jp/run/daily.sys.ycfg --cfg src/basic/cn_base/ff.ycfg  $*
#### build user alpha
# python src/cli/run.py --cfg config/jp/region/daily.ycfg --cfg config/jp/run/daily.user.ycfg --cfg src/user/kdr_py/ff_test.ycfg --always-run 'kdr*'

### run univ
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/univ_index.ycfg --always-run 'sup*' -t 5
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/cnall.ycfg --always-run 'sup*'
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/cnforbid.ycfg --always-run 'sup*'
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/cninfo.ycfg --always-run 'sup*'

