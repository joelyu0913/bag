set -e
source .env


#### run basic test
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg src/basic/cn_base/ff.ycfg  $*
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/kdr_py/ff_test.ycfg --always-run 'kdr*'


#### run univ
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/univ_index.ycfg --always-run 'sup*' -t 5
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/cnall.ycfg --always-run 'sup*'
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/cnforbid.ycfg --always-run 'sup*'
# python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/univ/cninfo.ycfg --always-run 'sup*'

#### run taq 
python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.sys.ycfg --cfg config/user/basic/sup/taq/5m_cn.ycfg --always-run 'sup*'
