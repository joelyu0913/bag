set -e
source .env

python src/cli/run.py --cfg config/region/daily.ycfg --cfg config/run/daily.user.ycfg --cfg src/user/post_ana/ff.ycfg
