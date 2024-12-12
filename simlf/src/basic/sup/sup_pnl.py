import io
import logging

from sim import Module
from basic.lib.pnl import PnlStats, pnl_dict


class SupPnl(Module):
    def run_impl(self):
        pnl_start_date = self.config.get("pnl_start_date", self.env.config.get("pnl_start_date"))
        pnl_end_date = self.config.get("pnl_end_date", self.env.config.get("pnl_end_date"))
        pnl_filter = self.env.config.get("filter", "")
        if pnl_start_date is None:
            start_di = self.start_di
        else:
            start_di = self.dates.lower_bound(pnl_start_date)
        if pnl_end_date is None:
            end_di = self.end_di
        else:
            end_di = self.dates.upper_bound(pnl_end_date)

        pnl_metrics = ["base"]
        if self.env.benchmark_index != "":
            pnl_metrics.append("hedge")
        if self.config.get("tpnl", False):
            pnl_metrics.append("tcost")
        pnl_metrics += ["ic", "ir"]
        pnl_args = {}
        for arg_name in ["book_size", "ret", "hedge", "buy_fee", "sell_fee"]:
            if arg_name in self.config:
                pnl_args[arg_name] = self.config[arg_name]
        pnl = PnlStats(pnl_metrics, pnl_args)

        for alpha_config in self.config.get("alphas", []):
            alpha = alpha_config["alpha"]
            ops = alpha_config.get("ops", "")
            daily_stats = pnl.compute(
                env=self.env,
                sig=self.cache_dir.get_path(alpha),
                ops=ops,
                start_di=start_di,
                end_di=end_di,
            )
            yearly_stats = pnl.summarize_yearly(self.env, daily_stats)

            if pnl_filter == "" or pnl_dict(yearly_stats).eval(pnl_filter):
                header = alpha
                if ops:
                    header += ":" + ops
                out_buf = io.StringIO()
                out_buf.write("\n")
                print(header, file=out_buf)
                pnl.show(yearly_stats, out=out_buf)
                logging.info(out_buf.getvalue())
