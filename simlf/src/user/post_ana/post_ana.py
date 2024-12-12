import logging
import sys, os

import numba
import numpy as np
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings("ignore")

from sim import Module
from basic.lib.pycommon.oper import ts_mean, c_demean
from basic.lib.simm import Sim, apply_ops

def dump_fig(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

from fpdf import FPDF
class PDF(FPDF):
    def header(self):
        # Title of the document
        self.set_font("Arial", 'B', 12)
        self.ln(10)

    def table(self, data, col_widths=None):
        # Set default column widths if not provided
        if col_widths is None:
            col_widths = [40] * len(data[0])
        
        self.set_font("Arial", size=10)
        
        # Draw the table header
        for heading in data[0]:
            self.cell(col_widths[data[0].index(heading)], 10, heading, 1, 0, 'C')
        self.ln()

        # Draw the table rows
        for row in data[1:]:
            for item, width in zip(row, col_widths):
                self.cell(width, 10, str(item), 1, 0, 'C')
            self.ln()

mp_exch = {0: 'SH', 1: 'SZ', 2: 'SZ', 3: 'SZ', 4: 'SH'}#, 5: 'SH', 6: 'BSE'}
output_fields = ['all', 'csi300', 'csi500', 'csi1000', 'SH', 'SZ']
class PostAna(Module):
    def run_impl(self):

        output_root = self.config.get('output_root', 'out')
        sim = Sim({"sys_cache": self.cache_dir.sys_dir, "user_cache": self.cache_dir.user_dir})

        start_date = self.config['start_date']
        end_date = self.config.get('end_date', None)
        if end_date is None:
            end_date = start_date
        start_di = sim.dates.lower_bound(start_date)
        end_di = sim.dates.upper_bound(end_date)    


        dvol = sim.load_mod("base/dvol")
        high = sim.load_mod("base/high")
        close = sim.load_mod("base/close")
        low = sim.load_mod("base/low")
        ret = sim.load_mod("base/ret")
        csi = {'csi300': sim.load_mod("sup_univ/csi300"), 'csi500': sim.load_mod("sup_univ/csi500"), 'csi1000': sim.load_mod("sup_univ/csi1000")}
        exch = sim.load_mod("base/exch")


        for cur_di in range(start_di, end_di):
            cur_date = sim.dates[cur_di]
            output_dir = f'{output_root}/{cur_date}'
            os.makedirs(output_dir, exist_ok=True)
            hist = {_: [] for _ in output_fields}
            for lookback in range(0, 21):
                mp = {_: np.array([0.,0.,0.,0.,0.]) for _ in output_fields}
                
                di = cur_di - lookback
                for ii in range(sim.univ_size):
                    dvol_ = dvol[di, ii]
                    hl_ = (high[di, ii] - low[di, ii]) / close[di, ii]
                    ret_ = ret[di, ii] 
                    abs_ret_ = np.abs(ret_)
                    vec = [1., dvol_, hl_, ret_, abs_ret_]
                    
                    if not (dvol_ > 0 and hl_ > 0 and np.isfinite(ret_)):
                        continue
                    mp['all'] += vec
                    for field in ['csi300', 'csi500', 'csi1000']:
                        if csi[field][di, ii]:
                            mp[field] += vec

                    for field in ['SH', 'SZ']:
                        exch_ = exch[di, ii]
                        if exch_ in mp_exch and mp_exch[exch_] == field:
                            mp[field] += vec
                    
                
                for field in output_fields:
                    hist[field].append(mp[field])
            
            for field in output_fields:
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                dvol_list = [_[1] for _ in hist[field]][::-1]
                hl_list = [_[2] / _[0] for _ in hist[field]][::-1]
                ret_list = [_[3] / _[0] for _ in hist[field]][::-1]
                abs_ret_list = [_[4] / _[0] for _ in hist[field]][::-1]

                axs[0, 0].plot(dvol_list)
                axs[0, 0].set_title('dvol')
                axs[0, 1].plot(hl_list)
                axs[0, 1].set_title('high - low')
                axs[1, 0].plot(ret_list)
                axs[1, 0].set_title('ret')
                axs[1, 1].plot(abs_ret_list)
                axs[1, 1].set_title('absolute ret')
                fig.tight_layout()
                plt.savefig(f"{output_dir}/{field}.png")
                plt.close()


# axs[1, 1].set_ylim(-10, 10)  # Limiting y-axis for tangent plot for better visualization
            if True:
                pdf = PDF()

                # Add a page

                # Set font for text
                pdf.set_font("Arial", size=16)
                pdf.set_auto_page_break(auto=True, margin=15)
                height = 120
                y_gap = 120

                #
                f_e = lambda x, k=3: ("{:." + f"{k}" +"e}").format(x)
                f_d = lambda x, k=3: ("{:." + f"{k}" +"f}").format(x)
                for field in output_fields:
                    pdf.add_page()
                    pdf.set_font("Arial", size=16)
                    if field == 'all':
                        pdf.cell(0, 10, txt="Market Report: CN", ln=True, align='C')
                    else:
                        pdf.cell(0, 10, txt=" --- ", ln=True, align='C')
                    pdf.cell(20, 10, txt=field, ln=True, align='C')
                    tb = [['field', 'today', '1d', '5d-mean', '20d-mean', 'today / 1d', 'today / 5d', 'today / 20d']]

                    dvol_list = [_[1] for _ in hist[field]]
                    hl_list = [_[2] / _[0] for _ in hist[field]] 
                    ret_list = [_[3] / _[0] for _ in hist[field]]
                    abs_ret_list = [_[4] / _[0] for _ in hist[field]]

                    def f(name, vv):
                        v_0 = vv[0]
                        v_1 = vv[1]
                        v_5 = np.mean([_ for _ in vv[1:6]])
                        v_20 = np.mean([_ for _ in vv[1:21]])
                        return [name, f_e(v_0), f_e(v_1), f_e(v_5), f_e(v_20), f_d(v_0 / v_1, 3), f_d(v_0 / v_5, 3), f_d(v_0 / v_20, 3)]
                    
                    dvol_v = f('dvol', dvol_list)
                    hl_v = f('hl', hl_list)
                    ret_v = f('ret', ret_list)
                    abs_ret_v = f('abs_ret', abs_ret_list)

                    xx = ['?', '?', '?', '?', '?', '?', '?', '?']

                    tb.append(dvol_v)
                    tb.append(hl_v)
                    tb.append(ret_v)
                    tb.append(abs_ret_v)
                    pdf.table(tb, col_widths=[23 for _ in range(8)])
                    # pdf.set_y(y_gap)
                    pdf.image(f"{output_dir}/{field}.png", x=20, y=y_gap, h=height)

                pdf.output(f"{output_dir}/market_cn_{cur_date}.pdf") 
                logging.info(f"Post report generated for {cur_date}")
