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
        csi = {'csi300': sim.load_mod("sup_univ/csi300"), 'csi500': sim.load_mod("sup_univ/csi500"), 'csi1000': sim.load_mod("sup_univ/csi1000")}


        for cur_di in range(start_di, end_di):
            cur_date = sim.dates[cur_di]
            output_dir = f'{output_root}/{cur_date}'
            os.makedirs(output_dir, exist_ok=True)
            hist = {'all': [], 'csi300': [], 'csi500': [], 'csi1000': []}
            for lookback in range(0, 21):
                mp = {'all': np.array([0., 0., 0.]), 'csi300': np.array([0., 0., 0.]), 'csi500': np.array([0., 0., 0.]), 'csi1000': np.array([0., 0., 0.])}
                
                di = cur_di - lookback
                for ii in range(sim.univ_size):
                    dvol_ = dvol[di, ii]
                    hl_ = (high[di, ii] - low[di, ii]) / close[di, ii]
                    
                    if not (dvol_ > 0 and hl_ > 0):
                        continue
                    mp['all'] += [1., dvol_, hl_]
                    for index in ['csi300', 'csi500', 'csi1000']:
                        if csi[index][di, ii]:
                            mp[index] += [1., dvol_, hl_]
                
                for index in ['all', 'csi300', 'csi500', 'csi1000']:
                    hist[index].append(mp[index])
            
            for index in ['all', 'csi300', 'csi500', 'csi1000']:
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                dvol_list = [_[1] for _ in hist[index]]
                hl_list = [_[2] / _[0] for _ in hist[index]] 

                axs[0, 0].plot(dvol_list[::-1])
                axs[0, 0].set_title('dvol')
                axs[0, 1].plot(hl_list[::-1])
                axs[0, 1].set_title('high - low')
                fig.tight_layout()
                plt.savefig(f"{output_dir}/{index}.png")
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
                for index in ['all', 'csi300', 'csi500', 'csi1000']:
                    pdf.add_page()
                    pdf.set_font("Arial", size=16)
                    if index == 'all':
                        pdf.cell(0, 10, txt="Post-Trading Daily Report: CN", ln=True, align='C')
                    else:
                        pdf.cell(0, 10, txt=" --- ", ln=True, align='C')
                    pdf.cell(20, 10, txt=index, ln=True, align='C')
                    tb = [['field', 'today', '1d', '5d-mean', '20d-mean', 'today / 1d', 'today / 5d', 'today / 20d']]

                    dvol_list = [_[1] for _ in hist[index]]
                    hl_list = [_[2] / _[0] for _ in hist[index]] 

                    v_0 = dvol_list[0]
                    v_1 = dvol_list[1]
                    v_5 = np.mean([_ for _ in dvol_list[1:6]])
                    v_20 = np.mean([_ for _ in dvol_list[1:21]])
                    dvol_v = ['dvol', f_e(v_0), f_e(v_1), f_e(v_5), f_e(v_20), f_d(v_0 / v_1, 3), f_d(v_0 / v_5, 3), f_d(v_0 / v_20, 3)]

                    v_0 = hl_list[0]
                    v_1 = hl_list[1]    
                    v_5 = np.mean([_ for _ in hl_list[1:6]])
                    v_20 = np.mean([_ for _ in hl_list[1:21]])
                    hl_v = ['hl', f_d(v_0), f_d(v_1), f_d(v_5), f_d(v_20), f_d(v_0 / v_1, 3), f_d(v_0 / v_5, 3), f_d(v_0 / v_20, 3)]

                    xx = ['?', '?', '?', '?', '?', '?', '?', '?']

                    tb.append(dvol_v)
                    tb.append(hl_v)
                    tb.append(xx)
                    tb.append(xx)
                    pdf.table(tb, col_widths=[23 for _ in range(8)])
                    # pdf.set_y(y_gap)
                    pdf.image(f"{output_dir}/{index}.png", x=20, y=y_gap, h=height)

                pdf.output(f"{output_dir}/Report_{cur_date}.pdf") 
                logging.info(f"Post report generated for {cur_date}")
