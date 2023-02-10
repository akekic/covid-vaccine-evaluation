# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import OrderedDict
from pathlib import Path

mm = 1/(10 * 2.54)  # millimeters in inches
SINGLE_COLUMN = 85 * mm
DOUBLE_COLUMN = 174 * mm
PAGE_WIDTH = 8.25  # A4 width in inches
PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4
TEXT_WIDTH_WORKSHOP = 5.5
FACECOLOR = "gainsboro"
AGE_COLORMAP = plt.cm.Oranges
VAC_COLORMAP = plt.cm.Purples

FONTSIZE = 7
# ggplot colors
# color_list = ['#E24A33', '#348ABD', '#988ED5', '#777777', '#FBC15E', '#8EBA42', '#FFB5B8']

# bright qualitative
# https://personal.sron.nl/~pault/
# color_list = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']

# vibrant qualitative
# https://personal.sron.nl/~pault/
# color_list = ['#EE7733', '#0077BB', '#009988', '#EE3377', '#CC3311', '#33BBEE', '#BBBBBB']

# Petroff
# https://github.com/mpetroff/accessible-color-cycles
color_list = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
# color_list = ["#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd", "#578dff", "#656364"]

# linestyle cycle
linestyle_list = ['--', '-', ':', '-.', '--', '-']

style_modifications = {
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "font.family": "sans-serif", # set the font globally
    "font.sans-serif": "Helvetica", # set the font name for a font family
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "axes.labelcolor": "black",
    "axes.prop_cycle": mpl.cycler(color=color_list, linestyle=linestyle_list),
    "xtick.color": "black",
    "ytick.color": "black",
    "errorbar.capsize": 2.5, 
}

START_FIRST_WAVE = datetime.date(year=2020, month=12, day=20)
END_FIRST_WAVE = datetime.date(year=2021, month=4, day=11)

START_SECOND_WAVE = datetime.date(year=2021, month=6, day=20)
END_SECOND_WAVE = datetime.date(year=2021, month=11, day=7)

ERROR_PERCENTILE_HIGH = 97.5
ERROR_PERCENTILE_LOW = 2.5


POLICY_NAME_MAP = OrderedDict({
    "elderly_first": "ElderlyFirst",
    "observed": "Factual",
    "uniform": "Uniform",
    "young_first": "YoungFirst",
    "risk_ranked": "RiskRanked",
    "risk_ranked_reversed": "RiskRankedReversed",
})
INPUT_DATA_DIR = Path("../data/preprocessed-data")
INPUT_DATA_DIR_PATH = Path("../data/preprocessed-data") / "israel_df.pkl"
# -


