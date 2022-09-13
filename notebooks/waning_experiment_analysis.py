# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: causal-covid-analysis
#     language: python
#     name: causal-covid-analysis
# ---

# # Waning experiment

# ## 1. Imports

# +
import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms

from pathlib import Path
from pprint import pprint
from collections import OrderedDict

from common_functions import (
    load_run,
    infection_incidence,
    infection_incidence_observed,
    severe_case_incidence,
    severe_case_incidence_observed,
)
from constants import (
    FONTSIZE,
    style_modifications,
    PAGE_WIDTH,
    PLOT_HEIGHT,
    PLOT_WIDTH,
    SINGLE_COLUMN,
    DOUBLE_COLUMN,
    FACECOLOR,
    AGE_COLORMAP,
    START_FIRST_WAVE,
    END_FIRST_WAVE,
    START_SECOND_WAVE,
    END_SECOND_WAVE,
    POLICY_NAME_MAP,
    ERROR_PERCENTILE_HIGH,
    ERROR_PERCENTILE_LOW,
    INPUT_DATA_DIR_PATH,
)

plt.rcParams.update(style_modifications)

SAVE_PLOTS = True

WANING_NAME_MAP = OrderedDict({
    "regular-waning": "Regular",
    "fast-waning": "Fast",
    "no-waning": "No Waning",
})
# -

# ## 2. Input: run directory
# The paths below need to be set to the respective local paths.

# +
# path to preprocessed data, i.e. output of ../data_preprocessing/israel_data_processing.py
INPUT_DATA_DIR_PATH = Path("../data/preprocessed-data/israel_df.pkl")

# path to waning experiment output directory, i.e. output of ../experiments/waning_exp.py
EXP_DIR = Path("../run/2022-08-17_15-14-44.102511_waning_exp")

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
# -

RUN_DIRS = {
    x.name: {
        y.name: y 
        for y in x.iterdir() if y.is_dir()
    }
    for x in EXP_DIR.iterdir() if x.is_dir()
}
RUNS = {
    x.name: {
        y.name: load_run(y)
        for y in x.iterdir() if y.is_dir()
    }
    for x in EXP_DIR.iterdir() if x.is_dir()
}
df_input = pd.read_pickle(INPUT_DATA_DIR_PATH)


# ## 3. Plots

# + code_folding=[0]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
        #         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 1.0 * SINGLE_COLUMN),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(4, 4)

    ax0 = fig.add_subplot(gs[:2, :3])
    ax1 = fig.add_subplot(gs[2:, :2])
    ax2 = fig.add_subplot(gs[2:, 2:])

    axes = [ax0, ax1, ax2]

    for i, (label, ax) in enumerate(zip(["(a)", "(b)", "(c)"], axes)):
        if i == 0:
            trans = mtransforms.ScaledTranslation(
                10 / 72, -15 / 72, fig.dpi_scale_trans
            )
        else:
            trans = mtransforms.ScaledTranslation(
                10 / 72, -10 / 72, fig.dpi_scale_trans
            )
        ax.text(
            0.0,
            1.0,
            label,
            transform=ax.transAxes + trans,
            fontsize="medium",
            verticalalignment="top",
            fontfamily="serif",
            bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0, alpha=0.75),
        )

    ls_map = {
        "regular-waning": "-",
        "no-waning": "--",
        "fast-waning": ":",
    }
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        run = RUNS[wp_name][policy_name]
        axes[0].plot(
            run["vaccine_efficacy_params"][:52],
            ls=ls_map[wp_name],
            color="black",
            label=wp_label,
        )
    axes[0].legend(fontsize="small")
    axes[0].set_title("Waning profiles")
    #     axes[0].set_xticks(ticks=run['age_groups'], labels=run['age_group_names'])
    axes[0].set_ylabel("Vaccine efficacy vs. infection\n(two doses BNT162b2)")
    axes[0].set_xlabel("Weeks since second dose")
    axes[0].set_xlim((0, 50))
    axes[0].set_ylim((0, None))

    error_kw = dict(lw=0.75, capsize=1.5, capthick=0.75)
    bar_width = 0.75

    no_infections = {}
    no_infections_high = {}
    no_infections_low = {}
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        no_infections[wp_label] = {}
        no_infections_high[wp_label] = {}
        no_infections_low[wp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[wp_name].keys():
                continue
            run = RUNS[wp_name][policy_name]
            if policy_name == "observed" and wp_name == "regular-waning":
                no_infections[wp_label][policy_label] = infection_incidence_observed(
                    df_input=df_input,
                    population=run["D_a"].sum(),
                )
                no_infections_high[wp_label][policy_label] = None
                no_infections_low[wp_label][policy_label] = None
            else:
                a = np.array(
                    [
                        infection_incidence(
                            infection_dynamics_sample=inf_sample,
                            population=run["D_a"].sum(),
                            week_dates=run["week_dates"],
                        )
                        for inf_sample in run["infection_dynamics_samples"]
                    ]
                )

                no_infections[wp_label][policy_label] = a.mean()
                no_infections_high[wp_label][policy_label] = np.percentile(a, ERROR_PERCENTILE_HIGH) - a.mean()
                no_infections_low[wp_label][policy_label] = a.mean() - np.percentile(a, ERROR_PERCENTILE_LOW)

    df = pd.DataFrame(no_infections)
    df_high = pd.DataFrame(no_infections_high)
    df_low = pd.DataFrame(no_infections_low)
    yerr = np.swapaxes(np.stack((df_low.values, df_high.values)), 0, 1)
    print(df)
    df.T.plot.bar(
        ax=axes[1],
        legend=False,
        yerr=yerr,
        width=bar_width,
        error_kw=error_kw,
    )
    axes[1].set_ylabel("Infections\n(cumulative, per 100k)")
#     axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    axes[1].tick_params(axis="x", labelrotation=45)

    no_severe_cases = {}
    no_severe_cases_high = {}
    no_severe_cases_low = {}
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        no_severe_cases[wp_label] = {}
        no_severe_cases_high[wp_label] = {}
        no_severe_cases_low[wp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[wp_name].keys():
                continue
            run = RUNS[wp_name][policy_name]
            n_weeks = len(run["weeks"])
            if policy_name == "observed" and wp_name == "regular-waning":
                no_severe_cases[wp_label][policy_label] = severe_case_incidence_observed(
                    df_input=df_input,
                    population=run["D_a"].sum(),
                )
                no_severe_cases_high[wp_label][policy_label] = None
                no_severe_cases_low[wp_label][policy_label] = None
            else:
                a = np.array(
                    [
                        severe_case_incidence(
                            res,
                            n_weeks=n_weeks,
                            week_dates=run["week_dates"],
                        )
                        for res in run["result_samples"]
                    ]
                )
                no_severe_cases[wp_label][policy_label] = a.mean()
                no_severe_cases_high[wp_label][policy_label] = (
                    np.percentile(a, ERROR_PERCENTILE_HIGH) - a.mean()
                )
                no_severe_cases_low[wp_label][policy_label] = a.mean() - np.percentile(a, ERROR_PERCENTILE_LOW)

    df = pd.DataFrame(no_severe_cases)
    df_high = pd.DataFrame(no_severe_cases_high)
    df_low = pd.DataFrame(no_severe_cases_low)
    print(df_high)
    print(np.stack((df_low.values, df_high.values)).shape)
    yerr = np.swapaxes(np.stack((df_low.values, df_high.values)), 0, 1)
    df.T.plot.bar(
        ax=axes[2],
        legend=False,
        yerr=yerr,
        width=bar_width,
        error_kw=error_kw,
    )
    axes[2].set_ylabel("Severe cases\n(cumulative, per 100k)")
#     axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].tick_params(axis="x", labelrotation=45)
    axes[2].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    h, l = axes[2].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=1,
        loc="lower right",
        bbox_to_anchor=(0.9925, 0.62),
        title="Vaccine\nallocation\nstrategy",
        fontsize="small",
    )

    if save:
        plt.savefig(OUTPUT_DIR / "waning_exp_overview3.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
