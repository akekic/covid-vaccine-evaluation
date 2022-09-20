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

# # Risk profile experiment

# ## 1. Imports

# +
import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker

from pathlib import Path
# from pprint import pprint
from collections import OrderedDict
# from itertools import product

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
    TEXT_WIDTH_WORKSHOP,
    FACECOLOR,
    AGE_COLORMAP,
    START_FIRST_WAVE,
    END_FIRST_WAVE,
    START_SECOND_WAVE,
    END_SECOND_WAVE,
    POLICY_NAME_MAP,
    ERROR_PERCENTILE_HIGH,
    ERROR_PERCENTILE_LOW,
)
plt.rcParams.update(style_modifications)
SAVE_PLOTS = True

PROFILE_NAME_MAP = OrderedDict(
    {
        "covid": "COVID-19",
        "spanish": "Spanish Flu",
        "flat": "Flat Profile",
    }
)
END_DATE = np.datetime64(END_FIRST_WAVE)
# -

# ## 2. Input: run directory
#
# The two paths below need to be set to the respective local paths.

# +
# path to preprocessed data, i.e. output of ../data_preprocessing/israel_data_processing.py
INPUT_DATA_DIR_PATH = Path("../data/preprocessed-data/israel_df.pkl")

# path to risk profile experiment output directory, i.e. output of ../experiments/risk_profile_exp.py
EXP_DIR = Path("../run/2022-08-16_14-31-01.669255_risk_profile_exp")

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

RUNS.keys()


# ## 3. Plots

# + code_folding=[]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
        figsize=(SINGLE_COLUMN, 0.9 * SINGLE_COLUMN),
        dpi=500,
        constrained_layout=False,
    )
    gs = gridspec.GridSpec(6, 3)

    error_kw = dict(lw=0.75, capsize=2, capthick=0.75)
    bar_width = 0.75

    ax00 = fig.add_subplot(gs[:2, 0])
    ax01 = fig.add_subplot(gs[:2, 1], sharey=ax00)
    ax02 = fig.add_subplot(gs[:2, 2], sharey=ax00)

    ax00.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    ax01.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    ax02.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    ax00.set_xlabel(r"Age group")
    ax01.set_xlabel(r"Age group")
    ax02.set_xlabel(r"Age group")

    profile_axes = [ax00, ax01, ax02]
    for i, (rp_name, rp_label) in enumerate(PROFILE_NAME_MAP.items()):
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        run = RUNS[rp_name][policy_name]
        df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
        df_g[0].plot.bar(ax=profile_axes[i], color="black", width=0.8)
        profile_axes[i].set_title(rp_label)
        profile_axes[i].set_ylabel("Relative risk")

    ax1 = fig.add_subplot(gs[2:4, :])
    no_infections = OrderedDict()
    no_infections_high = {}
    no_infections_low = {}
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_infections[rp_label] = OrderedDict()
        no_infections_high[rp_label] = {}
        no_infections_low[rp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            #             print(f"{rp_name}, {policy_name}")
            if policy_name not in RUNS[rp_name].keys():
                continue
            if rp_name == "flat" and policy_name in [
                "risk_ranked",
                "risk_ranked_reversed",
            ]:
                run = RUNS[rp_name]["uniform"]
            else:
                run = RUNS[rp_name][policy_name]
            if policy_name == "observed" and rp_name == "covid":
                no_infections[rp_label][policy_label] = infection_incidence_observed(
                    df_input=df_input,
                    population=run["D_a"].sum(),
                    split_date_to=END_DATE,
                )
                no_infections_high[rp_label][policy_label] = None
                no_infections_low[rp_label][policy_label] = None
            else:
                a = np.array(
                    [
                        infection_incidence(
                            infection_dynamics_sample=inf_sample,
                            population=run["D_a"].sum(),
                            week_dates=run["week_dates"],
                            split_date_to=END_DATE,
                        )
                        for inf_sample in run["infection_dynamics_samples"]
                    ]
                )

                no_infections[rp_label][policy_label] = a.mean()
                no_infections_high[rp_label][policy_label] = np.percentile(a, ERROR_PERCENTILE_HIGH) - a.mean()
                no_infections_low[rp_label][policy_label] = a.mean() - np.percentile(a, ERROR_PERCENTILE_LOW)

    df = pd.DataFrame(no_infections)
    df_high = pd.DataFrame(no_infections_high)
    df_low = pd.DataFrame(no_infections_low)
    yerr = np.swapaxes(np.stack((df_low.values, df_high.values)), 0, 1)

    bar_plot2 = df.T.plot.bar(
        ax=ax1,
        legend=False,
        yerr=yerr,
        width=bar_width,
        error_kw=error_kw,
    )
    ax1.set_ylabel("Infections\n(cum., per 100k)")
    ax1.tick_params(axis="x", labelrotation=0)
    ax1.set_ylim([0, 9.5e3])
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    ax2 = fig.add_subplot(gs[4:, :], sharex=ax1)

    no_severe_cases = OrderedDict()
    no_severe_cases_high = OrderedDict()
    no_severe_cases_low = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_severe_cases[rp_label] = OrderedDict()
        no_severe_cases_high[rp_label] = OrderedDict()
        no_severe_cases_low[rp_label] = OrderedDict()
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[rp_name].keys():
                continue
            if rp_name == "flat" and policy_name in [
                "risk_ranked",
                "risk_ranked_reversed",
            ]:
                run = RUNS[rp_name]["uniform"]
            else:
                run = RUNS[rp_name][policy_name]
            
            if policy_name == "observed" and rp_name == "covid":
                no_severe_cases[rp_label][policy_label] = severe_case_incidence_observed(
                    df_input=df_input,
                    population=run["D_a"].sum(),
                    split_date_to=END_DATE,
                )
                no_severe_cases_high[rp_label][policy_label] = None
                no_severe_cases_low[rp_label][policy_label] = None
            else:
                a = np.array(
                    [
                        severe_case_incidence(
                            res,
                            n_weeks=len(run["weeks"]),
                            week_dates=run["week_dates"],
                            split_date_to=END_DATE,
                        )
                        for res in run["result_samples"]
                    ]
                )
                no_severe_cases[rp_label][policy_label] = a.mean()
                no_severe_cases_high[rp_label][policy_label] = (
                    np.percentile(a, ERROR_PERCENTILE_HIGH) - a.mean()
                )
                no_severe_cases_low[rp_label][policy_label] = a.mean() - np.percentile(a, ERROR_PERCENTILE_LOW)

    df = pd.DataFrame(no_severe_cases)
    df_high = pd.DataFrame(no_severe_cases_high)
    df_low = pd.DataFrame(no_severe_cases_low)
    yerr = np.swapaxes(np.stack((df_low.values, df_high.values)), 0, 1)
    df.T.plot.bar(
        ax=ax2,
        legend=False,
        yerr=yerr,
        width=bar_width,
        error_kw=error_kw,
    )

    ax2.set_ylabel("Severe cases\n(cum., per 100k)")
    ax2.tick_params(axis="x", labelrotation=0)
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    axes = [ax00, ax01, ax02, ax1, ax2]
    trans = mtransforms.ScaledTranslation(5 / 72, -5 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)"], axes):
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

    h, l = ax1.get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(0.98, 1.0),
        title="Vaccine allocation strategy",
        fontsize="small",
    )

    if save:
        plt.savefig(OUTPUT_DIR / "risk_profile_exp_overview4.pdf")
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
        figsize=(0.9 * TEXT_WIDTH_WORKSHOP, 0.375 * TEXT_WIDTH_WORKSHOP),
        dpi=500,
        constrained_layout=False,
    )
    gs = gridspec.GridSpec(6, 8, wspace=1, hspace=8)

    ax00 = fig.add_subplot(gs[:2, :2])
    ax01 = fig.add_subplot(gs[2:4, :2], sharex=ax00, sharey=ax00)
    ax02 = fig.add_subplot(gs[4:, :2], sharex=ax00, sharey=ax00)

    ax00.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    ax01.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    ax02.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )
    ax02.set_xlabel(r"$A$", usetex=True)

    profile_axes = [ax00, ax01, ax02]
    for i, (rp_name, rp_label) in enumerate(PROFILE_NAME_MAP.items()):
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        run = RUNS[rp_name][policy_name]
        df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
        df_g[0].plot.bar(ax=profile_axes[i], color="black", width=0.8)
        profile_axes[i].set_title(rp_label, pad=3)
        profile_axes[i].set_ylabel(r"$g(V{=}0, A)$", usetex=True)
        profile_axes[i].set_xlabel(r"Age group $A$", usetex=True)

    ax1 = fig.add_subplot(gs[:3, 3:])
    no_infections = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_infections[rp_label] = OrderedDict()
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[rp_name].keys():
                continue
            if rp_name == "flat" and policy_name in [
                "risk_ranked",
                "risk_ranked_reversed",
            ]:
                run = RUNS[rp_name]["uniform"]
            else:
                run = RUNS[rp_name][policy_name]
            population = run["D_a"].sum()
            infection_dynamics_df = run["infection_dynamics_df"]
            infection_dynamics_df['Sunday_date'] = pd.to_datetime(infection_dynamics_df['Sunday_date'])
            infection_dynamics_df = infection_dynamics_df[infection_dynamics_df['Sunday_date'] < END_DATE]
            infection_dynamics_total = infection_dynamics_df[
                infection_dynamics_df["Age_group"] == "total"
            ]
            # TODO: remove hack
            if infection_dynamics_total.empty:
                infection_dynamics_total = (
                    infection_dynamics_df.groupby("Sunday_date")[
                        "total_infections_scenario"
                    ]
                    .sum()
                    .reset_index()
                )
            no_infections[rp_label][policy_label] = infection_dynamics_total[
                "total_infections_scenario"
            ].sum() * 1e5 / (population ) 
    df = pd.DataFrame(no_infections)
    bar_plot2 = df.T.plot.bar(ax=ax1, legend=False)
    ax1.set_ylabel("Infections\n(cum., per 100k)")
    ax1.tick_params(axis="x", labelrotation=0)
    ax1.set_ylim([0, 8.1e3])
    
    
    ax2 = fig.add_subplot(gs[3:, 3:], sharex=ax1)
    no_severe_cases = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_severe_cases[rp_label] = OrderedDict()
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[rp_name].keys():
                continue
            if rp_name == "flat" and policy_name in [
                "risk_ranked",
                "risk_ranked_reversed",
            ]:
                run = RUNS[rp_name]["uniform"]
            else:
                run = RUNS[rp_name][policy_name]
            population = run["D_a"].sum()
            end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
            n_weeks = len(run["weeks"])
            no_severe_cases[rp_label][policy_label] = (
                n_weeks * run["result"][:end_date_index, ...].sum() * (1e5)
            )
    df = pd.DataFrame(no_severe_cases)
    df.T.plot.bar(ax=ax2, legend=False)
    ax2.set_ylabel("Severe cases\n(cum., per 100k)")
    ax2.tick_params(axis="x", labelrotation=0)
    
    axes = [ax00, ax01, ax02, ax1, ax2]
    trans = mtransforms.ScaledTranslation(5 / 72, -5 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)"], axes):
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

    h, l = ax1.get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=6,
        loc="lower right",
        bbox_to_anchor=(0.910, 0.95),
        title="Vaccine allocation strategy",
        title_fontsize="medium",
        fontsize='x-small',
    )

    if save:
        plt.savefig(OUTPUT_DIR / "risk_profile_exp_overview_w.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
# -

