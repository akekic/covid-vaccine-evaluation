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

# +
import os
import import_ipynb

import pandas as pd
import numpy as np
import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
import matplotlib.ticker as mticker

from pathlib import Path
from pprint import pprint
from collections import OrderedDict
from itertools import product

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
    INPUT_DATA_DIR_PATH,
)

# plt.style.use("ggplot")
plt.rcParams.update(style_modifications)

LW_GLOBAL = 1
MS_GLOBAL = 3

SAVE_PLOTS = True

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

PROFILE_NAME_MAP = OrderedDict(
    {
        "covid": "COVID-19",
        "spanish": "Spanish Flu",
        "flat": "Flat Profile",
    }
)

END_DATE = np.datetime64(datetime.date(year=2021, month=4, day=4))



# latest results

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_16-52-03.602714_risk_profile_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_16-52-03.142740_risk_profile_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_16-52-00.311194_risk_profile_exp")

# error bars

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-08-20_09-51-38.947081_risk_profile_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
EXP_NAME, EXP_DIR = "exp", Path("../run/2022-08-16_14-31-01.669255_risk_profile_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-08-20_09-51-38.955053_risk_profile_exp")
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


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(1, 3, figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH), dpi=500, sharey=False, sharex=False)
    for rp, policies in RUNS.items():
        for p, run in policies.items():
            axes[0].plot(run['age_groups'], run['g'][0, :], label=rp)
            break
    axes[0].legend()
    axes[0].set_title("risk factors (unvaccinated)")
    axes[0].set_xticks(ticks=run['age_groups'], labels=run['age_group_names'])
    axes[0].set_ylabel('risk factor')
    axes[0].set_xlabel('age group')



    no_severe_cases = {}
    for rp, policies in RUNS.items():
        no_severe_cases[rp] = {}
        for p, run in policies.items():
            population = run['D_a'].sum()
            n_weeks = len(run['weeks'])
            no_severe_cases[rp][p] = population * n_weeks * run['result'].sum()
    df = pd.DataFrame(no_severe_cases)
    df.T.plot.bar(ax=axes[1], legend=False)
    axes[1].set_title("total number of severe cases")
    
    no_infections = {}
    for rp, policies in RUNS.items():
        no_infections[rp] = {}
        for p, run in policies.items():
            infection_dynamics_df = run['infection_dynamics_df']
            infection_dynamics_df = infection_dynamics_df[infection_dynamics_df['Age_group'] == 'total']
            no_infections[rp][p] = infection_dynamics_df['total_infections_scenario'].sum()
    df = pd.DataFrame(no_infections)
    df.T.plot.bar(ax=axes[2])
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[2].set_title("total number of infections")

# plot()

# + code_folding=[0]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
#         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(DOUBLE_COLUMN, 0.3 * DOUBLE_COLUMN),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(3, 8)
    gs.update(wspace=1.9, hspace=0.9)

    ax00 = fig.add_subplot(gs[0, :2])
    ax01 = fig.add_subplot(gs[1, :2], sharex=ax00, sharey=ax00)
    ax02 = fig.add_subplot(gs[2, :2], sharex=ax00, sharey=ax00)

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
    #     ax00.axes.xaxis.set_ticklabels([])
    #     ax00.axes.yaxis.set_ticklabels([])
    ax02.set_xlabel(r"$A$", usetex=True)

    profile_axes = [ax00, ax01, ax02]
    for i, (rp_name, rp_label) in enumerate(PROFILE_NAME_MAP.items()):
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        run = RUNS[rp_name][policy_name]
        df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
        df_g[0].plot.bar(ax=profile_axes[i], color="black")
        profile_axes[i].set_title(rp_label)
        profile_axes[i].set_ylabel(r"$g(V{=}0, A)$", usetex=True)

    ax1 = fig.add_subplot(gs[:3, 2:5])
    no_infections = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_infections[rp_label] = OrderedDict()
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
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title("Infections")
    ax1.set_ylabel("Cumulative Infections (per 100k)")
    ax1.tick_params(axis="x", labelrotation=0)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    
    
    ax2 = fig.add_subplot(gs[:3, 5:9])
    no_severe_cases = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_severe_cases[rp_label] = OrderedDict()
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
            population = run["D_a"].sum()
            end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
            n_weeks = len(run["weeks"])
            no_severe_cases[rp_label][policy_label] = (
                n_weeks * run["result"][:end_date_index, ...].sum() * (1e5)
            )
    df = pd.DataFrame(no_severe_cases)
    print(df)
    df.T.plot.bar(ax=ax2, legend=False)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax1.legend()
    ax2.set_title("Severe Cases")
    ax2.set_ylabel("Cumulative Severe Cases (per 100k)")
    ax2.tick_params(axis="x", labelrotation=0)

    h, l = ax1.get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=6,
        loc="lower right",
        bbox_to_anchor=(0.905, 1.0),
        title="Vaccine Allocation Strategy",
        fontsize='small',
    )

    if save:
        plt.savefig(OUTPUT_DIR / "risk_profile_exp_overview.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
# + code_folding=[0, 105]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
#         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 0.75 * SINGLE_COLUMN),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(6, 8, wspace=2, hspace=8)
#     gs.update(wspace=4, hspace=2.0)

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
    #     ax00.axes.xaxis.set_ticklabels([])
    #     ax00.axes.yaxis.set_ticklabels([])
    ax02.set_xlabel(r"$A$", usetex=True)

    profile_axes = [ax00, ax01, ax02]
    for i, (rp_name, rp_label) in enumerate(PROFILE_NAME_MAP.items()):
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        run = RUNS[rp_name][policy_name]
        df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
        df_g[0].plot.bar(ax=profile_axes[i], color="black", width=0.8)
        profile_axes[i].set_title(rp_label)
        profile_axes[i].set_ylabel(r"$g(V{=}0, A)$", usetex=True)

    ax1 = fig.add_subplot(gs[:3, 3:])
    no_infections = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_infections[rp_label] = OrderedDict()
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
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title("Infections")
    ax1.set_ylabel("Infections\n(cumulative, per 100k)")
    ax1.tick_params(axis="x", labelrotation=0)
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax1.set_ylim([0, 8.1e3])
    
    
    ax2 = fig.add_subplot(gs[3:, 3:], sharex=ax1)
    no_severe_cases = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_severe_cases[rp_label] = OrderedDict()
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
            population = run["D_a"].sum()
            end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
            n_weeks = len(run["weeks"])
            no_severe_cases[rp_label][policy_label] = (
                n_weeks * run["result"][:end_date_index, ...].sum() * (1e5)
            )
    df = pd.DataFrame(no_severe_cases)
    print(df)
    df.T.plot.bar(ax=ax2, legend=False)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # ax1.legend()
#     ax2.set_title("Severe Cases")
    ax2.set_ylabel("Severe cases\n(cumulative, per 100k)")
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
        ncol=3,
        loc="lower right",
        bbox_to_anchor=(0.915, 1.0),
        title="Vaccine Allocation Strategy",
        fontsize='small',
    )

    if save:
        plt.savefig(OUTPUT_DIR / "risk_profile_exp_overview2.pdf")
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[0]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
#         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 0.9 * SINGLE_COLUMN),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(6, 3)
#     gs.update(wspace=4, hspace=2.0)

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
    #     ax00.axes.xaxis.set_ticklabels([])
    #     ax00.axes.yaxis.set_ticklabels([])
    ax00.set_xlabel(r"$A$", usetex=True)
    ax01.set_xlabel(r"$A$", usetex=True)
    ax02.set_xlabel(r"$A$", usetex=True)

    profile_axes = [ax00, ax01, ax02]
    for i, (rp_name, rp_label) in enumerate(PROFILE_NAME_MAP.items()):
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        run = RUNS[rp_name][policy_name]
        df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
        df_g[0].plot.bar(ax=profile_axes[i], color="black", width=0.8)
        profile_axes[i].set_title(rp_label)
        profile_axes[i].set_ylabel(r"$g(V{=}0, A)$", usetex=True)

    ax1 = fig.add_subplot(gs[2:4, :])
    no_infections = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_infections[rp_label] = OrderedDict()
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
    
    
    ax2 = fig.add_subplot(gs[4:, :], sharex=ax1)
    no_severe_cases = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_severe_cases[rp_label] = OrderedDict()
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
            population = run["D_a"].sum()
            end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
            n_weeks = len(run["weeks"])
            no_severe_cases[rp_label][policy_label] = (
                n_weeks * run["result"][:end_date_index, ...].sum() * (1e5)
            )
    df = pd.DataFrame(no_severe_cases)
    print(df)
    df.T.plot.bar(ax=ax2, legend=False)

    ax2.set_ylabel("Severe cases\n(cum., per 100k)")
    ax2.tick_params(axis="x", labelrotation=0)
    ax2.xaxis.set_minor_locator(mticker.NullLocator())
    
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
        fontsize='small',
    )

    if save:
        plt.savefig(OUTPUT_DIR / "risk_profile_exp_overview3.pdf")
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[0]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
        #         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
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
    ax1.set_ylim([0, 8.1e3])
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
    print(df)
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


# + code_folding=[0]
def plot(runs_per_policy, run_name="run_name", save=False, path=None):
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=False,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        constrained_layout=False,
    )
    plt.suptitle(run_name)
    gs = gridspec.GridSpec(3, 6, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)

    ax = fig.add_subplot(gs[:3, :3])
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name in runs_per_policy.keys():
            run = runs_per_policy[policy_name]
            if run_name == "flat" and policy_name in ['risk_ranked', 'risk_ranked_reversed']:
                run = runs_per_policy["uniform"]
            id_df = run["infection_dynamics_df"]
            id_df = id_df[id_df['Sunday_date'] < END_DATE]
#             print(id_df.head(10))  # TODO: new runs have total infections
            population = run["D_a"].sum()
            ax.plot(
                id_df[id_df["Age_group"] == "total"]['Sunday_date'],
                id_df[id_df["Age_group"] == "total"]["total_infections_scenario"]
                * 1e5
                / population,
                label=policy_label,
                lw=LW_LOCAL,
            )
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("weekly infections (per 100k)")
    ax.set_title("total", loc="right", pad=-FONTSIZE, y=1.0)

    index_matrix = np.arange(9).reshape((3, 3))
    for i in range(9):
        indices = np.argwhere(index_matrix == i)[0]
        if i == 0:
            ax = fig.add_subplot(gs[indices[0], 3 + indices[1]])
            initial_ax = ax
        else:
            ax = fig.add_subplot(
                gs[indices[0], 3 + indices[1]],
                sharex=initial_ax,
                #                 sharey=initial_ax,
            )
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name in runs_per_policy.keys():
                run = runs_per_policy[policy_name]
                if run_name == "flat" and policy_name in ['risk_ranked', 'risk_ranked_reversed']:
                    run = runs_per_policy["uniform"]
                id_df = run["infection_dynamics_df"]
                id_df = id_df[id_df['Sunday_date'] < END_DATE]
                population_age_group = run["D_a"][i]
                ax.plot(
                    id_df[id_df["Age_group"] == run['age_group_names'][i]]["Sunday_date"],
                    id_df[id_df["Age_group"] == run['age_group_names'][i]]["total_infections_scenario"]
                    * 1e5
                    / population_age_group,
                    label=policy_label,
                    lw=LW_LOCAL,
                )
        ax.set_title(run["age_group_names"][i], loc="right", pad=-FONTSIZE, y=1.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)

        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(path)
    plt.show()

# for rp_name, rp_label in PROFILE_NAME_MAP.items():
#     plot(RUNS[rp_name], run_name=rp_name)


# + code_folding=[0]
def plot(runs_per_policy, run_name="run_name", save=False, path=None):
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=False,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        constrained_layout=False,
    )
    plt.suptitle(run_name)
    gs = gridspec.GridSpec(3, 6, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)

    ax = fig.add_subplot(gs[:3, :3])
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name in runs_per_policy.keys():
            run = runs_per_policy[policy_name]
            if run_name == "flat" and policy_name in ['risk_ranked', 'risk_ranked_reversed']:
                run = runs_per_policy["uniform"]
            end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
            population = run["D_a"].sum()
            n_weeks = len(run["weeks"])
            ax.plot(
                run["week_dates"][:end_date_index],
                population * n_weeks * run["result"][:end_date_index, ...].sum(axis=(1, 2)) / (1e5 * population),
                label=policy_label,
                lw=LW_LOCAL,
            )
    ax.legend()
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("weekly severe cases (per 100k)")
    ax.set_title("total", loc="right", pad=-FONTSIZE, y=1.0)

    index_matrix = np.arange(9).reshape((3, 3))
    for i in range(9):
        indices = np.argwhere(index_matrix == i)[0]
        if i == 0:
            ax = fig.add_subplot(gs[indices[0], 3 + indices[1]])
            initial_ax = ax
        else:
            ax = fig.add_subplot(
                gs[indices[0], 3 + indices[1]],
                sharex=initial_ax,
                #                 sharey=initial_ax,
            )
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name in runs_per_policy.keys():
                run = runs_per_policy[policy_name]
                if run_name == "flat" and policy_name in ['risk_ranked', 'risk_ranked_reversed']:
                    run = runs_per_policy["uniform"]
                end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
                population_age_group = run["D_a"][i]
                n_weeks = len(run["weeks"])
                ax.plot(
                    run["week_dates"][:end_date_index],
                    population
                    * n_weeks
                    * run["result"][:end_date_index, i, :].sum(axis=1)
                    / (1e5 * population_age_group),
                    label=policy_label,
                    lw=LW_LOCAL,
                )
        ax.set_title(run["age_group_names"][i], loc="right", pad=-FONTSIZE, y=1.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)

        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(path)
    plt.show()

# for rp_name, rp_label in PROFILE_NAME_MAP.items():
#     plot(RUNS[rp_name], run_name=rp_name)


# + code_folding=[0]
def plot(runs_per_policy, run_name="run_name", save=False, path=None):
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=False,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        constrained_layout=False,
    )
    plt.suptitle(run_name)
    gs = gridspec.GridSpec(3, 6, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)

    ax = fig.add_subplot(gs[:3, :3])
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name in runs_per_policy.keys():
            if policy_name != "observed":
                continue
            run = runs_per_policy[policy_name]
            if run_name == "flat" and policy_name in [
                "risk_ranked",
                "risk_ranked_reversed",
            ]:
                run = runs_per_policy["uniform"]
            id_df = run["infection_dynamics_df"]
            #             print(id_df.head(10))  # TODO: new runs have total infections
            population = run["D_a"].sum()
            n_weeks = len(run["weeks"])
            ax.plot(
                run["week_dates"],
                population
                * n_weeks
                * run["result"].sum(axis=(1, 2))
                / id_df[id_df["Age_group"] == "total"]["total_infections_scenario"],
                label=policy_label,
                lw=LW_LOCAL,
            )
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("infection severity rate")
    ax.set_title("total", loc="right", pad=-FONTSIZE, y=1.0)

    index_matrix = np.arange(9).reshape((3, 3))
    for i in range(9):
        indices = np.argwhere(index_matrix == i)[0]
        if i == 0:
            ax = fig.add_subplot(gs[indices[0], 3 + indices[1]])
            initial_ax = ax
        else:
            ax = fig.add_subplot(
                gs[indices[0], 3 + indices[1]],
                sharex=initial_ax,
                #                 sharey=initial_ax,
            )
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name in runs_per_policy.keys():
                if policy_name != "observed":
                    continue
                run = runs_per_policy[policy_name]
                if run_name == "flat" and policy_name in [
                    "risk_ranked",
                    "risk_ranked_reversed",
                ]:
                    run = runs_per_policy["uniform"]
                id_df = run["infection_dynamics_df"]
                population_age_group = run["D_a"][i]
                ax.plot(
                    run["week_dates"],
                    population
                    * n_weeks
                    * run["result"][:, i, :].sum(axis=1)
                    / id_df[id_df["Age_group"] == run['age_group_names'][i]]["total_infections_scenario"],
                    label=policy_label,
                    lw=LW_LOCAL,
                )
        ax.set_title(run["age_group_names"][i], loc="right", pad=-FONTSIZE, y=1.0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)

        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(path)
    plt.show()


# for rp_name, rp_label in PROFILE_NAME_MAP.items():
#     plot(RUNS[rp_name], run_name=rp_name)

# + code_folding=[0]
def compute_waning_times(run):
    weeks_extended = run['weeks_extended']
    weeks = run['weeks']
    age_groups = run['age_groups']
    U_2 = run['U_2']
    u_3 = run['u_3']
    P_t = run['P_t']
    P_a = run['P_a']
    D_a = run['D_a']
    vaccination_statuses = run['vaccination_statuses']
    
    w_t_a_v = np.zeros(
            (
                len(weeks),
                len(age_groups),
                len(vaccination_statuses),
            )
        )
    
    
    for t, a in product(weeks, age_groups):
        # unvaccinated
        tmp_0 = 0
        for t1, t2, t3 in product(
            weeks_extended[t + 1 :],
            weeks_extended[t + 1 :],
            weeks_extended[:],
        ):
            pass
        tmp_0 = 0
        w_t_a_v[t, a, 0] = tmp_0

        # after 1st dose
        tmp_1 = 0
        for t1, t2, t3 in product(
            weeks_extended[: t + 1],
            weeks_extended[t + 1 :],
            weeks_extended[:],
        ):
            waning_time = t - t1
#             waning_time = 1
            tmp_1 += U_2[a, t1, t2] * u_3[a, t2, t3] * waning_time
        tmp_1 *= P_a[a] / D_a[a]
        w_t_a_v[t, a, 1] = tmp_1

        # after 2nd dose
        tmp_2 = 0
        for t1, t2, t3 in product(
            weeks_extended[: t + 1],
            weeks_extended[: t + 1],
            weeks_extended[t + 1 :],
        ):
            waning_time = t - t2
#             waning_time = 1
            tmp_2 += U_2[a, t1, t2] * u_3[a, t2, t3] * waning_time
        tmp_2 *= P_a[a] / D_a[a]
        w_t_a_v[t, a, 2] = tmp_2

        # after 3rd dose
        tmp_3 = 0
        for t1, t2, t3 in product(
            weeks_extended[: t + 1],
            weeks_extended[: t + 1],
            weeks_extended[: t + 1],
        ):
            waning_time = t - t3
#             waning_time = 1
            tmp_3 += U_2[a, t1, t2] * u_3[a, t2, t3] * waning_time
        tmp_3 *= P_a[a] / D_a[a]
        w_t_a_v[t, a, 3] = tmp_3

    return w_t_a_v

# +
# w = compute_waning_times(RUNS['flat']['uniform'])
# w.shape

# + code_folding=[0]
# fig = plt.figure(
#     tight_layout=False,
#     figsize=(0.25 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
#     dpi=500,
#     constrained_layout=False,
# )
# LW_LOCAL = 0.8
# for policy_name, policy_label in POLICY_NAME_MAP.items():
#     if policy_name in ["risk_ranked", "risk_ranked_reversed", "uniform", "elderly_first"]:
#         continue
#     run = RUNS["flat"][policy_name]
#     w = compute_waning_times(run)
#     ls = "--" if policy_name == "young_first" else "-"
#     for v in range(1, 4):
#         plt.plot(
#             run["week_dates"],
#             w[..., v].sum(axis=(1)),
#             label=f"{policy_label}, v={v}",
#             lw=LW_LOCAL,
#             ls=ls,
#         )
# plt.ylim((0, 3))
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.show()
# -

RUNS["flat"]["uniform"].keys()

RUNS["flat"]["uniform"]["result"].shape

# +
fig = plt.figure(
    tight_layout=False,
    figsize=(0.25 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
    dpi=500,
    constrained_layout=False,
)
run = RUNS["flat"]["uniform"]
colors = AGE_COLORMAP(np.linspace(0, 1, len(run["age_group_names"])))
end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]


for ag in run["age_groups"]:
    plt.plot(
        run["week_dates"][:end_date_index],
        run["f_1"][ag, :end_date_index],
        label=run["age_group_names"][ag],
        c=colors[ag],
    )
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()
# + code_folding=[]
def plot(save):
    fig = plt.figure(
        tight_layout=True,
#         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(0.9 * TEXT_WIDTH_WORKSHOP, 0.375 * TEXT_WIDTH_WORKSHOP),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(6, 8, wspace=1, hspace=8)
#     gs.update(wspace=4, hspace=2.0)

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
    #     ax00.axes.xaxis.set_ticklabels([])
    #     ax00.axes.yaxis.set_ticklabels([])
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
#         profile_axes[i].set_ylabel(r"Relative risk", usetex=True)

    ax1 = fig.add_subplot(gs[:3, 3:])
    no_infections = OrderedDict()
    for rp_name, rp_label in PROFILE_NAME_MAP.items():
        no_infections[rp_label] = OrderedDict()
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
            population = run["D_a"].sum()
            end_date_index = np.argwhere(run["week_dates"] == END_DATE).flatten()[0]
            n_weeks = len(run["weeks"])
            no_severe_cases[rp_label][policy_label] = (
                n_weeks * run["result"][:end_date_index, ...].sum() * (1e5)
            )
    df = pd.DataFrame(no_severe_cases)
    print(df)
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



