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

# plt.style.use("ggplot")
plt.rcParams.update(style_modifications)

LW_GLOBAL = 1
MS_GLOBAL = 3

SAVE_PLOTS = True

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

WANING_NAME_MAP = OrderedDict({
    "regular-waning": "Regular",
    "fast-waning": "Fast",
    "no-waning": "No Waning",
})


# latest results

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_15-44-17.236026_waning_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_15-44-14.870299_waning_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_15-44-12.590993_waning_exp")

# error bars

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_15-44-17.236026_waning_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
EXP_NAME, EXP_DIR = "exp", Path("../run/2022-08-17_15-14-44.102511_waning_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_15-44-12.590993_waning_exp")
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


# + code_folding=[]
def plot(save):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN, 0.3 * DOUBLE_COLUMN),
        dpi=500,
        sharey=False,
        sharex=False,
        tight_layout=True,
    )
    plt.subplots_adjust(hspace=0.15, wspace=0.5)
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
    axes[0].set_title("Waning Profiles")
    #     axes[0].set_xticks(ticks=run['age_groups'], labels=run['age_group_names'])
    axes[0].set_ylabel("Vaccine Efficacy Against Infection\n(Two Doses BNT162b2)")
    axes[0].set_xlabel("Weeks Since Second Dose")

    no_infections = {}
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        no_infections[wp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[wp_name].keys():
                continue
            run = RUNS[wp_name][policy_name]
            population = run["D_a"].sum()
            infection_dynamics_df = run["infection_dynamics_df"]
            infection_dynamics_df = infection_dynamics_df[
                infection_dynamics_df["Age_group"] == "total"
            ]
            no_infections[wp_label][policy_label] = infection_dynamics_df[
                "total_infections_scenario"
            ].sum() * 1e5 / (population ) 
    df = pd.DataFrame(no_infections)
    print(df)
    df.T.plot.bar(ax=axes[1], legend=False)
    #     axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].set_title("Infections")
    axes[1].set_ylabel("Cumulative Infections (per 100k)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].tick_params(axis="x", labelrotation=0)
    
    
    no_severe_cases = {}
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        no_severe_cases[wp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[wp_name].keys():
                continue
            run = RUNS[wp_name][policy_name]
            population = run["D_a"].sum()
            n_weeks = len(run["weeks"])
            no_severe_cases[wp_label][policy_label] = (
                n_weeks * run["result"].sum() *1e5
            )
    df = pd.DataFrame(no_severe_cases)
    print(df)
    df.T.plot.bar(ax=axes[2], legend=False)
    axes[2].set_title("Severe Cases")
    axes[2].set_ylabel("Cumulative Severe Cases (per 100k)")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].tick_params(axis="x", labelrotation=0)

    h, l = axes[2].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=5,
        loc="lower right",
        bbox_to_anchor=(0.9925, 1.0),
        title="Vaccine Allocation Strategy",
    )

    if save:
        plt.savefig(OUTPUT_DIR / "waning_exp_overview.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
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
    #     gs.update(
    # #         wspace=1.9,
    #         hspace=2.0,
    #     )

    #     plt.subplots_adjust(hspace=0.15, wspace=0.5)

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
    axes[0].set_xlim((0,50))
    axes[0].set_ylim((0, None))

    no_infections = {}
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        no_infections[wp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[wp_name].keys():
                continue
            run = RUNS[wp_name][policy_name]
            population = run["D_a"].sum()
            infection_dynamics_df = run["infection_dynamics_df"]
            infection_dynamics_df = infection_dynamics_df[
                infection_dynamics_df["Age_group"] == "total"
            ]
            no_infections[wp_label][policy_label] = (
                infection_dynamics_df["total_infections_scenario"].sum()
                * 1e5
                / (population)
            )
    df = pd.DataFrame(no_infections)
    print(df)
    df.T.plot.bar(ax=axes[1], legend=False)
    #     axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     axes[1].set_title("Infections")
    axes[1].set_ylabel("Infections\n(cumulative, per 100k)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].tick_params(axis="x", labelrotation=45)

    no_severe_cases = {}
    for wp_name, wp_label in WANING_NAME_MAP.items():
        policy_name = list(POLICY_NAME_MAP.keys())[0]
        no_severe_cases[wp_label] = {}
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name not in RUNS[wp_name].keys():
                continue
            run = RUNS[wp_name][policy_name]
            population = run["D_a"].sum()
            n_weeks = len(run["weeks"])
            no_severe_cases[wp_label][policy_label] = (
                n_weeks * run["result"].sum() * 1e5
            )
    df = pd.DataFrame(no_severe_cases)
    print(df)
    df.T.plot.bar(ax=axes[2], legend=False)
#     axes[2].set_title("Severe Cases")
    axes[2].set_ylabel("Severe cases\n(cumulative, per 100k)")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[2].tick_params(axis="x", labelrotation=45)

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
        plt.savefig(OUTPUT_DIR / "waning_exp_overview2.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
# -

RUNS['no-waning']['observed']['week_dates']


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
#             id_df = id_df[id_df['Sunday_date'] < END_DATE]
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
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    ax.xaxis.set_major_locator(locator)

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
#                 id_df = id_df[id_df['Sunday_date'] < END_DATE]
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

for wp_name, wp_label in WANING_NAME_MAP.items():
    plot(RUNS[wp_name], run_name=wp_name)
# -



