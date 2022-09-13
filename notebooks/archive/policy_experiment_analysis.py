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
import datetime

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

from matplotlib.ticker import MaxNLocator

from matplotlib import cm

from pathlib import Path
from pprint import pprint

from common_functions import (
    load_run,
    compute_weekly_first_doses_per_age,
    compute_weekly_second_doses_per_age,
    compute_weekly_third_doses_per_age,
    infection_incidence,
    infection_incidence_observed,
    severe_case_incidence,
    severe_case_incidence_observed,
    severe_case_incidence_observed_trajectory,
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
# print(plt.rcParams)
plt.rcParams.update(style_modifications)

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_PLOTS = True
SPLIT_DATE_WAVES_0 = np.datetime64(datetime.date(year=2021, month=5, day=30))
# SPLIT_DATE_WAVES_1 = np.datetime64(datetime.date(year=2021, month=11, day=21))
SPLIT_DATE_WAVES_1 = np.datetime64(END_SECOND_WAVE)

STD_FACTOR = 1.96


# latest results

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C70", Path("../run/2022-06-21_15-23-27.437250_policy_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-06-21_15-23-27.773744_policy_exp")
# OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-08-01_18-32-04.576392_policy_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C90", Path("../run/2022-06-21_15-23-25.928573_policy_exp")

# latest results with error bars

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C70", Path("../run/2022-08-19_15-47-18.888241_policy_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-08-16_10-25-16.053831_policy_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C90", Path("../run/2022-08-19_15-46-57.554281_policy_exp")


# -

RUN_DIRS = {
    x.name: x
    for x in EXP_DIR.iterdir() if x.is_dir()
}
RUNS = {
    x.name: load_run(x)
    for x in EXP_DIR.iterdir() if x.is_dir()
}
df_input = pd.read_pickle(INPUT_DATA_DIR_PATH)

df_input.columns


# + code_folding=[0]
def plot_vaccination_policy(run, run_name):
    second_doses = compute_weekly_second_doses_per_age(run['U_2'])
    third_doses = compute_weekly_third_doses_per_age(run['U_2'], run['u_3'])

    colors_age = AGE_COLORMAP(np.linspace(0, 1, len(run["age_group_names"])))

    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_HEIGHT), sharey=True, sharex=False)
    plt.suptitle(f"Administered doses {run_name}")

    axes[0].stackplot(
        run['week_dates'],
        second_doses.cumsum(axis=1),
        labels=run['age_group_names'],
        alpha=0.8,
        colors=colors_age,
    )
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_minor_locator(mdates.MonthLocator())
    axes[0].set_ylabel("doses administered (cumulative)")
    axes[0].set_title("2nd doses")
    axes[0].set_facecolor(FACECOLOR)
    axes[0].set_xlim(run['week_dates'].min(), run['week_dates'].max())
    
    axes[1].stackplot(
        run['week_dates'],
        third_doses.cumsum(axis=1),
        labels=run['age_group_names'],
        alpha=0.8,
        colors=colors_age,
    )
    locator = mdates.AutoDateLocator(minticks=3, maxticks=8)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator())
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].set_ylabel("doses administered (cumulative)")
    axes[1].set_title("booster shots")
    axes[1].set_facecolor(FACECOLOR)
    axes[1].set_xlim(run['week_dates'].min(), run['week_dates'].max())


    plt.tight_layout()
    plt.show()

# for run_name, run in RUNS.items():
#     plot_vaccination_policy(run=run, run_name=run_name)

# + code_folding=[0]
def plot_severe_cases(run, run_name):
    result = run["result"]

    blues = cm.get_cmap("Blues", len(run["age_group_names"]))
    colors_age = blues(np.linspace(0, 1, num=len(run["age_group_names"])))

    greens = cm.get_cmap("Greens", len(run["vaccination_statuses"]))
    colors_vac = greens(np.linspace(0, 1, num=len(run["vaccination_statuses"])))

    fig, axes = plt.subplots(
        1, 2, figsize=(2 * PLOT_WIDTH, 1 * PLOT_HEIGHT), sharey=True, sharex=False
    )
    plt.suptitle(f"Severe cases {run_name}")

    axes[0].stackplot(
        run["week_dates"],
        result.sum(axis=2).T.cumsum(axis=1),
        labels=run["age_group_names"],
        alpha=0.8,
        colors=colors_age,
    )
    axes[0].set_ylabel("weekly severe cases")
    axes[0].set_title("by age group")
    axes[0].legend()

    axes[1].stackplot(
        run["week_dates"],
        result.sum(axis=1).T.cumsum(axis=1),
        labels=run["vaccination_statuses"],
        alpha=0.8,
        colors=colors_vac,
    )
    axes[1].set_ylabel("weekly severe cases")
    axes[1].set_title("by vaccination status")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# for run_name, run in RUNS.items():
#     plot_severe_cases(run=run, run_name=run_name)

# + code_folding=[0]
def plot(save):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=False,
    )
    def add_value_label(ax, x_list,y_list):
        for i in range(1, len(x_list)+1):
            ax.text(i,y_list[i-1],y_list[i-1], ha="center")

    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[policy_label] = (
            population * n_weeks * run["result"].sum() / (1e5 * population)
        )
    print(no_severe_cases)
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    print(df)
    axes[0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
#     add_value_label(axes[0], df.index, df[0])
    axes[0].set_title("severe cases")
    axes[0].set_ylabel("cumulative severe cases (per 100k)")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = infection_dynamics_df[
            "total_infections_scenario"
        ].sum() / (1e5 * population)
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    #     axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[1].set_title("infections")
    axes[1].set_ylabel("cumulative infections (per 100k)")

    #     h, l = axes[2].get_legend_handles_labels()
    #     fig.legend(
    #         handles=h,
    #         labels=l,
    #         ncol=5,
    #         loc="lower right",
    #         bbox_to_anchor=(0.905, 1.0),
    #         title="vaccine allocation strategy",
    #     )

    if save:
        plt.savefig(OUTPUT_DIR / "policy_exp_cumulative.pdf")
    plt.show()


# plot(save=SAVE_PLOTS)

# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=False,
    )
    plt.suptitle(f"Severe cases")
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]

        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        axes[0].plot(
            run["week_dates"],
            population * n_weeks * run["result"].sum(axis=(1, 2)),
            label=policy_label,
        )
        axes[1].plot(
            run["week_dates"],
            population * n_weeks * run["result"].sum(axis=(1, 2)).cumsum(),
            label=policy_label,
        )

    axes[0].legend()
    axes[1].legend()

    plt.show()


# plot()

# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=False,
    )
    plt.suptitle(f"Infections")
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        
        id_df = run['infection_dynamics_df']
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        axes[0].plot(
            run["week_dates"],
            id_df[id_df['Age_group'] == 'total']['total_infections_scenario'],
            label=policy_label,
        )
        axes[1].plot(
            run["week_dates"],
            id_df[id_df['Age_group'] == 'total']['total_infections_scenario'].cumsum(),
            label=policy_label,
        )

    axes[0].legend()
    axes[1].legend()

    plt.show()


# plot()

# + code_folding=[0]
def plot(save):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=False,
    )

    def add_value_label(ax, x_list, y_list):
        for i in range(1, len(x_list) + 1):
            ax.text(i, y_list[i - 1], y_list[i - 1], ha="center")

    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_0
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = infection_dynamics_df[
            "total_infections_scenario"
        ].sum() / (1e5)
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[0].set_title("infections first wave")
    axes[0].set_ylabel("cumulative infections (per 100k)")

    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            (infection_dynamics_df["Sunday_date"] >= SPLIT_DATE_WAVES_0)
            & (infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_1)
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = infection_dynamics_df[
            "total_infections_scenario"
        ].sum() / (1e5)
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[1].set_title("infections second wave")
    axes[1].set_ylabel("cumulative infections (per 100k)")

    if save:
        pass
    #         plt.savefig(OUTPUT_DIR / "policy_exp_cumulative.pdf")
    plt.show()


# plot(save=SAVE_PLOTS)

# + code_folding=[0]
def plot(save=False):
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=False,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(2, 2, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)
    
    ax0 = fig.add_subplot(gs[:2, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax11 = fig.add_subplot(gs[1, 1])
    axes = [ax0, ax01, ax11]
    
    
    import matplotlib.transforms as mtransforms
    trans = mtransforms.ScaledTranslation(10/72, -5/72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)"], axes):
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='medium', verticalalignment='top', fontfamily='serif',
                bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
    
    # severe cases both waves
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[policy_label] = (
            population * n_weeks * run["result"].sum() / (1e5 * population)
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    ax0.bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    ax0.set_title("severe cases both waves")
    ax0.set_ylabel("cumulative severe cases (per 100k)")
    print(no_severe_cases)
    
    # infections first wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_0
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = infection_dynamics_df[
            "total_infections_scenario"
        ].sum() / (1e5 * population)
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    ax01.bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    ax01.set_title("infections first wave")
    ax01.set_ylabel("cumulative infections (per 100k)")
    ax01.set_xticklabels(["","","",""]) 
    
    # infections second wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            (infection_dynamics_df["Sunday_date"] >= SPLIT_DATE_WAVES_0)
            & (infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_1)
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = infection_dynamics_df[
            "total_infections_scenario"
        ].sum() / (1e5 * population)
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    ax11.bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    ax11.set_title("infections second wave")
    ax11.set_ylabel("cumulative infections (per 100k)")
    
    if save:
        plt.savefig(OUTPUT_DIR / "policy_exp_cumulative2.pdf")
    
    plt.show()
    
# plot(save=SAVE_PLOTS)

# + code_folding=[0]
def plot(save=False):
    LW_LOCAL = 0.8
    labelrotation = 45

    fig, axes = plt.subplots(
        2,
        2,
        #         figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 0.8 * SINGLE_COLUMN),
        dpi=500,
        sharey="row",
        sharex=False,
        tight_layout=True,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.2,
    )
    TWO_LINE_NAME_MAP = {
        "elderly_first": "Elderly\nFirst",
        "observed": "Factual",
        "uniform": "Uniform",
        "young_first": "Young\nFirst",
        "risk_ranked": "Risk\nRanked",
        "risk_ranked_reversed": "Risk\nRanked\nReversed",
    }

    import matplotlib.transforms as mtransforms

#     trans = mtransforms.ScaledTranslation(10 / 72, -8 / 72, fig.dpi_scale_trans)
#     for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flatten()):
#         ax.text(
#             0.0,
#             1.0,
#             label,
#             transform=ax.transAxes + trans,
#             fontsize="medium",
#             verticalalignment="top",
#             fontfamily="serif",
#             bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0, alpha=0.75),
#         )

    # infections first wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_0
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = (
            infection_dynamics_df["total_infections_scenario"].sum()
            * 1e5
            / (population)
        )  # / 1e5
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[0, 0].set_xticklabels(["", "", "", ""])
    axes[0, 0].set_title("Third wave")
    axes[0, 0].set_ylabel("Infections\n(cumulative, per 100k)")

    # infections second wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            (infection_dynamics_df["Sunday_date"] >= SPLIT_DATE_WAVES_0)
            & (infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_1)
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = (
            infection_dynamics_df["total_infections_scenario"].sum() * 1e5 / population
        )
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 1].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    )
    axes[0, 1].set_xticklabels(["", "", "", ""])
    axes[0, 1].set_title("Fourth wave")

    # severe cases first wave
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = (
            n_weeks * run["result"][:split_date_index, ...].sum() * 1e5
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
#     axes[1, 0].set_title("Severe Cases Third Wave")
    axes[1, 0].set_ylabel("Severe cases\n(cumulative, per 100k)")
    axes[1, 0].tick_params(axis="x", labelrotation=labelrotation, labelsize="medium")
    print(no_severe_cases)

    # severe cases second wave
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        end_date_index = np.argwhere(run["week_dates"] == SPLIT_DATE_WAVES_1).flatten()[
            0
        ]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = (
            n_weeks * run["result"][split_date_index:end_date_index, ...].sum() * 1e5
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
#     axes[1, 1].set_title("Severe Cases Fourth Wave")
    axes[1, 1].tick_params(axis="x", labelrotation=labelrotation)
#     print(axes[1, 1].get_xticklabels())
    axes[1, 1].set_xticklabels( df.index, rotation=labelrotation)
    print(no_severe_cases)

    if save:
        plt.savefig(OUTPUT_DIR / f"policy_exp_cumulative3{OUTPUT_EXTENSION}.pdf")

    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[0, 4]
def plot(save=False):
    LW_LOCAL = 0.8
    labelrotation = 45

    fig, axes = plt.subplots(
        2,
        2,
        #         figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 0.8 * SINGLE_COLUMN),
        dpi=500,
        sharey="row",
        sharex=False,
        tight_layout=True,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.2,
    )
    TWO_LINE_NAME_MAP = {
        "elderly_first": "Elderly\nFirst",
        "observed": "Factual",
        "uniform": "Uniform",
        "young_first": "Young\nFirst",
        "risk_ranked": "Risk\nRanked",
        "risk_ranked_reversed": "Risk\nRanked\nReversed",
    }

    # infections first wave
    no_infections = {}
    no_infections_err_low = {}
    no_infections_err_high = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        if policy_name == "observed":
            no_infections[
                TWO_LINE_NAME_MAP[policy_name]
            ] = infection_incidence_observed(
                df_input=df_input,
                population=population,
                split_date_to=SPLIT_DATE_WAVES_0,
            )
            no_infections_err_low[TWO_LINE_NAME_MAP[policy_name]] = None
            no_infections_err_high[TWO_LINE_NAME_MAP[policy_name]] = None
        else:
            a = np.array(
                [
                    infection_incidence(
                        infection_dynamics_sample=inf_sample,
                        population=population,
                        week_dates=run["week_dates"],
                        split_date_to=SPLIT_DATE_WAVES_0,
                    )
                    for inf_sample in run["infection_dynamics_samples"]
                ]
            )

            no_infections[TWO_LINE_NAME_MAP[policy_name]] = a.mean()
            no_infections_err_low[TWO_LINE_NAME_MAP[policy_name]] = no_infections[
                TWO_LINE_NAME_MAP[policy_name]
            ] - np.percentile(a, 5)
            no_infections_err_high[TWO_LINE_NAME_MAP[policy_name]] = (
                np.percentile(a, 95) - no_infections[TWO_LINE_NAME_MAP[policy_name]]
            )

    df = pd.DataFrame.from_dict(no_infections, orient="index")
    df_err_low = pd.DataFrame.from_dict(no_infections_err_low, orient="index")
    df_err_high = pd.DataFrame.from_dict(no_infections_err_high, orient="index")
    axes[0, 0].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
        yerr=np.stack((df_err_low[0].values, df_err_high[0].values)),
    )
    axes[0, 0].set_xticklabels(["", "", "", ""])
    axes[0, 0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    axes[0, 0].set_title("Third wave")
    axes[0, 0].set_ylabel("Infections\n(cumulative, per 100k)")

    # infections second wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()

        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        if policy_name == "observed":
            no_infections[
                TWO_LINE_NAME_MAP[policy_name]
            ] = infection_incidence_observed(
                df_input=df_input,
                population=population,
                split_date_from=SPLIT_DATE_WAVES_0,
                end_date=SPLIT_DATE_WAVES_1,
            )
            no_infections_err_low[TWO_LINE_NAME_MAP[policy_name]] = None
            no_infections_err_high[TWO_LINE_NAME_MAP[policy_name]] = None
        else:
            a = np.array(
                [
                    infection_incidence(
                        infection_dynamics_sample=inf_sample,
                        population=population,
                        week_dates=run["week_dates"],
                        split_date_from=SPLIT_DATE_WAVES_0,
                        end_date=SPLIT_DATE_WAVES_1,
                    )
                    for inf_sample in run["infection_dynamics_samples"]
                ]
            )

            no_infections[TWO_LINE_NAME_MAP[policy_name]] = a.mean()
            no_infections_err_low[TWO_LINE_NAME_MAP[policy_name]] = no_infections[
                TWO_LINE_NAME_MAP[policy_name]
            ] - np.percentile(a, 5)
            no_infections_err_high[TWO_LINE_NAME_MAP[policy_name]] = (
                np.percentile(a, 95) - no_infections[TWO_LINE_NAME_MAP[policy_name]]
            )

    df = pd.DataFrame.from_dict(no_infections, orient="index")
    df_err_low = pd.DataFrame.from_dict(no_infections_err_low, orient="index")
    df_err_high = pd.DataFrame.from_dict(no_infections_err_high, orient="index")
    axes[0, 1].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
        yerr=np.stack((df_err_low[0].values, df_err_high[0].values)),
    )
    axes[0, 1].set_xticklabels(["", "", "", ""])
    axes[0, 1].set_title("Fourth wave")

    # severe cases first wave
    no_severe_cases = {}
    no_severe_cases_err_low = {}
    no_severe_cases_err_high = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()

        if policy_name == "observed":
            no_severe_cases[
                TWO_LINE_NAME_MAP[policy_name]
            ] = severe_case_incidence_observed(
                df_input=df_input,
                population=population,
                split_date_to=SPLIT_DATE_WAVES_0,
            )
            no_severe_cases_err_low[TWO_LINE_NAME_MAP[policy_name]] = None
            no_severe_cases_err_high[TWO_LINE_NAME_MAP[policy_name]] = None
        else:
            a = np.array(
                [
                    severe_case_incidence(
                        res,
                        len(run["weeks"]),
                        run["week_dates"],
                        split_date_to=SPLIT_DATE_WAVES_0,
                    )
                    for res in run["result_samples"]
                ]
            )
            no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = a.mean()
            no_severe_cases_err_low[TWO_LINE_NAME_MAP[policy_name]] = no_severe_cases[
                TWO_LINE_NAME_MAP[policy_name]
            ] - np.percentile(a, ERROR_PERCENTILE_LOW)
            no_severe_cases_err_high[TWO_LINE_NAME_MAP[policy_name]] = (
                np.percentile(a, ERROR_PERCENTILE_HIGH)
                - no_severe_cases[TWO_LINE_NAME_MAP[policy_name]]
            )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    df_err_low = pd.DataFrame.from_dict(no_severe_cases_err_low, orient="index")
    df_err_high = pd.DataFrame.from_dict(no_severe_cases_err_high, orient="index")
    axes[1, 0].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
        yerr=np.stack((df_err_low[0].values, df_err_high[0].values)),
    )
    axes[1, 0].set_ylabel("Severe cases\n(cumulative, per 100k)")
    axes[1, 0].tick_params(axis="x", labelrotation=labelrotation, labelsize="medium")
    axes[1, 0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    for policy in no_severe_cases.keys():
        if policy == "Factual":
            print(f"{policy:} {no_severe_cases[policy]:.2f}")
        else:
            print(
                f"{policy:} {no_severe_cases[policy]:.2f} + {no_severe_cases_err_high[policy]:.2f} - {no_severe_cases_err_low[policy]:.2f}"
            )

    # severe cases second wave
    no_severe_cases = {}
    no_severe_cases_err_low = {}
    no_severe_cases_err_high = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        if policy_name == "observed":
            no_severe_cases[
                TWO_LINE_NAME_MAP[policy_name]
            ] = severe_case_incidence_observed(
                df_input=df_input,
                population=population,
                split_date_from=SPLIT_DATE_WAVES_0,
                end_date=SPLIT_DATE_WAVES_1,
            )
            no_severe_cases_err_low[TWO_LINE_NAME_MAP[policy_name]] = None
            no_severe_cases_err_high[TWO_LINE_NAME_MAP[policy_name]] = None
        else:
            a = np.array(
                [
                    severe_case_incidence(
                        res,
                        len(run["weeks"]),
                        run["week_dates"],
                        split_date_from=SPLIT_DATE_WAVES_0,
                        end_date=SPLIT_DATE_WAVES_1,
                    )
                    for res in run["result_samples"]
                ]
            )
            no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = a.mean()
            no_severe_cases_err_low[TWO_LINE_NAME_MAP[policy_name]] = no_severe_cases[
                TWO_LINE_NAME_MAP[policy_name]
            ] - np.percentile(a, ERROR_PERCENTILE_LOW)
            no_severe_cases_err_high[TWO_LINE_NAME_MAP[policy_name]] = (
                np.percentile(a, ERROR_PERCENTILE_HIGH)
                - no_severe_cases[TWO_LINE_NAME_MAP[policy_name]]
            )

    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    df_err_low = pd.DataFrame.from_dict(no_severe_cases_err_low, orient="index")
    df_err_high = pd.DataFrame.from_dict(no_severe_cases_err_high, orient="index")
    axes[1, 1].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
        yerr=np.stack((df_err_low[0].values, df_err_high[0].values)),
    )
    axes[1, 1].tick_params(axis="x", labelrotation=labelrotation)
    axes[1, 1].set_xticklabels(df.index, rotation=labelrotation)
    for policy in no_severe_cases.keys():
        if policy == "Factual":
            print(f"{policy:} {no_severe_cases[policy]:.2f}")
        else:
            print(
                f"{policy:} {no_severe_cases[policy]:.2f} + {no_severe_cases_err_high[policy]:.2f} - {no_severe_cases_err_low[policy]:.2f}"
            )

    if save:
        plt.savefig(OUTPUT_DIR / f"policy_exp_cumulative4{OUTPUT_EXTENSION}.pdf")

    plt.show()


plot(save=SAVE_PLOTS)
# -

RUNS["observed"].keys()

RUNS["observed"]["observed_severity_df"]

# +
# df_id = RUNS["uniform"]["infection_dynamics_df"]
# df_tmp = df_id[df_id["Age_group"] != "total"]
# df_tmp["tmp"] = (
#     df_tmp["total_infections_var_scenario"]
# )

# df_tmp_total = df_tmp.groupby("Sunday_date")["tmp"].sum().reset_index()
# df_tmp_total["Age_group"] = "total"


# # df_tmp_total

# df_tmp = df_tmp_total.append(
#     df_id[df_id["Age_group"] != "total"][
#         ["Sunday_date", "Age_group", "total_infections_var_scenario"]
#     ].rename(columns={"total_infections_var_scenario": "tmp"})
# )
# df_tmp = df_tmp.sort_values(["Sunday_date", "Age_group"]).rename(columns={"tmp": "total_infections_var_scenario"})

# df_id.drop("total_infections_var_scenario", axis=1).merge(df_tmp, on=["Sunday_date", "Age_group"]).head(30)

# +
# df_id = RUNS["uniform"]["infection_dynamics_df"]
# df_tmp = df_id[df_id["Age_group"] != "total"]
# df_tmp = (
#     df_tmp.groupby("Sunday_date")["total_infections_var_scenario"].sum().reset_index()
# )
# df_tmp["Age_group"] = "total"
# df_tmp = df_id[df_id["Age_group"] != "total"][
#     ["Sunday_date", "Age_group", "total_infections_var_scenario"]
# ].append(df_tmp)

# df_id.drop("total_infections_var_scenario", axis=1).merge(
#     df_tmp, 
#     on=["Sunday_date", "Age_group"]
# ).head(30)

# + code_folding=[0, 31, 82, 95, 142]
# def plot(save=False):
#     LW_LOCAL = 0.8
#     labelrotation = 45

#     fig, axes = plt.subplots(
#         2,
#         2,
#         #         figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH),
#         figsize=(SINGLE_COLUMN, 0.8 * SINGLE_COLUMN),
#         dpi=500,
#         sharey="row",
#         sharex=False,
#         tight_layout=True,
#     )
#     #     plt.suptitle("Severe cases (per 100k)")
#     plt.subplots_adjust(
#         hspace=0.35,
#         wspace=0.2,
#     )
#     TWO_LINE_NAME_MAP = {
#         "elderly_first": "Elderly\nFirst",
#         "observed": "Factual",
#         "uniform": "Uniform",
#         "young_first": "Young\nFirst",
#         "risk_ranked": "Risk\nRanked",
#         "risk_ranked_reversed": "Risk\nRanked\nReversed",
#     }

#     # infections first wave
#     no_infections = {}
#     no_infections_std = {}
#     for policy_name, policy_label in POLICY_NAME_MAP.items():
#         if policy_name not in RUNS.keys():
#             continue
#         run = RUNS[policy_name]
#         population = run["D_a"].sum()
#         infection_dynamics_df = run["infection_dynamics_df"]

#         # generate infection variance for total population
#         df_tmp = infection_dynamics_df[infection_dynamics_df["Age_group"] != "total"]
#         df_tmp = (
#             df_tmp.groupby("Sunday_date")["total_infections_var_scenario"]
#             .sum()
#             .reset_index()
#         )
#         df_tmp["Age_group"] = "total"
#         df_tmp = infection_dynamics_df[infection_dynamics_df["Age_group"] != "total"][
#             ["Sunday_date", "Age_group", "total_infections_var_scenario"]
#         ].append(df_tmp)

#         infection_dynamics_df = infection_dynamics_df.drop(
#             "total_infections_var_scenario", axis=1
#         ).merge(df_tmp, on=["Sunday_date", "Age_group"])

#         infection_dynamics_df["Sunday_date"] = pd.to_datetime(
#             infection_dynamics_df["Sunday_date"]
#         )
#         infection_dynamics_df = infection_dynamics_df[
#             infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_0
#         ]
#         infection_dynamics_df = infection_dynamics_df[
#             infection_dynamics_df["Age_group"] == "total"
#         ]
        
#         no_infections[policy_label] = (
#             infection_dynamics_df["total_infections_scenario"].sum()
#             * 1e5
#             / (population)
#         )  # / 1e5
#         no_infections_std[policy_label] = STD_FACTOR * np.sqrt(
#             (
#                 infection_dynamics_df["total_infections_var_scenario"]
#                 * 1e5 ** 2
#                 / (population ** 2)
#             ).sum()
#         )

#     print("no_infections:", no_infections)
#     print("no_infections_std:", no_infections_std)

#     df = pd.DataFrame.from_dict(no_infections, orient="index")
#     df_std = pd.DataFrame.from_dict(no_infections_std, orient="index")
#     axes[0, 0].bar(
#         df.index,
#         df[0],
#         color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
#         yerr=df_std[0],
#     )
#     axes[0, 0].set_xticklabels(["", "", "", ""])
#     axes[0, 0].set_title("Third wave")
#     axes[0, 0].set_ylabel("Infections\n(cumulative, per 100k)")

#     # infections second wave
#     no_infections = {}
#     no_infections_std = {}
#     for policy_name, policy_label in POLICY_NAME_MAP.items():
#         if policy_name not in RUNS.keys():
#             continue
#         run = RUNS[policy_name]
#         population = run["D_a"].sum()
#         infection_dynamics_df = run["infection_dynamics_df"]

#         # generate infection variance for total population
#         df_tmp = infection_dynamics_df[infection_dynamics_df["Age_group"] != "total"]
#         df_tmp = (
#             df_tmp.groupby("Sunday_date")["total_infections_var_scenario"]
#             .sum()
#             .reset_index()
#         )
#         df_tmp["Age_group"] = "total"
#         df_tmp = infection_dynamics_df[infection_dynamics_df["Age_group"] != "total"][
#             ["Sunday_date", "Age_group", "total_infections_var_scenario"]
#         ].append(df_tmp)

#         infection_dynamics_df = infection_dynamics_df.drop(
#             "total_infections_var_scenario", axis=1
#         ).merge(df_tmp, on=["Sunday_date", "Age_group"])

#         infection_dynamics_df["Sunday_date"] = pd.to_datetime(
#             infection_dynamics_df["Sunday_date"]
#         )
#         infection_dynamics_df = infection_dynamics_df[
#             (infection_dynamics_df["Sunday_date"] >= SPLIT_DATE_WAVES_0)
#             & (infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_1)
#         ]
#         infection_dynamics_df = infection_dynamics_df[
#             infection_dynamics_df["Age_group"] == "total"
#         ]
        
#         no_infections[policy_label] = (
#             infection_dynamics_df["total_infections_scenario"].sum() * 1e5 / population
#         )
#         no_infections_std[policy_label] = STD_FACTOR * np.sqrt(
#             (
#                 infection_dynamics_df["total_infections_var_scenario"]
#                 * 1e5 ** 2
#                 / (population ** 2)
#             ).sum()
#         )
        
#     df = pd.DataFrame.from_dict(no_infections, orient="index")
#     df_std = pd.DataFrame.from_dict(no_infections_std, orient="index")
#     axes[0, 1].bar(
#         df.index,
#         df[0],
#         color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
#         yerr=df_std[0],
#     )
#     axes[0, 1].set_xticklabels(["", "", "", ""])
#     axes[0, 1].set_title("Fourth wave")

#     # severe cases first wave
#     no_severe_cases = {}
#     no_severe_cases_std = {}
#     for policy_name, policy_label in POLICY_NAME_MAP.items():
#         if policy_name not in RUNS.keys():
#             continue
#         run = RUNS[policy_name]
#         split_date_index = np.argwhere(
#             run["week_dates"] == SPLIT_DATE_WAVES_0
#         ).flatten()[0]
#         population = run["D_a"].sum()
#         n_weeks = len(run["weeks"])
#         error_propagation_coeffs = run["error_propagation_coeffs"][:split_date_index, ...]
#         f_1_var = run["f_1_var"][..., :split_date_index]
        
#         no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = (
#             n_weeks * run["result"][:split_date_index, ...].sum() * 1e5
#         )
#         no_severe_cases_std[TWO_LINE_NAME_MAP[policy_name]] = STD_FACTOR * np.sqrt(
#             (
#                 error_propagation_coeffs.T
#                 * f_1_var
#                 * 1e5 ** 2
#                 * n_weeks ** 2
#             ).sum()
#         )
    
#     df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
#     df_std = pd.DataFrame.from_dict(no_severe_cases_std, orient="index")
#     axes[1, 0].bar(
#         df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"], yerr=df_std[0]
#     )
#     #     axes[1, 0].set_title("Severe Cases Third Wave")
#     axes[1, 0].set_ylabel("Severe cases\n(cumulative, per 100k)")
#     axes[1, 0].tick_params(axis="x", labelrotation=labelrotation, labelsize="medium")
#     print("no_severe_cases:", no_severe_cases)
#     print("no_severe_cases_std:", no_severe_cases_std)

#     # severe cases second wave
#     no_severe_cases = {}
#     for policy_name, policy_label in POLICY_NAME_MAP.items():
#         if policy_name not in RUNS.keys():
#             continue
#         run = RUNS[policy_name]
#         split_date_index = np.argwhere(
#             run["week_dates"] == SPLIT_DATE_WAVES_0
#         ).flatten()[0]
#         end_date_index = np.argwhere(run["week_dates"] == SPLIT_DATE_WAVES_1).flatten()[
#             0
#         ]
#         population = run["D_a"].sum()
#         n_weeks = len(run["weeks"])
#         no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = (
#             n_weeks * run["result"][split_date_index:end_date_index, ...].sum() * 1e5
#         )
#     df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
#     axes[1, 1].bar(
#         df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     )
#     #     axes[1, 1].set_title("Severe Cases Fourth Wave")
#     axes[1, 1].tick_params(axis="x", labelrotation=labelrotation)
#     #     print(axes[1, 1].get_xticklabels())
#     axes[1, 1].set_xticklabels(df.index, rotation=labelrotation)
#     print(no_severe_cases)

#     if save:
#         plt.savefig(OUTPUT_DIR / f"policy_exp_cumulative4{OUTPUT_EXTENSION}.pdf")

#     plt.show()


# plot(save=SAVE_PLOTS)

# + code_folding=[0]
def plot(save=False):
    LW_LOCAL = 0.8
    labelrotation = 30

    fig, axes = plt.subplots(
        3,
        2,
        #         figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 1.2 * SINGLE_COLUMN),
        dpi=500,
        sharey="row",
        sharex=False,
        tight_layout=True,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    plt.subplots_adjust(
        hspace=0.35,
        wspace=0.2,
    )

    import matplotlib.transforms as mtransforms

#     trans = mtransforms.ScaledTranslation(10 / 72, -8 / 72, fig.dpi_scale_trans)
#     for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flatten()):
#         ax.text(
#             0.0,
#             1.0,
#             label,
#             transform=ax.transAxes + trans,
#             fontsize="medium",
#             verticalalignment="top",
#             fontfamily="serif",
#             bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0, alpha=0.75),
#         )

    # infections first wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_0
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = (
            infection_dynamics_df["total_infections_scenario"].sum()
            * 1e5
            / (population)
        )  # / 1e5
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[0, 0].set_xticklabels(["", "", "", ""])
    axes[0, 0].set_title("Infections Third Wave")
    axes[0, 0].set_ylabel("Cumulative Infections\n(per 100k)")

    # infections second wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            (infection_dynamics_df["Sunday_date"] >= SPLIT_DATE_WAVES_0)
            & (infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_1)
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = (
            infection_dynamics_df["total_infections_scenario"].sum() * 1e5 / population
        )
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 1].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    )
    axes[0, 1].set_xticklabels(["", "", "", ""])
    axes[0, 1].set_title("Infections Fourth Wave")

    # severe cases first wave
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[policy_label] = (
            n_weeks * run["result"][:split_date_index, ...].sum() * 1e5
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[1, 0].set_title("Severe Cases Third Wave")
    axes[1, 0].set_ylabel("Cumulative Severe Cases\n(per 100k)")
    axes[1, 0].set_xticklabels(["", "", "", ""])
    axes[1, 0].tick_params(axis="x", labelrotation=labelrotation)
    print(no_severe_cases)

    # severe cases second wave
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        end_date_index = np.argwhere(run["week_dates"] == SPLIT_DATE_WAVES_1).flatten()[
            0
        ]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[policy_label] = (
            n_weeks * run["result"][split_date_index:end_date_index, ...].sum() * 1e5
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[1, 1].set_title("Severe Cases Fourth Wave")
    axes[1, 1].set_xticklabels(["", "", "", ""])
    axes[1, 1].tick_params(axis="x", labelrotation=labelrotation)
    print(no_severe_cases)

    # yll first wave
    yll = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        yll_per_age = (84 - np.array([10, 25, 35, 45, 55, 65, 75, 85, 90])).clip(min=0)
        print(yll_per_age)
        yll[policy_label] = (
            n_weeks * run["result"][:split_date_index, ...].sum(axis=(0, 2)) * population * yll_per_age
        ).sum()
    df = pd.DataFrame.from_dict(yll, orient="index")
    axes[2, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[2, 0].set_title("YLL Third Wave")
    axes[2, 0].set_ylabel("Cumulative YLL")
    axes[2, 0].tick_params(axis="x", labelrotation=labelrotation)
    print(yll)

    # yll second wave
    yll = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        end_date_index = np.argwhere(run["week_dates"] == SPLIT_DATE_WAVES_1).flatten()[
            0
        ]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        yll_per_age = (84 - np.array([10, 25, 35, 45, 55, 65, 75, 85, 90])).clip(min=0)
        print(yll_per_age)
        yll[policy_label] = (
            n_weeks * run["result"][split_date_index:end_date_index, ...].sum(axis=(0, 2)) * population * yll_per_age
        ).sum()
    df = pd.DataFrame.from_dict(yll, orient="index")
    axes[2, 1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[2, 1].set_title("YLL Fourth Wave")
    axes[2, 1].tick_params(axis="x", labelrotation=labelrotation)
    print(yll)

    if save:
        plt.savefig(OUTPUT_DIR / "policy_exp_cumulative_yll.pdf")

    plt.show()


# plot(save=SAVE_PLOTS)

# + code_folding=[0]
def plot():
    run_list = ["observed", "uniform", "elderly_first", "young_first"]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        sharey="row",
        sharex=True,
    )
    plt.subplots_adjust(hspace=0, wspace=0)

    for i, run_name in enumerate(run_list):
        run = RUNS[run_name]
        second_doses = compute_weekly_second_doses_per_age(run["U_2"])
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])

        blues = cm.get_cmap("Blues", len(run["age_group_names"]))
        colors_age = blues(np.linspace(0, 1, num=len(run["age_group_names"])))

        axes[0, i].set_title(run_name)
        axes[0, i].stackplot(
            run["week_dates"],
            second_doses.cumsum(axis=1),
            labels=run["age_group_names"],
            alpha=0.8,
            colors=colors_age,
        )

        axes[1, i].stackplot(
            run["week_dates"],
            run["result"].sum(axis=2).T.cumsum(axis=1) * n_weeks * population,
            labels=run["age_group_names"],
            alpha=0.8,
            colors=colors_age,
        )
        axes[1, i].tick_params(axis="x", labelrotation=45)
    #     axes[1, i].grid()

    axes[0, 0].set_ylabel("second doses administered (cumulative)")
    axes[1, 0].set_ylabel("severe cases (cumulative)")
    #     plt.savefig(OUTPUT_DIR / "policy_exp_policies", dpi=500)
    plt.show()


# plot()

# + code_folding=[0]
def plot(save=False):
    run_list = ["observed", "uniform", "elderly_first", "young_first"]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        dpi=500,
        sharey="row",
        sharex=True,
    )
    plt.subplots_adjust(hspace=0.15, wspace=0.1)

    for i, run_name in enumerate(run_list):
        run = RUNS[run_name]
        second_doses = compute_weekly_second_doses_per_age(run["U_2"])
        third_doses = compute_weekly_third_doses_per_age(run["U_2"], run["u_3"])
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])

        blues = cm.get_cmap("Blues", len(run["age_group_names"]))
        colors_age = AGE_COLORMAP(np.linspace(0, 1, len(run["age_group_names"])))

        axes[0, i].set_title(run_name)
        axes[0, i].stackplot(
            run["week_dates"],
            second_doses.cumsum(axis=1),
            labels=run["age_group_names"],
            alpha=0.8,
            colors=colors_age,
        )

        axes[1, i].stackplot(
            run["week_dates"],
            third_doses.cumsum(axis=1),
            labels=run["age_group_names"],
            alpha=0.8,
            colors=colors_age,
        )
        axes[1, i].tick_params(axis="x", labelrotation=45)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        axes[1, i].xaxis.set_major_locator(locator)
        if i > 0:
            axes[0, i].tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False,
            )
            axes[1, i].tick_params(
                axis="y",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                right=False,
                left=False,
                labelleft=False,
            )
    axes[0, 0].tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
        left=False,
        labelleft=False,
    )

    axes[0, 0].set_ylabel("2nd doses (cumulative)")
    axes[1, 0].set_ylabel("3rd doses (cumulative)")
    #     axes[1, 0].get_yaxis().get_offset_text().set_position((-0.25,-0.25))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles=h, 
        labels=l, 
        ncol=9, bbox_to_anchor=(0.405, 0.55, 0.5, 0.5))
    if save:
        plt.savefig(OUTPUT_DIR / "vaccine_allocation_strategies", dpi=500)
    plt.show()


# plot(save=SAVE_PLOTS)

# + code_folding=[0]
def plot(save=False):
    run_list = ["observed", "uniform", "elderly_first", "young_first"]
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=False,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(3, 6, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)

    ax = fig.add_subplot(gs[:3, :3])
    for run_name, run in RUNS.items():
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        ax.plot(
            run["week_dates"],
            population * n_weeks * run["result"].sum(axis=(1, 2)),
            label=run_name,
            lw=LW_LOCAL,
        )
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("weekly severe cases")
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
        for run_name, run in RUNS.items():
            population = run["D_a"].sum()
            population_age_group = run["D_a"][i]
            n_weeks = len(run["weeks"])
            ax.plot(
                run["week_dates"],
                population
                * n_weeks
                * run["result"][:, i, :].sum(axis=1),
                label=run_name,
                lw=LW_LOCAL,
            )
        ax.set_title(run["age_group_names"][i], loc="right", pad=-FONTSIZE, y=1.0)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)

        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(OUTPUT_DIR / "policy_exp_severe_cases", dpi=500)
    plt.show()

# plot()


# + code_folding=[0]
def plot(save=False):
    run_list = ["observed", "uniform", "elderly_first", "young_first"]
    LW_LOCAL = 1.5

    fig = plt.figure(
#         tight_layout=True,
        #         figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        figsize=(DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN),
        dpi=500,
        constrained_layout=True,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(3, 36, figure=fig)
    ax = fig.add_subplot(gs[1:3, :12])
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name in RUNS.keys():
            run = RUNS[policy_name]
            population = run["D_a"].sum()
            n_weeks = len(run["weeks"])
            severe_case_incidence = (
                n_weeks
                #                 * population
                * run["result"].sum(axis=(1, 2))
                * (1e5)
            )  # TODO: check this formula
            print(f"{policy_name}: {severe_case_incidence.sum()}")
            ax.plot(
                run["week_dates"],
                severe_case_incidence,
                label=policy_label,
                lw=LW_LOCAL,
            )

    ax.tick_params(axis="x", labelrotation=0)
    ax.set_ylabel("Severe cases (weekly, per 100k)")
    ax.set_title("All age groups ", loc="right", pad=-FONTSIZE, y=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=run["week_dates"].min(), right=run["week_dates"].max())
    
    myFmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(myFmt)
    
    locator = mdates.AutoDateLocator(minticks=7, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    
    ax.set_xlabel(2021)
    
    h, l = ax.get_legend_handles_labels()
    
    ax = fig.add_subplot(gs[0, :12])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    ax.legend(
        handles=h,
        labels=l,
        ncol=2,
        loc="center",
#         bbox_to_anchor=(0.905, 1.0),
        title="Vaccine allocation strategy",
        fontsize='medium',
    )

    index_matrix = np.arange(9).reshape((3, 3))
    for i in range(9):
        indices = np.argwhere(index_matrix == i)[0]
        sub_grid = gs[indices[0], 12 + 8 * indices[1] : 12 + 8 * (indices[1] + 1)]
        if i == 0:
            ax = fig.add_subplot(sub_grid)
            initial_ax = ax
        else:
            ax = fig.add_subplot(
                sub_grid,
                sharex=initial_ax,
            )
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name in RUNS.keys():
                run = RUNS[policy_name]
                population = run["D_a"].sum()
                population_age_group = run["D_a"][i]
                P_a = run["P_a"][i]
                n_weeks = len(run["weeks"])
                severe_case_incidence = (
                    n_weeks
                    #                     * population
                    * run["result"][:, i, :].sum(axis=1)
                    * 1e5
                    / P_a
                    #                     / (1e5 * P_a)
                )
                print(f"{policy_name}, {i}: {severe_case_incidence.sum()}")

                ax.plot(
                    run["week_dates"],
                    severe_case_incidence,
                    label=policy_label,
                    lw=LW_LOCAL,
                )
        ax.set_title(run["age_group_names"][i] + " ", loc="right", pad=-FONTSIZE, y=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=run["week_dates"].min(), right=run["week_dates"].max())

        locator = mdates.AutoDateLocator(minticks=7, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        
        
        myFmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(myFmt)
        
#         ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.locator_params(axis="y", integer=True, tight=True)

        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", labelrotation=0)
            ax.set_xlabel(2021)

    if save:
        plt.savefig(
            OUTPUT_DIR / f"policy_exp_severe_cases_per_100k{OUTPUT_EXTENSION}.pdf"
        )
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[]
def plot(save=False):
    run_list = ["observed", "uniform", "elderly_first", "young_first"]
    LW_LOCAL = 1.5

    fig = plt.figure(
#         tight_layout=True,
        #         figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        figsize=(DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN),
        dpi=500,
        constrained_layout=True,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(3, 36, figure=fig)
    ax = fig.add_subplot(gs[1:3, :12])
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name in RUNS.keys():
            run = RUNS[policy_name]
            population = run["D_a"].sum()
            if policy_name == "observed":
                severe_case_incidence = severe_case_incidence_observed_trajectory(
                    df_input=df_input,
                    population=population,
                )
            else:
                n_weeks = len(run["weeks"])
                severe_case_incidence = (
                    n_weeks
                    * run["result"].sum(axis=(1, 2))
                    * (1e5)
                )  # TODO: check this formula
            
            print(f"{policy_name}: {severe_case_incidence.sum()}")
            ax.plot(
                run["week_dates"],
                severe_case_incidence,
                label=policy_label,
                lw=LW_LOCAL,
            )

    ax.tick_params(axis="x", labelrotation=0)
    ax.set_ylabel("Severe cases (weekly, per 100k)")
    ax.set_title("All age groups ", loc="right", pad=-FONTSIZE, y=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=run["week_dates"].min(), right=run["week_dates"].max())
    
    myFmt = mdates.DateFormatter('%b')
    ax.xaxis.set_major_formatter(myFmt)
    
    locator = mdates.AutoDateLocator(minticks=7, maxticks=7)
    ax.xaxis.set_major_locator(locator)
    
    ax.set_xlabel(2021)
    
    h, l = ax.get_legend_handles_labels()
    
    ax = fig.add_subplot(gs[0, :12])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    
    ax.legend(
        handles=h,
        labels=l,
        ncol=2,
        loc="center",
#         bbox_to_anchor=(0.905, 1.0),
        title="Vaccine allocation strategy",
        fontsize='medium',
    )

    index_matrix = np.arange(9).reshape((3, 3))
    for i in range(9):
        indices = np.argwhere(index_matrix == i)[0]
        sub_grid = gs[indices[0], 12 + 8 * indices[1] : 12 + 8 * (indices[1] + 1)]
        if i == 0:
            ax = fig.add_subplot(sub_grid)
            initial_ax = ax
        else:
            ax = fig.add_subplot(
                sub_grid,
                sharex=initial_ax,
            )
        for policy_name, policy_label in POLICY_NAME_MAP.items():
            if policy_name in RUNS.keys():
                run = RUNS[policy_name]
                population = run["D_a"].sum()
                population_age_group = run["D_a"][i]
                age_group_name = run["age_group_names"][i]
                if policy_name == "observed":
                    severe_case_incidence = severe_case_incidence_observed_trajectory(
                        df_input=df_input,
                        population=population_age_group,
                        age_group=age_group_name,
                    )
                else:
                    population = run["D_a"].sum()
                    population_age_group = run["D_a"][i]
                    P_a = run["P_a"][i]
                    n_weeks = len(run["weeks"])
                    severe_case_incidence = (
                        n_weeks
                        * run["result"][:, i, :].sum(axis=1)
                        * 1e5
                        / P_a
                    )
                    print(f"{policy_name}, {i}: {severe_case_incidence.sum()}")

                ax.plot(
                    run["week_dates"],
                    severe_case_incidence,
                    label=policy_label,
                    lw=LW_LOCAL,
                )
        ax.set_title(run["age_group_names"][i] + " ", loc="right", pad=-FONTSIZE, y=1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=run["week_dates"].min(), right=run["week_dates"].max())

        locator = mdates.AutoDateLocator(minticks=7, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        
        
        myFmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_formatter(myFmt)
        
#         ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.locator_params(axis="y", integer=True, tight=True)

        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.tick_params(axis="x", labelrotation=0)
            ax.set_xlabel(2021)

    if save:
        plt.savefig(
            OUTPUT_DIR / f"policy_exp_severe_cases_per_100k2{OUTPUT_EXTENSION}.pdf"
        )
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[0]
def plot(save=False):
    run_list = ["observed", "uniform", "elderly_first", "young_first"]
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=False,
        figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(3, 6, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)

    ax = fig.add_subplot(gs[:3, :3])
    for run_name, run in RUNS.items():
        id_df = run["infection_dynamics_df"]
        ax.plot(
            run["week_dates"],
            id_df[id_df["Age_group"] == "total"]["total_infections_scenario"],
            label=run_name,
            lw=LW_LOCAL,
        )
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("weekly infections")
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
        for run_name, run in RUNS.items():
            id_df = run["infection_dynamics_df"]
            ax.plot(
                run["week_dates"],
                id_df[id_df["Age_group"] == "total"]["total_infections_scenario"],
                label=run_name,
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
        plt.savefig(OUTPUT_DIR / "policy_exp_infections", dpi=500)
    plt.show()


# plot()

# + code_folding=[0]
def plot(save=False, path=None):
    LW_LOCAL = 0.8

    fig = plt.figure(
        tight_layout=True,
#         figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        figsize=(DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN),
        dpi=500,
        constrained_layout=False,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(3, 6, figure=fig)
    gs.update(wspace=0.35, hspace=0.3)

    ax = fig.add_subplot(gs[:3, :3])
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name in RUNS.keys():
            run = RUNS[policy_name]
            id_df = run["infection_dynamics_df"]
            population = run["D_a"].sum()
            infection_incidence = (
                id_df[id_df["Age_group"] == "total"]["total_infections_scenario"]
                * 1e5
                / population
            )
            ax.plot(
                run["week_dates"],
                infection_incidence,
                label=policy_label,
                lw=LW_LOCAL,
            )
            print(f"{policy_name}: {infection_incidence.sum()}")
    ax.legend(loc="upper left")
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("weekly infections (per 100k)")
    ax.set_title("total ", loc="right", pad=-FONTSIZE, y=1.0)

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
            if policy_name in RUNS.keys():
                run = RUNS[policy_name]
                id_df = run["infection_dynamics_df"]
                population_age_group = run["D_a"][i]
                infection_incidence = (
                    id_df[id_df["Age_group"] == run["age_group_names"][i]]["total_infections_scenario"]
                    * 1e5
                    / population_age_group
                )
                ax.plot(
                    run["week_dates"],
                    infection_incidence,
                    label=policy_label,
                    lw=LW_LOCAL,
                )
                print(f"{policy_name}, {i}: {infection_incidence.sum()}")
        ax.set_title(run["age_group_names"][i] + " ", loc="right", pad=-FONTSIZE, y=1.0)
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


plot(save=SAVE_PLOTS, path=OUTPUT_DIR / "policy_exp_infections_per_100k.pdf")


# + code_folding=[0]
def print_R(run, run_name, save=False):
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * PLOT_WIDTH, 1 * PLOT_HEIGHT), sharey=True, sharex=False
    )
    plt.suptitle(f"Reproduction numbers {run_name}")
    plt.tight_layout()

    colors = AGE_COLORMAP(np.linspace(0, 1, len(run["age_group_names"])))

    axes[0].set_title("base R")
    for ag in run["age_groups"]:
        axes[0].plot(
            run["week_dates"],
            run["median_weekly_base_R_t"][:, ag],
            label=run["age_group_names"][ag],
            c=colors[ag],
        )
    axes[0].set_facecolor(FACECOLOR)
    axes[0].axvspan(
        START_FIRST_WAVE,
        END_FIRST_WAVE,
        alpha=1.0,
        color="skyblue",
    )
    axes[0].axvspan(
        START_SECOND_WAVE,
        END_SECOND_WAVE,
        alpha=1.0,
        color="skyblue",
    )
    axes[0].set_ylabel("base R_T")
    axes[0].grid()

    axes[1].set_title("effective R")
    for ag in run["age_groups"]:
        axes[1].plot(
            run["week_dates"],
            run["median_weekly_eff_R_t"][:, ag],
            label=run["age_group_names"][ag],
            c=colors[ag],
        )
    axes[1].set_facecolor(FACECOLOR)
    axes[1].axvspan(
        START_FIRST_WAVE,
        END_FIRST_WAVE,
        alpha=1.0,
        color="skyblue",
    )
    axes[1].axvspan(
        START_SECOND_WAVE,
        END_SECOND_WAVE,
        alpha=1.0,
        color="skyblue",
    )
    axes[1].set_ylabel("eff R_T")
    axes[1].grid()

    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="age groups",
        facecolor=FACECOLOR,
    )
    plt.tight_layout()
    if save:
        plt.savefig(OUTPUT_DIR / f"R_{run_name}", dpi=500)

    plt.show()


# for run_name, run in RUNS.items():
#     print_R(run=run, run_name=run_name, save=SAVE_PLOTS)

# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * PLOT_WIDTH, 1 * PLOT_HEIGHT), sharey=True, sharex=False
    )
    plt.suptitle("Weighted average R_base")

    # plt.tight_layout()
    run = RUNS[list(RUNS.keys())[0]]

    infection_dynamics_df = run["infection_dynamics_df"]
    infection_dynamics_df = infection_dynamics_df[
        infection_dynamics_df["Age_group"] != "total"
    ]
    infection_array = infection_dynamics_df.sort_values(["Sunday_date", "Age_group"])[
        ["total_infections_observed"]
    ].values.reshape((-1, 9))

    weight_rising = infection_array[:-1, :] * (np.diff(infection_array, axis=0) > 0)
    weighted_rising_avg_base_R_t = (
        run["median_weekly_base_R_t"][:-1, :] * weight_rising
    ).sum(axis=0) / weight_rising.sum(axis=0)

    weight_inf = infection_array[:-1, :]
    weighted_inf_avg_base_R_t = (run["median_weekly_base_R_t"][:-1, :] * weight_inf).sum(
        axis=0
    ) / weight_inf.sum(axis=0)

    weight_uniform = np.ones_like(weight_inf)
    avg_base_R_t = (run["median_weekly_base_R_t"][:-1, :] * weight_uniform).sum(
        axis=0
    ) / weight_uniform.sum(axis=0)

    axes[0].set_title("unweighted")
    pd.DataFrame(
        {
            "age_groups": run["age_group_names"],
            "avg_R_base": avg_base_R_t,
        }
    ).set_index("age_groups").plot.bar(ax=axes[0])

    axes[1].set_title("weighted by infections")
    pd.DataFrame(
        {
            "age_groups": run["age_group_names"],
            "avg_R_base": weighted_inf_avg_base_R_t,
        }
    ).set_index("age_groups").plot.bar(ax=axes[1])

    axes[2].set_title("weighted by infections (rising only)")
    pd.DataFrame(
        {
            "age_groups": run["age_group_names"],
            "avg_R_base": weighted_rising_avg_base_R_t,
        }
    ).set_index("age_groups").plot.bar(ax=axes[2])


    plt.show()

# plot()

# + code_folding=[0]
def print_R(run, run_name, save=False):
    fig, axes = plt.subplots(
        1,
        2,
#         figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        figsize=(DOUBLE_COLUMN, 0.3 * DOUBLE_COLUMN),
        dpi=500,
        sharey=False,
        sharex=False,
        tight_layout=True,
    )
    plt.subplots_adjust(
        #         hspace=0.15,
        wspace=0.3,
    )

    colors = AGE_COLORMAP(np.linspace(0, 1, len(run["age_group_names"])))

    #     axes[0].set_title("base R")
    for ag in run["age_groups"]:
        axes[0].plot(
            run["week_dates"],
            run["median_weekly_base_R_t"][:, ag],
            label=run["age_group_names"][ag],
            c=colors[ag],
        )
    #     axes[0].set_facecolor(FACECOLOR)
    #     axes[0].axvspan(
    #         START_FIRST_WAVE,
    #         END_FIRST_WAVE,
    #         alpha=1.0,
    #         color="skyblue",
    #     )
    #     axes[0].axvspan(
    #         START_SECOND_WAVE,
    #         END_SECOND_WAVE,
    #         alpha=1.0,
    #         color="skyblue",
    #     )
    y_arrow = 3.85
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0
    
    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), y_arrow
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
    )
    axes[0].add_patch(arrow)
    axes[0].text(x_tail + 0.5*dx, y_head + 0.1, "3rd Wave", ha="center", va="bottom")
    
    x_tail, y_tail = mdates.date2num(START_SECOND_WAVE), y_arrow
    x_head, y_head = mdates.date2num(END_SECOND_WAVE), y_arrow
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
    )
    axes[0].add_patch(arrow)
    axes[0].text(x_tail + 0.5*dx, y_head + 0.1, "4th Wave", ha="center", va="bottom")
    
    axes[0].set_ylabel("Base reproduction number")
    axes[0].set_xlim(left=run["week_dates"].min(), right=run["week_dates"].max())

    infection_dynamics_df = run["infection_dynamics_df"]
    infection_dynamics_df = infection_dynamics_df[
        infection_dynamics_df["Age_group"] != "total"
    ]
    infection_array = infection_dynamics_df.sort_values(["Sunday_date", "Age_group"])[
        ["total_infections_observed"]
    ].values.reshape((-1, 9))
    weight_inf = infection_array[:-1, :]
    weighted_inf_avg_base_R_t = (
        run["median_weekly_base_R_t"][:-1, :] * weight_inf
    ).sum(axis=0) / weight_inf.sum(axis=0)

    #     axes[1].set_title("Reproduction Number")
    axes[1].bar(run["age_group_names"], weighted_inf_avg_base_R_t, color=colors)
    #     pd.DataFrame(
    #         {
    #             "age_groups": run["age_group_names"],
    #             "avg_R_base": weighted_inf_avg_base_R_t,
    #         }
    #     ).set_index("age_groups").plot.bar(ax=axes[1], legend=False)
    axes[1].set_ylabel("Infection Weighted Average\nBase Reproduction Number")
    axes[1].set_xlabel("Age Group")
    axes[1].tick_params(axis="x", labelrotation=0)

    h, l = axes[1].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=5,
        loc="lower right",
        bbox_to_anchor=(0.99, 1.0),
        title="Age Group",
    )

    if save:
        plt.savefig(OUTPUT_DIR / f"reproduction_number.pdf")

    plt.show()


run = RUNS[list(RUNS.keys())[0]]
run_name = "test"
# print_R(run=run, run_name=run_name, save=SAVE_PLOTS)
# + code_folding=[]
def print_R2(run, run_name, save=False):
    fig = plt.figure(
        figsize=(SINGLE_COLUMN, 0.5 * SINGLE_COLUMN),
        dpi=500,
        tight_layout=True,
    )
    gs = gridspec.GridSpec(1, 5)
#     gs.update(wspace=1.9, hspace=0.9)
    
#     plt.subplots_adjust(
#         #         hspace=0.15,
#         wspace=0.3,
#     )
    
    ax0 = fig.add_subplot(gs[0, :3])
    ax1 = fig.add_subplot(gs[0, 3:])
    axes = [ax0, ax1]

    colors = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))

    #     axes[0].set_title("base R")
    for ag in run["age_groups"]:
        axes[0].plot(
            run["week_dates"],
            run["median_weekly_base_R_t"][:, ag],
            label=run["age_group_names"][ag],
            c=colors[ag],
            ls="-",
        )
    #     axes[0].set_facecolor(FACECOLOR)
    #     axes[0].axvspan(
    #         START_FIRST_WAVE,
    #         END_FIRST_WAVE,
    #         alpha=1.0,
    #         color="skyblue",
    #     )
    #     axes[0].axvspan(
    #         START_SECOND_WAVE,
    #         END_SECOND_WAVE,
    #         alpha=1.0,
    #         color="skyblue",
    #     )
    y_arrow = 3.85
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0.0
    
    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), y_arrow
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
    )
    axes[0].add_patch(arrow)
    axes[0].text(x_tail + 0.5*dx, y_head + 0.1, "3rd Wave", ha="center", va="bottom")
    
    x_tail, y_tail = mdates.date2num(START_SECOND_WAVE), y_arrow
    x_head, y_head = mdates.date2num(END_SECOND_WAVE), y_arrow
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
    )
    axes[0].add_patch(arrow)
    axes[0].text(x_tail + 0.5*dx, y_head + 0.1, "4th Wave", ha="center", va="bottom")
    
    axes[0].set_ylabel("Base reproduction number")
    axes[0].tick_params(axis="x", labelrotation=0)
    axes[0].set_ylim(bottom=None, top=4.5)
    axes[0].set_xlabel(2021)
    
    myFmt = mdates.DateFormatter('%b')
    axes[0].xaxis.set_major_formatter(myFmt)
    axes[0].set_xlim(left=datetime.date(year=2020, month=12, day=15), right=datetime.date(year=2021, month=12, day=31))

    infection_dynamics_df = run["infection_dynamics_df"]
    infection_dynamics_df = infection_dynamics_df[
        infection_dynamics_df["Age_group"] != "total"
    ]
    infection_array = infection_dynamics_df.sort_values(["Sunday_date", "Age_group"])[
        ["total_infections_observed"]
    ].values.reshape((-1, 9))
    weight_inf = infection_array[:-1, :]
    weighted_inf_avg_base_R_t = (
        run["median_weekly_base_R_t"][:-1, :] * weight_inf
    ).sum(axis=0) / weight_inf.sum(axis=0)

    #     axes[1].set_title("Reproduction Number")
    axes[1].bar(run["age_group_names"], weighted_inf_avg_base_R_t, color=colors)
    #     pd.DataFrame(
    #         {
    #             "age_groups": run["age_group_names"],
    #             "avg_R_base": weighted_inf_avg_base_R_t,
    #         }
    #     ).set_index("age_groups").plot.bar(ax=axes[1], legend=False)
    axes[1].set_ylabel("Infection weighted average\nbase reproduction number")
    axes[1].set_xlabel("Age group")
    axes[1].tick_params(axis="x", labelrotation=90, labelbottom=False)

    h, l = axes[1].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=5,
        loc="lower right",
        bbox_to_anchor=(0.99, 1.0),
        title="Age group",
    )

    if save:
        plt.savefig(OUTPUT_DIR / f"reproduction_number2.pdf")

    plt.show()


run = RUNS[list(RUNS.keys())[0]]
run_name = "test"
print_R2(run=run, run_name=run_name, save=SAVE_PLOTS)


# + code_folding=[]
def print_R2(run, run_name, save=False):
    def add_event(ax, date, y=3.0, text="hi", down=True, dx_text=0.0):
        if down:
            y_event_head = y
            y_event_tail = y_event_head + 0.5
        else:
            y_event_tail = y - 0.5
            y_event_head = y
        
        arrow_scale = 5
        arrow_color = "black"
        arrow_lw = 1
        arrow_shrink = 0.0
        
        x_event = mdates.date2num(date)
        
        x_tail, y_tail = x_event, y_event_tail
        x_head, y_head = x_event, y_event_head
        arrow = mpatches.FancyArrowPatch(
            (x_tail, y_tail),
            (x_head, y_head),
            mutation_scale=arrow_scale,
            arrowstyle="->",
            color=arrow_color,
            lw=arrow_lw,
            shrinkA=arrow_shrink,
            shrinkB=arrow_shrink,
        )
        ax.add_patch(arrow)
        dy = 0.1 if down else - 0.1
        ax.text(x_tail + dx_text, y_tail + dy, text, ha="center", va="bottom" if down else "top")
    
    fig = plt.figure(
        figsize=(SINGLE_COLUMN, 0.5 * SINGLE_COLUMN),
        dpi=500,
        tight_layout=True,
    )
    gs = gridspec.GridSpec(1, 5)
    
    ax0 = fig.add_subplot(gs[0, :3])
    ax1 = fig.add_subplot(gs[0, 3:])
    axes = [ax0, ax1]

    colors = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))

    #     axes[0].set_title("base R")
    for ag in run["age_groups"]:
        axes[0].plot(
            run["week_dates"],
            run["median_weekly_base_R_t"][:, ag],
            label=run["age_group_names"][ag],
            c=colors[ag],
            ls="-",
        )
    
    # event arrows
    add_event(ax=axes[0], date=datetime.date(year=2020, month=12, day=27), y=2.0, text="a")
    add_event(ax=axes[0], date=datetime.date(year=2021, month=2, day=7), y=2.5, text="b")
    add_event(ax=axes[0], date=datetime.date(year=2021, month=2, day=21), y=2.5, text="c")
    add_event(ax=axes[0], date=datetime.date(year=2021, month=3, day=7), y=2.2, text="d")
    add_event(ax=axes[0], date=datetime.date(year=2021, month=3, day=19), y=2.7, text="e")
    add_event(ax=axes[0], date=datetime.date(year=2021, month=6, day=1), y=1.1, text="f", down=False)
    add_event(ax=axes[0], date=datetime.date(year=2021, month=6, day=20), y=1.1, text="g", down=False, dx_text=0)
    add_event(ax=axes[0], date=datetime.date(year=2021, month=6, day=25), y=1.6, text="h", down=False, dx_text=8)
    add_event(ax=axes[0], date=datetime.date(year=2021, month=7, day=29), y=1.1, text="i", down=False)
    add_event(ax=axes[0], date=datetime.date(year=2021, month=8, day=31), y=2.5, text="j")
    
    
    y_arrow = 4.0
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0.0
    
    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), y_arrow
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
        clip_on=False,
    )
    axes[0].add_patch(arrow)
    axes[0].text(x_tail + 0.5*dx, y_head + 0.1, "3rd Wave", ha="center", va="bottom")
    
    x_tail, y_tail = mdates.date2num(START_SECOND_WAVE), y_arrow
    x_head, y_head = mdates.date2num(END_SECOND_WAVE), y_arrow
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
        clip_on=False,
    )
    axes[0].add_patch(arrow)
    axes[0].text(x_tail + 0.5*dx, y_head + 0.1, "4th Wave", ha="center", va="bottom")
    
    axes[0].set_ylabel("Base reproduction number")
    axes[0].tick_params(axis="x", labelrotation=0)
    axes[0].set_ylim(bottom=None, top=4.7)
    axes[0].set_xlabel(2021)
    
    myFmt = mdates.DateFormatter('%b')
    axes[0].xaxis.set_major_formatter(myFmt)
    axes[0].set_xlim(left=datetime.date(year=2020, month=12, day=15), right=datetime.date(year=2021, month=12, day=31))

    infection_dynamics_df = run["infection_dynamics_df"]
    infection_dynamics_df = infection_dynamics_df[
        infection_dynamics_df["Age_group"] != "total"
    ]
    infection_array = infection_dynamics_df.sort_values(["Sunday_date", "Age_group"])[
        ["total_infections_observed"]
    ].values.reshape((-1, 9))
    weight_inf = infection_array[:-1, :]
    weighted_inf_avg_base_R_t = (
        run["median_weekly_base_R_t"][:-1, :] * weight_inf
    ).sum(axis=0) / weight_inf.sum(axis=0)

    #     axes[1].set_title("Reproduction Number")
    axes[1].bar(run["age_group_names"], weighted_inf_avg_base_R_t, color=colors)
    axes[1].set_ylabel("Infection weighted average\nbase reproduction number")
    axes[1].set_xlabel("Age group")
    axes[1].tick_params(axis="x", labelrotation=90, labelbottom=False)

    h, l = axes[1].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=5,
        loc="lower right",
        bbox_to_anchor=(0.99, 1.0),
        title="Age group",
    )

    if save:
        plt.savefig(OUTPUT_DIR / f"reproduction_number3.pdf")

    plt.show()


run = RUNS[list(RUNS.keys())[0]]
run_name = "test"
print_R2(run=run, run_name=run_name, save=SAVE_PLOTS)


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(
        1, 3, 
        figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        dpi=500,
        sharey=True, 
        sharex=True,
    )
    plt.suptitle("Final vaccine distribution")

    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        
        first_doses = compute_weekly_first_doses_per_age(run["U_2"]).sum(axis=-1) / run["D_a"]
        axes[0].plot(run['age_group_names'], first_doses, label=policy_label)
        
        second_doses = compute_weekly_second_doses_per_age(run["U_2"]).sum(axis=-1) / run["D_a"]
        axes[1].plot(run['age_group_names'], second_doses, label=policy_label)
        
        third_doses = compute_weekly_third_doses_per_age(run["U_2"], run["u_3"]).sum(axis=-1) / run["D_a"]
        axes[2].plot(run['age_group_names'], third_doses, label=policy_label)
        
    
    axes[0].set_title("1st dose")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend()
    
    axes[1].set_title("2nd dose")
    axes[1].tick_params(axis="x", labelrotation=45)
    
    axes[2].set_title("3rd dose")
    axes[2].tick_params(axis="x", labelrotation=45)


    plt.show()

plot()
# -
run = RUNS['observed']
possible_third_doses = run["D_a"] - compute_weekly_third_doses_per_age(run["U_2"], run["u_3"]).sum(axis=-1)
possible_second_doses = run["D_a"] - compute_weekly_second_doses_per_age(run["U_2"]).sum(axis=-1)
possible_first_doses = run["D_a"] - compute_weekly_first_doses_per_age(run["U_2"]).sum(axis=-1)

possible_third_doses

possible_second_doses

possible_first_doses

0.0060 * run["D_a"].sum()

possible_first_doses[-4:].sum()


# +
def plot(save=False):
    LW_LOCAL = 0.8
    labelrotation = 45

    fig = plt.figure(
        #         tight_layout=True,
        #         figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        figsize=(TEXT_WIDTH_WORKSHOP, 0.4 * TEXT_WIDTH_WORKSHOP),
        dpi=500,
        constrained_layout=True,
    )
    #     plt.suptitle("Severe cases (per 100k)")
    gs = gridspec.GridSpec(4, 4, figure=fig)
    ax00 = fig.add_subplot(gs[:2, 2])
    ax01 = fig.add_subplot(gs[:2, 3], sharey=ax00)
    ax10 = fig.add_subplot(gs[2:, 2])
    ax11 = fig.add_subplot(gs[2:, 3], sharey=ax10)

    ax12 = fig.add_subplot(gs[1:, :2])

    axes = np.array([[ax00, ax01], [ax10, ax11]])
    
    all_axes = np.array([ax12, ax00, ax01, ax10, ax11])

    #     fig, axes = plt.subplots(
    #         2,
    #         2,
    #         #         figsize=(0.8 * PAGE_WIDTH, 0.45 * PAGE_WIDTH),
    #         figsize=(SINGLE_COLUMN, 0.8 * SINGLE_COLUMN),
    #         dpi=500,
    #         sharey="row",
    #         sharex=False,
    #         tight_layout=True,
    #     )
    #     #     plt.suptitle("Severe cases (per 100k)")
    #     plt.subplots_adjust(
    #         hspace=0.35,
    #         wspace=0.2,
    #     )
    TWO_LINE_NAME_MAP = {
        "elderly_first": "Elderly\nFirst",
        "observed": "Factual",
        "uniform": "Uniform",
        "young_first": "Young\nFirst",
        "risk_ranked": "Risk\nRanked",
        "risk_ranked_reversed": "Risk\nRanked\nReversed",
    }

    import matplotlib.transforms as mtransforms

    trans = mtransforms.ScaledTranslation(8 / 72, -8 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)"], all_axes.flatten()):
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

    # infections first wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_0
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = (
            infection_dynamics_df["total_infections_scenario"].sum()
            * 1e5
            / (population)
        )  # / 1e5
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[0, 0].set_xticklabels(["", "", "", ""])
    axes[0, 0].set_title("Third wave")
    axes[0, 0].set_ylabel("Infections\n(cum., per 100k)")

    # infections second wave
    no_infections = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        population = run["D_a"].sum()
        infection_dynamics_df = run["infection_dynamics_df"]
        infection_dynamics_df["Sunday_date"] = pd.to_datetime(
            infection_dynamics_df["Sunday_date"]
        )
        infection_dynamics_df = infection_dynamics_df[
            (infection_dynamics_df["Sunday_date"] >= SPLIT_DATE_WAVES_0)
            & (infection_dynamics_df["Sunday_date"] < SPLIT_DATE_WAVES_1)
        ]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections[policy_label] = (
            infection_dynamics_df["total_infections_scenario"].sum() * 1e5 / population
        )
    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 1].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    )
    axes[0, 1].set_xticklabels(["", "", "", ""])
#     axes[0, 1].set_yticklabels([])
    axes[0, 1].set_title("Fourth wave")
    axes[0, 1].tick_params(
        axis="y",
        which="both",
        labelleft=False,
    )

    # severe cases first wave
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = (
            n_weeks * run["result"][:split_date_index, ...].sum() * 1e5
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    #     axes[1, 0].set_title("Severe Cases Third Wave")
    axes[1, 0].set_ylabel("Severe cases\n(cum., per 100k)")
    axes[1, 0].tick_params(axis="x", labelrotation=labelrotation, labelsize="medium")
    print(no_severe_cases)

    # severe cases second wave
    no_severe_cases = {}
    for policy_name, policy_label in POLICY_NAME_MAP.items():
        if policy_name not in RUNS.keys():
            continue
        run = RUNS[policy_name]
        split_date_index = np.argwhere(
            run["week_dates"] == SPLIT_DATE_WAVES_0
        ).flatten()[0]
        end_date_index = np.argwhere(run["week_dates"] == SPLIT_DATE_WAVES_1).flatten()[
            0
        ]
        population = run["D_a"].sum()
        n_weeks = len(run["weeks"])
        no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = (
            n_weeks * run["result"][split_date_index:end_date_index, ...].sum() * 1e5
        )
    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    #     axes[1, 1].set_title("Severe Cases Fourth Wave")
    axes[1, 1].tick_params(axis="x", labelrotation=labelrotation)
    #     print(axes[1, 1].get_xticklabels())
    axes[1, 1].set_xticklabels(df.index, rotation=labelrotation)
    axes[1, 1].tick_params(
        axis="y",
        which="both",
        labelleft=False,
    )
    print(no_severe_cases)

    colors = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    run = RUNS["observed"]
    #     axes[0].set_title("base R")
    for ag in run["age_groups"]:
        ax12.plot(
            run["week_dates"],
            run["median_weekly_base_R_t"][:, ag],
            label=run["age_group_names"][ag],
            c=colors[ag],
            ls="-",
        )

    y_arrow_first = 3.2
    y_arrow_second = 3.85
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0.0

    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), y_arrow_first
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), y_arrow_first
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
    )
    ax12.add_patch(arrow)
    ax12.text(x_tail + 0.5 * dx, y_head + 0.1, "3rd Wave", ha="center", va="bottom")

    x_tail, y_tail = mdates.date2num(START_SECOND_WAVE), y_arrow_second
    x_head, y_head = mdates.date2num(END_SECOND_WAVE), y_arrow_second
    dx = x_head - x_tail
    arrow = mpatches.FancyArrowPatch(
        (x_tail, y_tail),
        (x_head, y_head),
        mutation_scale=arrow_scale,
        arrowstyle="|-|",
        color=arrow_color,
        lw=arrow_lw,
        shrinkA=arrow_shrink,
        shrinkB=arrow_shrink,
    )
    ax12.add_patch(arrow)
    ax12.text(x_tail + 0.5 * dx, y_head + 0.1, "4th Wave", ha="center", va="bottom")

    ax12.set_ylabel("Base reproduction number")
    ax12.tick_params(axis="x", labelrotation=0)
    ax12.set_ylim(bottom=None, top=5)
    ax12.set_xlabel(2021)

    myFmt = mdates.DateFormatter("%b")
    ax12.xaxis.set_major_formatter(myFmt)
    ax12.set_xlim(
        left=datetime.date(year=2020, month=12, day=15),
        right=datetime.date(year=2021, month=12, day=31),
    )
    h, l = ax12.get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.75),
        title="Age group",
        fontsize="small",
    )

#     ax12.legend(
#         loc="lower center",
#         ncol=3,
#         title="Age group",
#         fontsize="small",
#         bbox_to_anchor=(0.5, 1.0),
#     )

    if save:
        plt.savefig(OUTPUT_DIR / f"policy_exp_cumulative_w{OUTPUT_EXTENSION}.pdf")

    plt.show()


plot(save=SAVE_PLOTS)
# -


