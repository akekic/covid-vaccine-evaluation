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

# # Counterfactual vaccine allocation strategy evaluation

# ## 1. Imports

# +
import os
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
)


plt.rcParams.update(style_modifications)

SAVE_PLOTS = True
SPLIT_DATE_WAVES_0 = np.datetime64(datetime.date(year=2021, month=5, day=30))

END_FIRST_WAVE = np.datetime64(END_FIRST_WAVE)

START_SECOND_WAVE = np.datetime64(START_SECOND_WAVE)
END_SECOND_WAVE = np.datetime64(END_SECOND_WAVE)

# SPLIT_DATE_WAVES_1 = np.datetime64(END_SECOND_WAVE)


# -

# ## 2. Input: run directory
# The paths below need to be set to the respective local paths.

# +
# path to preprocessed data, i.e. output of ../data_preprocessing/israel_data_processing.py
INPUT_DATA_DIR_PATH = Path("../data/preprocessed-data/israel_df.pkl")

# path to risk profile experiment output directory, i.e. output of ../experiments/policy_exp.py
# OUTPUT_EXTENSION indicates which mixing factor C_mat_param was used
# C_mat_param=90
# OUTPUT_EXTENSION, EXP_DIR = "_C70", Path("../run/2022-08-19_15-47-18.888241_policy_exp")
# C_mat_param=80
OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-08-16_10-25-16.053831_policy_exp")
# C_mat_param=70
# OUTPUT_EXTENSION, EXP_DIR = "_C90", Path("../run/2022-08-19_15-46-57.554281_policy_exp")


OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
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


# ## 3. Plots

# + code_folding=[]
def plot(save=False):
    LW_LOCAL = 0.8
    labelrotation = 45

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COLUMN, 0.8 * SINGLE_COLUMN),
        dpi=500,
        sharey="row",
        sharex=False,
        tight_layout=True,
    )
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
                split_date_to=END_FIRST_WAVE,
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
                        split_date_to=END_FIRST_WAVE,
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
    no_infections_err_low = {}
    no_infections_err_high = {}
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
                split_date_from=START_SECOND_WAVE,
                end_date=END_SECOND_WAVE,
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
                        split_date_from=START_SECOND_WAVE,
                        end_date=END_SECOND_WAVE,
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
                split_date_to=END_FIRST_WAVE,
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
                        split_date_to=END_FIRST_WAVE,
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
                split_date_from=START_SECOND_WAVE,
                end_date=END_SECOND_WAVE,
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
                        split_date_from=START_SECOND_WAVE,
                        end_date=END_SECOND_WAVE,
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


# + code_folding=[0]
def plot(save=False):
    run_list = ["observed", "uniform", "elderly_first", "young_first"]
    LW_LOCAL = 1.5

    fig = plt.figure(
        figsize=(DOUBLE_COLUMN, 0.4 * DOUBLE_COLUMN),
        dpi=500,
        constrained_layout=True,
    )
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
                )
            
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


# + code_folding=[]
def plot(save=False):
    LW_LOCAL = 0.8
    labelrotation = 45

    fig = plt.figure(
        figsize=(TEXT_WIDTH_WORKSHOP, 0.4 * TEXT_WIDTH_WORKSHOP),
        dpi=500,
        constrained_layout=True,
    )
    gs = gridspec.GridSpec(4, 4, figure=fig)
    ax00 = fig.add_subplot(gs[:2, 2])
    ax01 = fig.add_subplot(gs[:2, 3], sharey=ax00)
    ax10 = fig.add_subplot(gs[2:, 2])
    ax11 = fig.add_subplot(gs[2:, 3], sharey=ax10)

    ax12 = fig.add_subplot(gs[1:, :2])

    axes = np.array([[ax00, ax01], [ax10, ax11]])
    
    all_axes = np.array([ax12, ax00, ax01, ax10, ax11])
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
        if policy_name == "observed":
            no_infections[
                TWO_LINE_NAME_MAP[policy_name]
            ] = infection_incidence_observed(
                df_input=df_input,
                population=population,
                split_date_to=END_FIRST_WAVE,
            )
        else:
            a = np.array(
                [
                    infection_incidence(
                        infection_dynamics_sample=inf_sample,
                        population=population,
                        week_dates=run["week_dates"],
                        split_date_to=END_FIRST_WAVE,
                    )
                    for inf_sample in run["infection_dynamics_samples"]
                ]
            )

            no_infections[TWO_LINE_NAME_MAP[policy_name]] = a.mean()
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
                split_date_from=START_SECOND_WAVE,
                end_date=END_SECOND_WAVE,
            )
        else:
            a = np.array(
                [
                    infection_incidence(
                        infection_dynamics_sample=inf_sample,
                        population=population,
                        week_dates=run["week_dates"],
                        split_date_from=START_SECOND_WAVE,
                        end_date=END_SECOND_WAVE,
                    )
                    for inf_sample in run["infection_dynamics_samples"]
                ]
            )

            no_infections[TWO_LINE_NAME_MAP[policy_name]] = a.mean()

    df = pd.DataFrame.from_dict(no_infections, orient="index")
    axes[0, 1].bar(
        df.index,
        df[0],
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    )
    axes[0, 1].set_xticklabels(["", "", "", ""])
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
        population = run["D_a"].sum()

        if policy_name == "observed":
            no_severe_cases[
                TWO_LINE_NAME_MAP[policy_name]
            ] = severe_case_incidence_observed(
                df_input=df_input,
                population=population,
                split_date_to=END_FIRST_WAVE,
            )
        else:
            a = np.array(
                [
                    severe_case_incidence(
                        res,
                        len(run["weeks"]),
                        run["week_dates"],
                        split_date_to=END_FIRST_WAVE,
                    )
                    for res in run["result_samples"]
                ]
            )
            no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = a.mean()

    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 0].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[1, 0].set_ylabel("Severe cases\n(cum., per 100k)")
    axes[1, 0].tick_params(axis="x", labelrotation=labelrotation, labelsize="medium")
    print(no_severe_cases)

    # severe cases second wave
    no_severe_cases = {}
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
                split_date_from=START_SECOND_WAVE,
                end_date=END_SECOND_WAVE,
            )
        else:
            a = np.array(
                [
                    severe_case_incidence(
                        res,
                        len(run["weeks"]),
                        run["week_dates"],
                        split_date_from=START_SECOND_WAVE,
                        end_date=END_SECOND_WAVE,
                    )
                    for res in run["result_samples"]
                ]
            )
            no_severe_cases[TWO_LINE_NAME_MAP[policy_name]] = a.mean()

    df = pd.DataFrame.from_dict(no_severe_cases, orient="index")
    axes[1, 1].bar(
        df.index, df[0], color=plt.rcParams["axes.prop_cycle"].by_key()["color"]
    )
    axes[1, 1].tick_params(axis="x", labelrotation=labelrotation)
    axes[1, 1].set_xticklabels(df.index, rotation=labelrotation)
    axes[1, 1].tick_params(
        axis="y",
        which="both",
        labelleft=False,
    )
    print(no_severe_cases)

    colors = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    run = RUNS["observed"]
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

    if save:
        plt.savefig(OUTPUT_DIR / f"policy_exp_cumulative_w{OUTPUT_EXTENSION}.pdf")

    plt.show()


plot(save=SAVE_PLOTS)
# -


