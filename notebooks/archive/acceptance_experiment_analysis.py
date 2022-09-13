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

# # Vaccine Acceptance Experiment

# ## 1. Imports

# +
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools

from pathlib import Path
from itertools import product

from common_functions import (
    load_run,
    infection_incidence,
    severe_case_incidence,
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
)

# plt.style.use("ggplot")
plt.rcParams.update(style_modifications)

LW_GLOBAL = 1
MS_GLOBAL = 3

SAVE_PLOTS = True

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# C_mat_param=100, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-50.164332_acc_exp)
# C_mat_param=100, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=2
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-42.880729_acc_exp")

# C_mat_param=95, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-45.325898_acc_exp")
# C_mat_param=95, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=2
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-45.103742_acc_exp")

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-20_16-05-47.641929_acc_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-44.399465_acc_exp")
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_09-26-12.803871_acc_exp")
# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=2
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-53.867483_acc_exp")
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-14_13-20-00.387376_acc_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-20_16-05-40.579653_acc_exp")

# C_mat_param=60, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-43.615181_acc_exp")
# C_mat_param=60, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=2
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-42.748400_acc_exp")

# C_mat_param=40, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-43.875522_acc_exp")
# C_mat_param=40, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=2
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-45.570660_acc_exp")

# C_mat_param=20, V1_eff=70, V2_eff=90, V3_eff=95, draws=200, influx=1
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-43.648108_acc_exp")
# C_mat_param=20, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=2
# EXP_NAME, EXP_DIR = "exp", Path("../run/2022-05-26_09-35-42.775513_acc_exp")

# latest results

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C70", Path("../run/2022-06-21_20-48-02.403717_acc_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-06-21_20-48-02.504481_acc_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C90", Path("../run/2022-06-21_20-47-59.809703_acc_exp")

# error bars

# C_mat_param=90, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C70", Path("../run/2022-08-19_15-46-36.455990_acc_exp")

# C_mat_param=80, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-08-18_10-27-59.244998_acc_exp")
OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-08-24_09-53-48.517966_acc_exp")

# C_mat_param=70, V1_eff=70, V2_eff=90, V3_eff=95, draws=500, influx=0.5
# OUTPUT_EXTENSION, EXP_DIR = "_C90", Path("../run/2022-08-19_15-46-57.553547_acc_exp")
# -

# ## 2. Experiment: Global Acceptance Rate Change
# ### 2.1 Load Data

# ! ls -la ../run/2022-03-14_11-12-49_acc_exp_elderly_first

# + code_folding=[14]
# get all deltas for which there is a run
try:
    DELTAS_GLOBAL = [
        float(x.name[11:])
        for x in (EXP_DIR / "global").iterdir()
        if x.is_dir() and x.name[:11] == "delta_rate_"
    ]
    RUN_DIRS_GLOBAL = {
        float(x.name[11:]): x
        for x in (EXP_DIR / "global").iterdir()
        if x.is_dir() and x.name[:11] == "delta_rate_"
    }
    RUNS_GLOBAL = {delta: load_run(run_dir) for delta, run_dir in RUN_DIRS_GLOBAL.items()}
    GLOBAL = True
except FileNotFoundError:
    GLOBAL = False
    DELTAS_GLOBAL = None
    RUN_DIRS_GLOBAL = None
    RUNS_GLOBAL = None


# -

# ### 2.2 Vaccinations

# + code_folding=[0, 5, 10, 21]
def compute_weekly_first_doses(U_2):
    weekly_first_doses = U_2.sum(axis=(0, 2))[:-1]
    return weekly_first_doses


def compute_weekly_second_doses(U_2):
    weekly_second_doses = U_2.sum(axis=(0, 1))[:-1]
    return weekly_second_doses


def compute_weekly_third_doses(U_2, u_3):
    U_2_repeat = np.repeat(
        U_2[:, :-1, :-1, np.newaxis], U_2.shape[1] - 1, axis=3
    )  # copy over dimension t3
    u_3_repeat = np.repeat(
        u_3[:, np.newaxis, :-1, :-1], u_3.shape[1] - 1, axis=1
    )  # copy over dimension t1
    weekly_third_doses = (U_2_repeat * u_3_repeat).sum(axis=(0, 1, 2))
    return weekly_third_doses
    

def compute_weekly_doses(U_2, u_3):
    weekly_first_doses = compute_weekly_first_doses(U_2)
    weekly_second_doses = compute_weekly_second_doses(U_2)
    weekly_third_doses = compute_weekly_third_doses(U_2, u_3)
    
    return weekly_first_doses + weekly_second_doses + weekly_third_doses


# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    
    fig, axes = plt.subplots(4, 2, figsize=(2*PLOT_WIDTH, 4*PLOT_HEIGHT), sharey=False, sharex=False)
    plt.suptitle("Administered doses")
    
    for ard in acceptance_rate_delta:
        U_2 = RUNS_GLOBAL[ard]['U_2']
        u_3 = RUNS_GLOBAL[ard]['u_3']
        week_dates = RUNS_GLOBAL[ard]['week_dates']
        
        weekly_doses = compute_weekly_doses(U_2, u_3)
        axes[0,0].plot(week_dates, weekly_doses, label=ard)
        axes[0,1].plot(week_dates, weekly_doses.cumsum(), label=ard)
        
        weekly_first_doses = compute_weekly_first_doses(U_2)
        axes[1,0].plot(week_dates, weekly_first_doses, label=ard)
        axes[1,1].plot(week_dates, weekly_first_doses.cumsum(), label=ard)
        
        weekly_second_doses = compute_weekly_second_doses(U_2)
        axes[2,0].plot(week_dates, weekly_second_doses, label=ard)
        axes[2,1].plot(week_dates, weekly_second_doses.cumsum(), label=ard)
        
        weekly_third_doses = compute_weekly_third_doses(U_2, u_3)
        axes[3,0].plot(week_dates, weekly_third_doses, label=ard)
        axes[3,1].plot(week_dates, weekly_third_doses.cumsum(), label=ard)
    
    axes[0,0].set_title("Total")
    axes[0,0].set_ylabel("administered doses (weekly)")
    axes[0,0].grid()
    axes[0,0].legend(title="$\Delta$ acceptance rate")
    
    axes[0,1].set_title("Total")
    axes[0,1].set_ylabel("administered doses (cumulative)")
    axes[0,1].grid()
    axes[0,1].legend(title="$\Delta$ acceptance rate")
    
    axes[1,0].set_title("1st Doses")
    axes[1,0].set_ylabel("administered doses (weekly)")
    axes[1,0].grid()
    axes[1,0].legend(title="$\Delta$ acceptance rate")
    
    axes[1,1].set_title("1st Doses")
    axes[1,1].set_ylabel("administered doses (cumulative)")
    axes[1,1].grid()
    axes[1,1].legend(title="$\Delta$ acceptance rate")
    
    axes[2,0].set_title("2nd Doses")
    axes[2,0].set_ylabel("administered doses (weekly)")
    axes[2,0].grid()
    axes[2,0].legend(title="$\Delta$ acceptance rate")
    
    axes[2,1].set_title("2nd Doses")
    axes[2,1].set_ylabel("administered doses (cumulative)")
    axes[2,1].grid()
    axes[2,1].legend(title="$\Delta$ acceptance rate")
    
    axes[3,0].set_title("3rd Doses")
    axes[3,0].set_ylabel("administered doses (weekly)")
    axes[3,0].grid()
    axes[3,0].legend(title="$\Delta$ acceptance rate")
    
    axes[3,1].set_title("3rd Doses")
    axes[3,1].set_ylabel("administered doses (cumulative)")
    axes[3,1].grid()
    axes[3,1].legend(title="$\Delta$ acceptance rate")
    
    plt.tight_layout()
    plt.show()

if GLOBAL:
    plot()
# -

# ### 2.3 Infections

# + code_folding=[0]
if GLOBAL:
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))

    plt.figure()

    for ard in acceptance_rate_delta:
        infection_dynamics_df = RUNS_GLOBAL[ard]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        week_dates = RUNS_GLOBAL[ard]["week_dates"]
        plt.plot(
            week_dates,
            infection_dynamics_df["total_infections_scenario"],
            label=ard,
        )
    plt.ylabel("Number of infections")
    plt.title("Infection dynamics")
    plt.legend(title="$\Delta$ acceptance rate", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()


# -

# ### 2.4 Severe Cases

# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    population = RUNS_GLOBAL[acceptance_rate_delta[0]]["D_a"].sum()
    n_weeks = len(RUNS_GLOBAL[acceptance_rate_delta[0]]["weeks"])
    no_severe_cases = (
        population
        * n_weeks
        * np.array([RUNS_GLOBAL[acc]["result"].sum() for acc in acceptance_rate_delta])
    )
    no_severe_cases_no_id = (
        population
        * n_weeks
        * np.array(
            [RUNS_GLOBAL[acc]["result_no_id"].sum() for acc in acceptance_rate_delta]
        )
    )

    plt.figure(figsize=(0.4 * PAGE_WIDTH, 0.4 * PAGE_WIDTH), dpi=200)
    plt.title("Severe cases")
    ax = plt.gca()
    plt.plot(100 * acceptance_rate_delta, no_severe_cases, "-o", label="inf. dyn.")
    plt.plot(
        100 * acceptance_rate_delta,
        no_severe_cases_no_id,
        "-o",
        label="no inf. dyn.",
        lw=LW_GLOBAL,
        ms=MS_GLOBAL,
    )
    plt.xlabel("Change in acceptance rate (percentage points)")
    plt.ylabel("No. of severe cases")
    plt.legend()
    plt.show()


if GLOBAL:
    plot()


# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    age_groups = RUNS_GLOBAL[acceptance_rate_delta[0]]["age_groups"]
    age_group_names = RUNS_GLOBAL[acceptance_rate_delta[0]]["age_group_names"]
    D_a = RUNS_GLOBAL[acceptance_rate_delta[0]]["D_a"]
    n_weeks = len(RUNS_GLOBAL[acceptance_rate_delta[0]]["weeks"])

    fig, axes = plt.subplots(
        1, 2, figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH), sharey=True, sharex=True, dpi=500
    )

    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle("Severe cases by age group")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(age_groups)))

    for ag, c in zip(age_groups, colors):
        no_severe_cases = (
            D_a[ag]
            * n_weeks
            * np.array(
                [
                    RUNS_GLOBAL[acc]["result"][:, ag, :].sum()
                    for acc in acceptance_rate_delta
                ]
            )
        )
        no_severe_cases_no_id = (
            D_a[ag]
            * n_weeks
            * np.array(
                [
                    RUNS_GLOBAL[acc]["result_no_id"][:, ag, :].sum()
                    for acc in acceptance_rate_delta
                ]
            )
        )

        axes[0].plot(
            100 * acceptance_rate_delta,
            no_severe_cases,
            "-o",
            label=age_group_names[ag],
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
        axes[1].plot(
            100 * acceptance_rate_delta,
            no_severe_cases_no_id,
            "-o",
            label=age_group_names[ag],
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )

    axes[0].set_title("with infection dynamics")
    axes[0].set_xlabel("Change in acceptance rate (percentage points)")
    axes[0].set_ylabel("No. of severe cases")

    axes[1].set_title("no infection dynamics")
    axes[1].set_xlabel("Change in acceptance rate (percentage points)")
    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="age groups",
    )

    plt.show()


if GLOBAL:
    plot()


# -

# ### 2.5 Change in Severity

# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    severity = np.array(
        [RUNS_GLOBAL[acc]["result"].sum() for acc in acceptance_rate_delta]
    )
    severity_no_id = np.array(
        [RUNS_GLOBAL[acc]["result_no_id"].sum() for acc in acceptance_rate_delta]
    )
    normalisation = severity[np.argwhere(acceptance_rate_delta == 0)].flatten()
    normalisation_no_id = severity_no_id[
        np.argwhere(acceptance_rate_delta == 0)
    ].flatten()

    plt.figure(figsize=(0.4 * PAGE_WIDTH, 0.4 * PAGE_WIDTH), dpi=200)
    plt.title("Change in severity (relative)")
    ax = plt.gca()
    plt.plot(
        100 * acceptance_rate_delta,
        100 * (severity / normalisation - 1),
        "-o",
        label="inf. dyn.",
        lw=LW_GLOBAL,
        ms=MS_GLOBAL,
    )
    plt.plot(
        100 * acceptance_rate_delta,
        100 * (severity_no_id / normalisation_no_id - 1),
        "-o",
        label="no inf. dyn.",
        lw=LW_GLOBAL,
        ms=MS_GLOBAL,
    )
    plt.xlabel("Change in acceptance rate (percentage points)")
    plt.ylabel("Change in no. of severe cases (%)")
    plt.legend()
    plt.show()


if GLOBAL:
    plot()


# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    age_groups = RUNS_GLOBAL[acceptance_rate_delta[0]]['age_groups']
    age_group_names = RUNS_GLOBAL[acceptance_rate_delta[0]]['age_group_names']
    
    fig, axes = plt.subplots(1, 2, figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH), dpi=500, sharey=True, sharex=True)
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(age_groups)))
    
    for ag, c in zip(age_groups, colors):
        severity = np.array([RUNS_GLOBAL[acc]['result'][:, ag, :].sum() for acc in acceptance_rate_delta])
        normalisation = severity[np.argwhere(acceptance_rate_delta == 0)].flatten()
        
        severity_no_id = np.array([RUNS_GLOBAL[acc]['result_no_id'][:, ag, :].sum() for acc in acceptance_rate_delta])
        normalisation_no_id = severity_no_id[np.argwhere(acceptance_rate_delta == 0)].flatten()

        axes[0].plot(
            100*acceptance_rate_delta,
            100*(severity/normalisation - 1),
            "-o",
            label=age_group_names[ag],
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
        axes[1].plot(
            100*acceptance_rate_delta,
            100*(severity_no_id/normalisation_no_id - 1),
            "-o",
            label=age_group_names[ag],
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
    
    axes[0].set_title("with infection dynamics")
    axes[0].set_xlabel("Change in acceptance rate (percentage points)")
    axes[0].set_ylabel("Change in no. of severe cases (%)")
    
    axes[1].set_title("no infection dynamics")
    axes[1].set_xlabel("Change in acceptance rate (percentage points)")
    axes[1].set_ylabel("Change in no. of severe cases (%)")
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='age groups', facecolor=FACECOLOR)
    plt.show()

if GLOBAL: plot()


# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    age_groups = RUNS_GLOBAL[acceptance_rate_delta[0]]["age_groups"]
    age_group_names = RUNS_GLOBAL[acceptance_rate_delta[0]]["age_group_names"]

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(0.8 * PAGE_WIDTH, 0.8 * PAGE_WIDTH),
        dpi=500,
        sharey=True,
        sharex=True,
    )
    ax_flat = axes.flatten()

    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle("Change in severity by age group (relative)")

    for ag, ax in zip(age_groups, ax_flat):
        severity = np.array(
            [
                RUNS_GLOBAL[acc]["result"][:, ag, :].sum()
                for acc in acceptance_rate_delta
            ]
        )
        normalisation = severity[np.argwhere(acceptance_rate_delta == 0)].flatten()

        severity_no_id = np.array(
            [
                RUNS_GLOBAL[acc]["result_no_id"][:, ag, :].sum()
                for acc in acceptance_rate_delta
            ]
        )
        normalisation_no_id = severity_no_id[
            np.argwhere(acceptance_rate_delta == 0)
        ].flatten()

        ax.plot(
            100 * acceptance_rate_delta,
            100 * (severity / normalisation - 1),
            "-o",
            label="inf. dyn.",
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
        ax.plot(
            100 * acceptance_rate_delta,
            100 * (severity_no_id / normalisation_no_id - 1),
            "-o",
            label="no inf. dyn.",
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
        #         ax.grid()
        ax.set_title(age_group_names[ag])
        ax.legend()
    fig.supxlabel("Change in acceptance rate (percentage points)")
    fig.supylabel("Change in no. of severe cases (%)")
    plt.tight_layout()
    plt.show()


if GLOBAL:
    plot()


# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))

    population = RUNS_GLOBAL[acceptance_rate_delta[0]]["D_a"].sum()
    n_weeks = len(RUNS_GLOBAL[acceptance_rate_delta[0]]["weeks"])
    no_severe_cases = (
        population
        * n_weeks
        * np.array([RUNS_GLOBAL[acc]["result"].sum() for acc in acceptance_rate_delta])
    )
    normalisation = no_severe_cases[np.argwhere(acceptance_rate_delta == 0)].flatten()

    no_severe_cases_no_id = (
        population
        * n_weeks
        * np.array(
            [RUNS_GLOBAL[acc]["result_no_id"].sum() for acc in acceptance_rate_delta]
        )
    )
    normalisation_no_id = no_severe_cases_no_id[
        np.argwhere(acceptance_rate_delta == 0)
    ].flatten()

    plt.figure(figsize=(0.4 * PAGE_WIDTH, 0.4 * PAGE_WIDTH), dpi=200)
    plt.title("Change in severity (absolute)")
    ax = plt.gca()
    plt.plot(
        100 * acceptance_rate_delta,
        no_severe_cases - normalisation,
        "-o",
        label="inf. dyn.",
        lw=LW_GLOBAL,
        ms=MS_GLOBAL,
    )
    plt.plot(
        100 * acceptance_rate_delta,
        no_severe_cases_no_id - normalisation_no_id,
        "-o",
        label="no inf. dyn.",
        lw=LW_GLOBAL,
        ms=MS_GLOBAL,
    )
    plt.xlabel("Change in acceptance rate (percentage points)")
    plt.ylabel("Change in no. of severe cases")
    plt.legend()
    plt.show()


if GLOBAL:
    plot()


# + code_folding=[0]
def plot():
    acceptance_rate_delta = np.array(sorted(DELTAS_GLOBAL))
    age_groups = RUNS_GLOBAL[acceptance_rate_delta[0]]["age_groups"]
    age_group_names = RUNS_GLOBAL[acceptance_rate_delta[0]]["age_group_names"]
    D_a = RUNS_GLOBAL[acceptance_rate_delta[0]]["D_a"]
    n_weeks = len(RUNS_GLOBAL[acceptance_rate_delta[0]]["weeks"])
    population = RUNS_GLOBAL[acceptance_rate_delta[0]]["D_a"].sum()

    fig, axes = plt.subplots(
        3, 3, figsize=(0.8 * PAGE_WIDTH, 0.8 * PAGE_WIDTH), dpi=500, sharey=True, sharex=True
    )
    ax_flat = axes.flatten()

    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle("Change in severity by age group (absolute)")

    for ag, ax in zip(age_groups, ax_flat):
        no_severe_cases = (
            population
            * n_weeks
            * np.array(
                [
                    RUNS_GLOBAL[acc]["result"][:, ag, :].sum()
                    for acc in acceptance_rate_delta
                ]
            )
        )
        normalisation = no_severe_cases[
            np.argwhere(acceptance_rate_delta == 0)
        ].flatten()

        no_severe_cases_no_id = (
            population
            * n_weeks
            * np.array(
                [
                    RUNS_GLOBAL[acc]["result_no_id"][:, ag, :].sum()
                    for acc in acceptance_rate_delta
                ]
            )
        )
        normalisation_no_id = no_severe_cases_no_id[
            np.argwhere(acceptance_rate_delta == 0)
        ].flatten()

        ax.plot(
            100 * acceptance_rate_delta,
            no_severe_cases - normalisation,
            "-o",
            label="inf. dyn.",
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
        ax.plot(
            100 * acceptance_rate_delta,
            no_severe_cases_no_id - normalisation_no_id,
            "-o",
            label="no inf. dyn.",
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )
        ax.set_title(age_group_names[ag])
        ax.legend()
    fig.supxlabel("Change in acceptance rate (percentage points)")
    fig.supylabel("Change in no. of severe cases")
    plt.tight_layout()
    plt.show()


if GLOBAL: plot()
# -
# ## 3. Experiment: Acceptance Rate Change per Age Group (relative)


# + code_folding=[1, 18]
# get all deltas for which there is a run
try:
    AGE_GROUPS_REL = sorted([
        x.name for x in (EXP_DIR/"age_rel").iterdir() if x.is_dir()
    ])
    DELTAS_REL = sorted([
        float(x.name[11:]) for x in (EXP_DIR/"age_rel"/AGE_GROUPS_REL[0]).iterdir() if x.is_dir() and x.name[:11] == 'delta_rate_'
    ])
    RUN_DIRS_REL = {
        x.name: {float(y.name[11:]): y for y in x.iterdir() if y.is_dir() and y.name[:11] == 'delta_rate_'}
        for x in (EXP_DIR/"age_rel").iterdir() if x.is_dir()
    }

    RUNS_REL = {
        age_group: {delta: load_run(run_dir) for delta, run_dir in delta_dict.items()} 
        for age_group, delta_dict in RUN_DIRS_REL.items()
    }
    REL = True
except FileNotFoundError:
    REL = False
    DELTAS_REL = None
    RUN_DIRS_REL = None
    RUNS_REL = None
# -

if REL:
    pd.DataFrame(
        {
            "age_group": RUNS_REL[AGE_GROUPS_REL[0]][DELTAS_REL[0]]['age_group_names'],
            "population": RUNS_REL[AGE_GROUPS_REL[0]][DELTAS_REL[0]]['D_a'],
        }
    ).set_index("age_group").plot.bar()
    plt.show()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=True,
    )
    LW_LOCAL = 0.8

    population = RUNS_REL[AGE_GROUPS_REL[0]][min(DELTAS_REL)]["D_a"].sum()
    n_weeks = len(RUNS_REL[AGE_GROUPS_REL[0]][min(DELTAS_REL)]["weeks"])
    acceptance_rate_delta = np.array(sorted(DELTAS_REL))

    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    # plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(AGE_GROUPS_REL)))

    for age_group, c in zip(AGE_GROUPS_REL, colors):
        no_severe_cases = (
            population
            * n_weeks
            * np.array([RUNS_REL[age_group][acc]["result"].sum() for acc in DELTAS_REL])
        )
        normalisation = no_severe_cases[
            np.argwhere(acceptance_rate_delta == 0)
        ].flatten()
        axes[0].plot(
            100 * acceptance_rate_delta,
            no_severe_cases - normalisation,
            "-o",
            label=f"{age_group}",
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )

        no_infections = []
        for acc in DELTAS_REL:
            infection_dynamics_df = RUNS_REL[age_group][acc]["infection_dynamics_df"]
            infection_dynamics_df = infection_dynamics_df[
                infection_dynamics_df["Age_group"] == "total"
            ]
            no_infections.append(
                infection_dynamics_df["total_infections_scenario"].sum()
            )
        no_infections = np.array(no_infections)

        normalisation = no_infections[np.argwhere(acceptance_rate_delta == 0)].flatten()
        axes[1].plot(
            100 * acceptance_rate_delta,
            no_infections - normalisation,
            "-o",
            label=f"{age_group}",
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )

    axes[0].set_title("Severe Cases")
    axes[0].set_xlabel("Change in acceptance rate (ppt equivalent)")
    axes[0].set_ylabel("Change in no. of severe cases")

    axes[1].set_title("Infections")
    axes[1].set_xlabel("Change in acceptance rate (ppt equivalent)")
    axes[1].set_ylabel("Change in no. of infections")
    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), title="age groups")

    plt.tight_layout()
    plt.show()


if REL:
    plot()
# -

# ## 4. Experiment: Acceptance Rate Change per Age Group (absolute)


# + code_folding=[]
# get all deltas for which there is a run
try:
    AGE_GROUPS_ABS = sorted([
        x.name for x in (EXP_DIR/"age_abs").iterdir() if x.is_dir()
    ])
    DELTAS_ABS = np.array(sorted([
        float(x.name[11:]) for x in (EXP_DIR/"age_abs"/AGE_GROUPS_ABS[0]).iterdir() if x.is_dir() and x.name[:11] == 'delta_rate_'
    ]))
    RUN_DIRS_ABS = {
        x.name: {float(y.name[11:]): y for y in x.iterdir() if y.is_dir() and y.name[:11] == 'delta_rate_'}
        for x in (EXP_DIR/"age_abs").iterdir() if x.is_dir()
    }

    RUNS_ABS = {
        age_group: {delta: load_run(run_dir) for delta, run_dir in delta_dict.items()} 
        for age_group, delta_dict in RUN_DIRS_ABS.items()
    }
    ABS = True
except FileNotFoundError:
    ABS = False
    DELTAS_ABS = None
    RUN_DIRS_ABS = None
    RUNS_ABS = None

# +
# TODO: remove
# AGE_GROUPS_ABS = ['0-19', '20-29', '30-39', '40-49', '60+']

# + code_folding=[0]
if ABS:
    age_group_names = np.array(AGE_GROUPS_ABS)
    populations = np.zeros_like(age_group_names, dtype=float)
    populations[:5] = RUNS_ABS[AGE_GROUPS_ABS[0]][DELTAS_ABS[0]]['D_a'][:5]
    populations[5] = RUNS_ABS[AGE_GROUPS_ABS[0]][DELTAS_ABS[0]]['D_a'][5:].sum()

    pd.DataFrame(
        {
            "age_group": age_group_names,
            "population": populations,
        }
    ).set_index("age_group").plot.bar()
    plt.show()


# + code_folding=[0]
def plot(save=False):
    fig, axes = plt.subplots(
        1, 2, figsize=(0.8 * PAGE_WIDTH, 0.3 * PAGE_WIDTH), dpi=500, sharey=False, sharex=True
    )
    

    population = RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["D_a"].sum()
    n_weeks = len(RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["weeks"])
    acceptance_rate_delta = np.array(sorted(DELTAS_ABS))

    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    # plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(AGE_GROUPS_ABS)))

    for age_group, c in zip(AGE_GROUPS_ABS, colors):
        no_severe_cases = (
            population
            * n_weeks
            * np.array([RUNS_ABS[age_group][acc]["result"].sum() for acc in DELTAS_ABS])
        )
        normalisation = no_severe_cases[
            np.argwhere(acceptance_rate_delta == 0)
        ].flatten()
        axes[0].plot(
            100 * DELTAS_ABS,
            no_severe_cases - normalisation,
            "-o",
            label=f"{age_group}",
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )

        no_infections = []
        for acc in DELTAS_ABS:
            infection_dynamics_df = RUNS_ABS[age_group][acc]["infection_dynamics_df"]
            infection_dynamics_df = infection_dynamics_df[
                infection_dynamics_df["Age_group"] == "total"
            ]
            no_infections.append(
                infection_dynamics_df["total_infections_scenario"].sum()
            )
        no_infections = np.array(no_infections)

        normalisation = no_infections[np.argwhere(DELTAS_ABS == 0)].flatten()
        axes[1].plot(
            100 * DELTAS_ABS,
            no_infections - normalisation,
            "-o",
            label=f"{age_group}",
            c=c,
            lw=LW_GLOBAL,
            ms=MS_GLOBAL,
        )

    axes[0].set_title("Severe Cases")
    axes[0].set_xlabel("Change in acceptance rate (ppt equivalent)")
    axes[0].set_ylabel("Change in no. of severe cases")

    axes[1].set_title("Infections")
    axes[1].set_xlabel("Change in acceptance rate (ppt equivalent)")
    axes[1].set_ylabel("Change in no. of infections")
    axes[1].legend(
#         loc="center left",
#         bbox_to_anchor=(1, 0.5),
        title="age groups",
    )

    plt.tight_layout()
#     if save:
#         plt.savefig(OUTPUT_DIR / "acc_rate_exp_abs_overview", dpi=500)
    plt.show()


if ABS:
    plot(save=SAVE_PLOTS)


# + code_folding=[]
def plot(save=False):
    fig, axes = plt.subplots(
        1,
        2,
#         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 0.5 * SINGLE_COLUMN),
        dpi=500,
        sharey=False,
        sharex=True,
        tight_layout=True,
    )


    population = RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["D_a"].sum()
    n_weeks = len(RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["weeks"])
    acceptance_rate_delta = np.array(sorted(DELTAS_ABS))

    DELTA_PLOT = 0.006

    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    # plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(AGE_GROUPS_ABS)))

    severe_cases_diff_per_ag_scenario = {}
    infections_diff_per_ag_scenario = {}

    for age_group, c in zip(AGE_GROUPS_ABS, colors):
        no_severe_cases = (
            population * n_weeks * RUNS_ABS[age_group][DELTA_PLOT]["result"].sum()
        )
        normalisation = population * n_weeks * RUNS_ABS[age_group][0]["result"].sum()
        severe_cases_diff_per_ag_scenario[f"{age_group}"] = (
            - 100 * (no_severe_cases - normalisation) / normalisation
        )

        infection_dynamics_df = RUNS_ABS[age_group][DELTA_PLOT]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections = infection_dynamics_df["total_infections_scenario"].sum()
        infection_dynamics_df = RUNS_ABS[age_group][0]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        normalisation = infection_dynamics_df["total_infections_scenario"].sum()
        infections_diff_per_ag_scenario[f"{age_group}"] = (
            - 100 * (no_infections - normalisation) / normalisation
        )
    
    df_infections = pd.DataFrame.from_dict(
        infections_diff_per_ag_scenario, orient="index", columns=["delta I"]
    ).rename_axis("scenario")
    df_infections.plot.bar(ax=axes[0], legend=False, color="black")
    
    df_severe = pd.DataFrame.from_dict(
        severe_cases_diff_per_ag_scenario, orient="index", columns=["delta S"]
    ).rename_axis("scenario")
    df_severe.plot.bar(ax=axes[1], legend=False, color="black")
    
    axes[0].set_title("Impact on infections")
    axes[0].set_ylabel("Reduction in\nentire population (%)")
    axes[0].set_xlabel("Age group with UR increase")
    axes[0].tick_params(axis="x", labelrotation=45)


    axes[1].set_title("Impact on severe cases")
#     axes[1].set_ylabel("Reduction in\nentire population (%)")
    axes[1].set_xlabel("Age group with UR increase")
    axes[1].tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(OUTPUT_DIR / f"acc_rate_exp_abs_overview{OUTPUT_EXTENSION}.pdf")
    plt.show()

if ABS:
    plot(save=SAVE_PLOTS)

# + code_folding=[3]
SPLIT_DATE_WAVES_0 = np.datetime64(datetime.date(year=2021, month=5, day=30))


def plot(save=False):
    fig, axes = plt.subplots(
        1,
        2,
        #         figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        figsize=(SINGLE_COLUMN, 0.5 * SINGLE_COLUMN),
        dpi=500,
        sharey=False,
        sharex=True,
        tight_layout=True,
    )

    population = RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["D_a"].sum()
    n_weeks = len(RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["weeks"])
    acceptance_rate_delta = np.array(sorted(DELTAS_ABS))

#     DELTA_PLOT = 0.009899999999999999
#     DELTA_PLOT = 0.01
    DELTA_PLOT = 0.006


    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    # plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(AGE_GROUPS_ABS)))

    severe_cases_diff_per_ag_scenario = {}
    severe_cases_diff_per_ag_scenario_high = {}
    severe_cases_diff_per_ag_scenario_low = {}
    infections_diff_per_ag_scenario = {}
    infections_diff_per_ag_scenario_high = {}
    infections_diff_per_ag_scenario_low = {}

    for age_group, c in zip(AGE_GROUPS_ABS, colors):
        run_scenario = RUNS_ABS[age_group][DELTA_PLOT]
        run_baseline = RUNS_ABS[age_group][0]

        sev_total_samples_scenario = np.array(
            [
                severe_case_incidence(
                    res,
                    n_weeks=len(run_scenario["weeks"]),
                    week_dates=run_scenario["week_dates"],
                    #                     split_date_to=SPLIT_DATE_WAVES_0,
                )
                for res in run_scenario["result_samples"]
            ]
        )
        sev_total_samples_baseline = np.array(
            [
                severe_case_incidence(
                    res,
                    n_weeks=len(run_baseline["weeks"]),
                    week_dates=run_baseline["week_dates"],
                    #                     split_date_to=SPLIT_DATE_WAVES_0,
                )
                for res in run_baseline["result_samples"]
            ]
        )
        sev_diff_samples = (
            -100
            * (sev_total_samples_scenario - sev_total_samples_baseline)
            / sev_total_samples_baseline
        )
        severe_cases_diff_per_ag_scenario[f"{age_group}"] = sev_diff_samples.mean()
        severe_cases_diff_per_ag_scenario_high[f"{age_group}"] = (
            np.percentile(sev_diff_samples, ERROR_PERCENTILE_HIGH) - sev_diff_samples.mean()
        )
        severe_cases_diff_per_ag_scenario_low[
            f"{age_group}"
        ] = sev_diff_samples.mean() - np.percentile(sev_diff_samples, ERROR_PERCENTILE_LOW)
        
        print(
            np.percentile(sev_diff_samples, ERROR_PERCENTILE_LOW),
            sev_diff_samples.mean(),
            np.percentile(sev_diff_samples, ERROR_PERCENTILE_HIGH),
        )

        inf_total_samples_scenario = np.array(
            [
                infection_incidence(
                    infection_dynamics_sample=inf_sample,
                    population=run_scenario["D_a"].sum(),
                    week_dates=run_scenario["week_dates"],
                )
                for inf_sample in run_scenario["infection_dynamics_samples"]
            ]
        )
        inf_total_samples_baseline = np.array(
            [
                infection_incidence(
                    infection_dynamics_sample=inf_sample,
                    population=run_baseline["D_a"].sum(),
                    week_dates=run_baseline["week_dates"],
                )
                for inf_sample in run_baseline["infection_dynamics_samples"]
            ]
        )
        inf_diff_samples = (
            -100
            * (inf_total_samples_scenario - inf_total_samples_baseline)
            / inf_total_samples_baseline
        )
        infections_diff_per_ag_scenario[f"{age_group}"] = inf_diff_samples.mean()
        infections_diff_per_ag_scenario_high[f"{age_group}"] = (
            np.percentile(inf_diff_samples, ERROR_PERCENTILE_HIGH) - inf_diff_samples.mean()
        )
        infections_diff_per_ag_scenario_low[
            f"{age_group}"
        ] = inf_diff_samples.mean() - np.percentile(inf_diff_samples, ERROR_PERCENTILE_LOW)
        print(
            np.percentile(inf_diff_samples, ERROR_PERCENTILE_LOW),
            inf_diff_samples.mean(),
            np.percentile(inf_diff_samples, ERROR_PERCENTILE_HIGH),
        )

    df_infections = pd.DataFrame.from_dict(
        infections_diff_per_ag_scenario, orient="index", columns=["delta I"]
    ).rename_axis("scenario")
    df_infections_high = pd.DataFrame.from_dict(
        infections_diff_per_ag_scenario_high, orient="index", columns=["err_high"]
    ).rename_axis("scenario")
    df_infections_low = pd.DataFrame.from_dict(
        infections_diff_per_ag_scenario_low, orient="index", columns=["err_low"]
    ).rename_axis("scenario")

    df_infections.plot.bar(
        ax=axes[0],
        legend=False,
        color="gray",
        yerr=np.stack(
            (df_infections_low["err_low"].values, df_infections_high["err_high"].values)
        ),
    )

    df_severe = pd.DataFrame.from_dict(
        severe_cases_diff_per_ag_scenario, orient="index", columns=["delta S"]
    ).rename_axis("scenario")
    df_severe_high = pd.DataFrame.from_dict(
        severe_cases_diff_per_ag_scenario_high, orient="index", columns=["err_high"]
    ).rename_axis("scenario")
    df_severe_low = pd.DataFrame.from_dict(
        severe_cases_diff_per_ag_scenario_low, orient="index", columns=["err_low"]
    ).rename_axis("scenario")
    df_severe.plot.bar(
        ax=axes[1],
        legend=False,
        color="gray",
        yerr=np.stack(
            (df_severe_low["err_low"].values, df_severe_high["err_high"].values)
        ),
    )

    axes[0].set_title("Impact on infections")
    axes[0].set_ylabel("Reduction in\nentire population (%)")
    axes[0].set_xlabel("Age group with UR increase")
    axes[0].tick_params(axis="x", labelrotation=45)

    axes[1].set_title("Impact on severe cases")
    #     axes[1].set_ylabel("Reduction in\nentire population (%)")
    axes[1].set_xlabel("Age group with UR increase")
    axes[1].tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(OUTPUT_DIR / f"acc_rate_exp_abs_overview2{OUTPUT_EXTENSION}.pdf")
    plt.show()


if ABS:
    plot(save=SAVE_PLOTS)


# + code_folding=[0]
def plot(save=False):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=False,
    )


    population = RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["D_a"].sum()
    n_weeks = len(RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["weeks"])
    acceptance_rate_delta = np.array(sorted(DELTAS_ABS))

    DELTA_PLOT = 0.01

    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    # plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(AGE_GROUPS_ABS)))

    severe_cases_diff_per_ag_scenario = {}
    infections_diff_per_ag_scenario = {}

    for age_group, c in zip(AGE_GROUPS_ABS, colors):
        no_severe_cases = (
            population * n_weeks * RUNS_ABS[age_group][DELTA_PLOT]["result"].sum()
        )
        normalisation = population * n_weeks * RUNS_ABS[age_group][0]["result"].sum()
        severe_cases_diff_per_ag_scenario[f"{age_group}"] = (
            - 100 * (no_severe_cases - normalisation) / normalisation
        )

        infection_dynamics_df = RUNS_ABS[age_group][DELTA_PLOT]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections = infection_dynamics_df["total_infections_scenario"].sum()
        infection_dynamics_df = RUNS_ABS[age_group][0]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        normalisation = infection_dynamics_df["total_infections_scenario"].sum()
        infections_diff_per_ag_scenario[f"{age_group}"] = (
            - 100 * (no_infections - normalisation) / normalisation
        )

    df_severe = pd.DataFrame.from_dict(
        severe_cases_diff_per_ag_scenario, orient="index", columns=["delta S"]
    ).rename_axis("scenario")
    df_severe.plot.bar(ax=axes[0], legend=False)

    df_infections = pd.DataFrame.from_dict(
        infections_diff_per_ag_scenario, orient="index", columns=["delta I"]
    ).rename_axis("scenario")
    df_infections.plot.bar(ax=axes[1], legend=False)


    axes[0].set_title("Severe Cases")
    axes[0].set_ylabel("Reduction in no. of severe cases\n in entire population (%)")
    axes[0].set_xlabel("Age group with AR increase")
    axes[0].tick_params(axis="x", labelrotation=0)

    axes[1].set_title("Infections")
    axes[1].set_ylabel("Reduction in no. of infections\n in entire population (%)")
    axes[1].set_xlabel("Age group with AR increase")
    axes[1].tick_params(axis="x", labelrotation=0)
    
    run = RUNS_ABS['0-19'][0.0]
    infection_dynamics_df = run["infection_dynamics_df"]
    infection_dynamics_df = infection_dynamics_df[
        infection_dynamics_df["Age_group"] != "total"
    ]
    infection_array = infection_dynamics_df.sort_values(["Sunday_date", "Age_group"])[
        ["total_infections_observed"]
    ].values.reshape((-1, 9))
    weight_inf = infection_array[:-1, :]
    weighted_inf_avg_base_R_t = (run["median_weekly_base_R_t"][:-1, :] * weight_inf).sum(
        axis=0
    ) / weight_inf.sum(axis=0)
    print(f"weighted_inf_avg_base_R_t: {weighted_inf_avg_base_R_t}")
    print(f"weighted_inf_avg_base_R_t[-4:]: {weighted_inf_avg_base_R_t[-4:]}")
    weighted_inf_avg_base_R_t_60 = np.sum(weighted_inf_avg_base_R_t[-4:] * run['D_a'][-4:] / run['D_a'][-4:].sum())
    print(f"weighted_inf_avg_base_R_t_60: {weighted_inf_avg_base_R_t_60}")
    weighted_inf_avg_base_R_t_new = np.zeros_like(weighted_inf_avg_base_R_t[:-3])
    weighted_inf_avg_base_R_t_new[:-1] = weighted_inf_avg_base_R_t[:-4]
    weighted_inf_avg_base_R_t_new[-1] = weighted_inf_avg_base_R_t_60
    
    age_group_names_new = list(run["age_group_names"][:-4]) + ['60+']
    print(f"weighted_inf_avg_base_R_t_new: {weighted_inf_avg_base_R_t_new}")
    
    axes[2].set_title("Reproduction Number")
    pd.DataFrame(
        {
            "age_groups": age_group_names_new,
            "avg_R_base": weighted_inf_avg_base_R_t_new,
        }
    ).set_index("age_groups").plot.bar(ax=axes[2], legend=False)
    axes[2].set_ylabel("Infection weighted average R_base")
    axes[2].set_xlabel("Age group")
    axes[2].tick_params(axis="x", labelrotation=0)

    if save:
        plt.savefig(OUTPUT_DIR / "acc_rate_exp_abs_overview_2.pdf")
    plt.show()

if ABS:
    plot(save=SAVE_PLOTS)


# + code_folding=[0]
def plot(save=False):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(0.8 * PAGE_WIDTH, 0.25 * PAGE_WIDTH),
        dpi=500,
        sharey=False,
        sharex=True,
    )
    delta_doses = 3 * 10323.333333333334

    population = RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["D_a"].sum()
    n_weeks = len(RUNS_ABS[AGE_GROUPS_ABS[0]][min(DELTAS_ABS)]["weeks"])
    acceptance_rate_delta = np.array(sorted(DELTAS_ABS))

    DELTA_PLOT = 0.01

    plt.subplots_adjust(hspace=0.15, wspace=0.4)
    # plt.suptitle("Change in severity by age group (relative)")
    colors = AGE_COLORMAP(np.linspace(0, 1, len(AGE_GROUPS_ABS)))

    vacs_needed_to_save_severe_case = {}
    vacs_needed_to_save_infection = {}

    for age_group, c in zip(AGE_GROUPS_ABS, colors):
        no_severe_cases = (
            population * n_weeks * RUNS_ABS[age_group][DELTA_PLOT]["result"].sum()
        )
        normalisation = population * n_weeks * RUNS_ABS[age_group][0]["result"].sum()
        delta_severe_cases = -(no_severe_cases - normalisation)
        vacs_needed_to_save_severe_case[f"{age_group}"] = delta_doses / delta_severe_cases

        infection_dynamics_df = RUNS_ABS[age_group][DELTA_PLOT]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        no_infections = infection_dynamics_df["total_infections_scenario"].sum()
        infection_dynamics_df = RUNS_ABS[age_group][0]["infection_dynamics_df"]
        infection_dynamics_df = infection_dynamics_df[
            infection_dynamics_df["Age_group"] == "total"
        ]
        normalisation = infection_dynamics_df["total_infections_scenario"].sum()
        delta_infections = -(no_infections - normalisation)
        vacs_needed_to_save_infection[f"{age_group}"] = delta_doses / delta_infections

    df_severe = pd.DataFrame.from_dict(
        vacs_needed_to_save_severe_case, orient="index", columns=["delta S"]
    ).rename_axis("scenario")
    df_severe.plot.bar(ax=axes[0], legend=False)

    df_infections = pd.DataFrame.from_dict(
        vacs_needed_to_save_infection, orient="index", columns=["delta I"]
    ).rename_axis("scenario")
    df_infections.plot.bar(ax=axes[1], legend=False)


    axes[0].set_title("Severe Cases")
    axes[0].set_ylabel("No. additional vaccinations needed\n per prevented severe case")
    axes[0].set_xlabel("Age group with AR increase")
    axes[0].tick_params(axis="x", labelrotation=0)

    axes[1].set_title("Infections")
    axes[1].set_ylabel("No. additional vaccinations needed\n per prevented infection")
    axes[1].set_xlabel("Age group with AR increase")
    axes[1].tick_params(axis="x", labelrotation=0)

    if save:
        plt.savefig(OUTPUT_DIR / "acc_rate_exp_abs_overview_2", dpi=500)
    plt.show()

if ABS:
    plot(save=SAVE_PLOTS)

# + code_folding=[0]
if ABS:
    fig, axes = plt.subplots(2, 3, figsize=(3*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=False)
    ax_flat = axes.flatten()

    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle("Change in number of infections per age group")

    for age_group, ax in zip(AGE_GROUPS_ABS, ax_flat):
        infection_dynamics_df_max = RUNS_ABS[age_group][DELTAS_ABS.max()]['infection_dynamics_df']
        infection_dynamics_df_max = infection_dynamics_df_max.loc[infection_dynamics_df_max['Age_group'] != "total"]

        infection_dynamics_df_baseline = RUNS_ABS[age_group][0]['infection_dynamics_df']
        infection_dynamics_df_baseline = infection_dynamics_df_baseline.loc[infection_dynamics_df_baseline['Age_group'] != "total", :]

        infection_dynamics_df_max['diff'] = (
            infection_dynamics_df_max['total_infections_scenario'] - infection_dynamics_df_baseline['total_infections_scenario']
        )
        (
            100 * infection_dynamics_df_max.groupby('Age_group')['diff'].sum() 
            / infection_dynamics_df_baseline.groupby('Age_group')['total_infections_scenario'].sum()
        ).plot.bar(ax=ax, title=f"Increased acceptance rate for {age_group}")
        ax.grid()
        ax.set_ylabel("Change in cumulative no. of infections (%)")

    plt.tight_layout()
    plt.show()
# + code_folding=[0]
def plot(run):
    fig, axes = plt.subplots(
        1, 3, figsize=(3 * PLOT_WIDTH, 1 * PLOT_HEIGHT), sharey=True, sharex=False
    )
    plt.suptitle("Weighted average R_base")


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

plot(run=RUNS_ABS['0-19'][0.0])
# -


RUNS_ABS['0-19'][0.0]['D_a']

RUNS_ABS['0-19'][0.0].keys()


