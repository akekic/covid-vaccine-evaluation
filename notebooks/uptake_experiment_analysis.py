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

# # Vaccine uptake experiment

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

plt.rcParams.update(style_modifications)

SAVE_PLOTS = True
# -

# ## 2. Input
# The paths below need to be set to the respective local paths.

# +
# path to risk profile experiment output directory, i.e. output of ../experiments/vaccine_acceptance_exp.py
# OUTPUT_EXTENSION indicates which mixing factor C_mat_param was used
# C_mat_param=90
# OUTPUT_EXTENSION, EXP_DIR = "_C70", Path("../run/2022-08-19_15-46-36.455990_acc_exp")
# C_mat_param=80
OUTPUT_EXTENSION, EXP_DIR = "", Path("../run/2022-08-24_09-53-48.517966_acc_exp")
# C_mat_param=70
# OUTPUT_EXTENSION, EXP_DIR = "_C90", Path("../run/2022-08-19_15-46-57.553547_acc_exp")

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)
# -

# ## 3. Plots


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


# + code_folding=[0]
def plot(save=False):
    fig, axes = plt.subplots(
        1,
        2,
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
    axes[1].set_xlabel("Age group with UR increase")
    axes[1].tick_params(axis="x", labelrotation=45)

    if save:
        plt.savefig(OUTPUT_DIR / f"acc_rate_exp_abs_overview{OUTPUT_EXTENSION}.pdf")
    plt.show()

if ABS:
    plot(save=SAVE_PLOTS)


# + code_folding=[0]
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
