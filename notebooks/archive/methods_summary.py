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

# # Imports

# +
import os
import import_ipynb
import datetime

import pandas as pd
import numpy as np
import scipy.optimize as opt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from matplotlib import cm
from mpl_toolkits.axisartist.axislines import AxesZero

from pathlib import Path
from pprint import pprint

from common_functions import (
    load_run,
    compute_weekly_second_doses_per_age,
    compute_weekly_third_doses_per_age,
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
    VAC_COLORMAP,
    START_FIRST_WAVE,
    END_FIRST_WAVE,
    START_SECOND_WAVE,
    END_SECOND_WAVE,
    POLICY_NAME_MAP,
)

# plt.style.use("fivethirtyeight")
# plt.rcParams.update(style_modifications)

# plt.style.use("ggplot")
# print(plt.rcParams)
plt.rcParams.update(style_modifications)
LOCAL_FONTSIZE = 6
local_style_modifications = {
    "font.size": LOCAL_FONTSIZE,
    "axes.titlesize": LOCAL_FONTSIZE,
    "axes.labelsize": LOCAL_FONTSIZE,
    "xtick.labelsize": LOCAL_FONTSIZE,
    "ytick.labelsize": LOCAL_FONTSIZE,
    "svg.fonttype": "none",
    "text.usetex": False,
    "font.serif": [],
}
plt.rcParams.update(local_style_modifications)

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

SAVE_PLOTS = True


EXP_NAME, EXP_DIR = "exp", Path("../run/2022-06-21_15-23-27.773744_policy_exp")
AGE_GROUPS_SUBSET = [0, 2, 5]
SMALL_PLOT_HEIGHT = 0.18
SMALL_PLOT_WIDTH = 0.2
LABELPAD = 1

# +
# print(plt.rcParams)
# -

RUN_DIRS = {
    x.name: x
    for x in EXP_DIR.iterdir() if x.is_dir()
}
RUNS = {
    x.name: load_run(x)
    for x in EXP_DIR.iterdir() if x.is_dir()
}
AGE_GROUP_NAMES_SUBSET = RUNS[list(RUNS.keys())[0]]["age_group_names"][AGE_GROUPS_SUBSET]


# # Vaccinations

# + code_folding=[0]
def plot_vaccination_policy(run, save, output_name):
    second_doses = compute_weekly_second_doses_per_age(run["U_2"])
    third_doses = compute_weekly_third_doses_per_age(run["U_2"], run["u_3"])

    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))

    fig = plt.figure(
        #         figsize=(1 * 0.3 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    #     plt.suptitle(f"Administered doses {run_name}")

    ax.stackplot(
        run["week_dates"],
        second_doses[AGE_GROUPS_SUBSET, :].cumsum(axis=1),
        labels=run["age_group_names"][AGE_GROUPS_SUBSET],
        alpha=0.8,
        colors=colors_age[AGE_GROUPS_SUBSET],
    )
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.set_ylabel("2nd doses\n(cum.)", labelpad=LABELPAD)
    ax.set_xlabel("Time", labelpad=LABELPAD)
    ax.set_xlim(run["week_dates"].min(), run["week_dates"].max())
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


plot_vaccination_policy(
    RUNS["observed"], save=SAVE_PLOTS, output_name="methods_summary_obs_vac.svg"
)
plot_vaccination_policy(
    RUNS["young_first"], save=SAVE_PLOTS, output_name="methods_summary_ctf_vac.svg"
)
# -

# # Severe cases

RUNS["observed"]["result"].shape


# + code_folding=[0]
def plot_severe_cases(run, save, output_name, y_max=None, stack=False):
    second_doses = compute_weekly_second_doses_per_age(run["U_2"])
    third_doses = compute_weekly_third_doses_per_age(run["U_2"], run["u_3"])

    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))

    fig = plt.figure(
        #         figsize=(1 * 0.3 * PAGE_WIDTH, 0.3 * PAGE_WIDTH),
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    #     plt.suptitle(f"Administered doses {run_name}")

    population = run["D_a"].sum()
    P_a = run["P_a"]
    n_weeks = len(run["weeks"])
    severe_case_incidence = n_weeks * run["result"].sum(axis=(2)) * 1e5 / P_a
    #     print(severe_case_incidence[:, AGE_GROUPS_SUBSET].cumsum(axis=0).T.sum(axis=0).max())
    
    if stack:
        ax.stackplot(
            run["week_dates"],
            severe_case_incidence[:, AGE_GROUPS_SUBSET].cumsum(axis=0).T,
            labels=run["age_group_names"][AGE_GROUPS_SUBSET],
            alpha=0.8,
            colors=colors_age[AGE_GROUPS_SUBSET],
        )
    else:
        for a in AGE_GROUPS_SUBSET:
            ax.plot(
                run["week_dates"],
                severe_case_incidence[:, a],
                label=run["age_group_names"][a],
                color=colors_age[a],
                lw=1,
            )
    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.set_ylabel("Severe cases\n(per 100k)", labelpad=LABELPAD)
    ax.set_xlabel("Time", labelpad=LABELPAD)
    ax.set_xlim(run["week_dates"].min(), run["week_dates"].max())
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


y_max = 120
plot_severe_cases(
    RUNS["observed"],
    save=SAVE_PLOTS,
    output_name="methods_summary_obs_severe.svg",
    y_max=y_max,
)
plot_severe_cases(
    RUNS["young_first"],
    save=SAVE_PLOTS,
    output_name="methods_summary_ctf_severe.svg",
    y_max=y_max,
)


# -

# # Infections

# + code_folding=[0, 59]
def plot_infections(run, save, output_name, y_max=None, stack=False):
    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))

    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()

    population = run["D_a"].sum()
    population_age_group = run["D_a"]
    n_weeks = len(run["weeks"])
    id_df = run["infection_dynamics_df"]

    id_df = id_df[id_df['Age_group'] != "total"][["Age_group", "Sunday_date", "total_infections_scenario"]]
    id_df = id_df.pivot(columns="Age_group", values="total_infections_scenario", index="Sunday_date")
    infection_incidence = id_df.values * 1e5 / population_age_group

    if stack:
        ax.stackplot(
            run["week_dates"],
            infection_incidence[:, AGE_GROUPS_SUBSET].cumsum(axis=0).T,
            labels=run["age_group_names"][AGE_GROUPS_SUBSET],
            alpha=0.8,
            colors=colors_age[AGE_GROUPS_SUBSET],
        )
    else:
        for a in AGE_GROUPS_SUBSET:
            ax.plot(
                run["week_dates"],
                infection_incidence[:, a],
                label=run["age_group_names"][a],
#                 alpha=0.8,
                color=colors_age[a],
                lw=1,
            )
            
    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.set_ylabel("Infections\n(per 100k)", labelpad=LABELPAD)
    ax.set_xlabel("Time", labelpad=LABELPAD)
    ax.set_xlim(run["week_dates"].min(), run["week_dates"].max())
    if y_max is not None:
        ax.set_ylim(top=y_max)
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


y_max = 1100
plot_infections(
    RUNS["observed"],
    save=SAVE_PLOTS,
    output_name="methods_summary_obs_infections.svg",
    y_max=y_max,
)
plot_infections(
    RUNS["young_first"],
    save=SAVE_PLOTS,
    output_name="methods_summary_ctf_infections.svg",
    y_max=y_max,
)


# -

# # Waning curve

# + code_folding=[0]
def plot_waning_curve(run, save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
    ve = run["vaccine_efficacy_params"]
    h = (1 - ve) / (1 - ve[0])
    ax.plot(np.arange(len(h)), h, lw=1, color="black")
#     ax.set_yticks([1, h[-1]], [1, r"$h_\mathrm{max}$"], usetex=True)
#     ax.axhline(y=h[-1], ls="--", c="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("Weeks Since Last Dose")
    ax.set_xlabel(r"$W$", usetex=True, labelpad=LABELPAD)
    ax.set_ylabel(r"$h(W)$", usetex=True, labelpad=LABELPAD)
#     ax.yaxis.set_label_coords(-0.05, 0.5)
    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


plot_waning_curve(
    RUNS["observed"],
    save=SAVE_PLOTS,
    output_name="methods_summary_waning_curve.svg",
)

# +
# vaccine efficacy curve
# <1, 1-<2, 2-<3, 3-<4, 4-<5, 5+ months
vaccine_efficacy_values_months = np.array([0.87, 0.85, 0.78, 0.67, 0.61, 0.50])

# fit waning curve
x_max = 24
x_min = 0
y = np.array([vaccine_efficacy_values_months[0]] + list(vaccine_efficacy_values_months) + [0])
x = np.array([x_min] + list(np.arange(6) + 0.5) + [x_max])

y = np.array(list(vaccine_efficacy_values_months) + [0])
x = np.array(list(np.arange(6) + 0.5) + [x_max])

print(f"x: {x}, y: {y}")

def f_ve(x, a, b, c, d):
    return (a / (1. + np.exp(-c * (x - d)))) + b


popt, pcov = opt.curve_fit(f_ve, x, y, method="trf")
x_fit = np.linspace(x_min, x_max, num=100)
y_fit = f_ve(x_fit, *popt)

print(f"a = {popt[0]}, b = {popt[1]}, c = {popt[2]}, d = {popt[3]}")


def plot_vaccine_efficacy(save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
#     plt.plot(x, y, '.', label='data')
    plt.plot(x_fit, y_fit, '-', label='logistic fit', color="black", lw=1)
    plt.ylabel("Vaccine\nefficacy", labelpad=LABELPAD, size="small")
    plt.xlabel("Time since dose", labelpad=LABELPAD, size="small")
#     plt.grid()
#     plt.legend()
#     ax.set_xlabel(r"$A$", usetex=True, labelpad=LABELPAD)

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([], minor=True)
    
#     ax.tick_params(axis="x", labelrotation=0, pad=LABELPAD, labelsize="small")
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()

plot_vaccine_efficacy(save=SAVE_PLOTS, output_name="methods_summary_vaccine_efficacy.svg")
# -

# # Risk factors

# +
df_g = pd.DataFrame(RUNS["observed"]["g"], columns=RUNS["observed"]["age_group_names"]).T
df_g = df_g[df_g.index.isin(AGE_GROUP_NAMES_SUBSET)]

df_g


# + code_folding=[]
def plot_risk_factors(run, save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
    df_g = df_g[df_g.index.isin(AGE_GROUP_NAMES_SUBSET)]
    colors_vac = VAC_COLORMAP(np.linspace(0.4, 1, len(run["vaccination_statuses"])))
    df_g.plot.bar(ax=ax, color=colors_vac, width=0.7, legend=False)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\log g(V, A)$", usetex=True, labelpad=LABELPAD)
#     ax.set_xlabel(r"$A$", usetex=True, labelpad=LABELPAD)

    
#     ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([], minor=True)
    
    ax.tick_params(axis="x", labelrotation=0, pad=LABELPAD, labelsize="small")
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


plot_risk_factors(
    RUNS["observed"],
    save=SAVE_PLOTS,
    output_name="methods_summary_risk_factors.svg",
)


# -

# # Time dependence

# + code_folding=[]
def plot_time_dependence(run, save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
    ax.plot(run["week_dates"], run["f_0"], color="black", lw=1)
    
    ax.set_ylabel(r"$f^0(T)$", usetex=True, labelpad=LABELPAD)
    ax.set_xlabel(r"$T$", usetex=True, labelpad=LABELPAD)

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


plot_time_dependence(
    RUNS["observed"],
    save=SAVE_PLOTS,
    output_name="methods_summary_time_dependence.svg",
)


# -

# # Correction factors

# + code_folding=[0]
def plot_correction_factor(run, save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        if age_group_name not in AGE_GROUP_NAMES_SUBSET:
            continue
        ax.plot(
            run["week_dates"],
            run["f_1"].T[:, i],
            label=age_group_name,
            color=colors_age[i],
            lw=1,
        )
    ax.axhline(y=1, ls="--", c="black", lw=0.5, alpha=0.5)
    
    ax.set_ylabel(r"$f^1(A, T)$", usetex=True, labelpad=LABELPAD)
    ax.set_xlabel(r"$T$", usetex=True, labelpad=LABELPAD)

    
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    if save:
        plt.savefig(OUTPUT_DIR / output_name)

    plt.show()


plot_correction_factor(
    RUNS["young_first"],
    save=SAVE_PLOTS,
    output_name="methods_summary_correction_factor.svg",
)


# -

# # Legends

# + code_folding=[0]
def plot_age_legend(run, save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        if age_group_name not in AGE_GROUP_NAMES_SUBSET:
            continue
        ax.plot(
            run["week_dates"],
            run["f_1"].T[:, i],
            label=age_group_name,
            color=colors_age[i],
            lw=3,
        )
    
    fig.show()
    
    figsize = (3, 3)
    fig_leg = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
    )
    ax_leg = fig_leg.add_subplot(111)
    
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', title="age groups")
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.show()
    
    if save:
        fig_leg.savefig(OUTPUT_DIR / output_name)


plot_age_legend(
    RUNS["young_first"],
    save=SAVE_PLOTS,
    output_name="methods_summary_age_legend.svg",
)


# + code_folding=[0]
def plot_vac_legend(run, save, output_name):
    fig = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
        tight_layout=False,
    )
    ax = fig.add_subplot()
    
    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
    df_g = df_g[df_g.index.isin(AGE_GROUP_NAMES_SUBSET)]
    colors_vac = VAC_COLORMAP(np.linspace(0.4, 1, len(run["vaccination_statuses"])))
    df_g.plot.bar(ax=ax, color=colors_vac, width=0.7, legend=False)

    fig.show()
    
    fig_leg = plt.figure(
        figsize=(SMALL_PLOT_WIDTH * SINGLE_COLUMN, SMALL_PLOT_HEIGHT * SINGLE_COLUMN),
        dpi=400,
    )
    ax_leg = fig_leg.add_subplot(111)
    
    # add the legend from the previous axes
    ax_leg.legend(*ax.get_legend_handles_labels(), loc='center', title="vaccination\n status")
    # hide the axes frame and the x/y labels
    ax_leg.axis('off')
    fig_leg.show()
    
    if save:
        fig_leg.savefig(OUTPUT_DIR / output_name)


plot_vac_legend(
    RUNS["observed"],
    save=SAVE_PLOTS,
    output_name="methods_summary_vac_legend.svg",
)
