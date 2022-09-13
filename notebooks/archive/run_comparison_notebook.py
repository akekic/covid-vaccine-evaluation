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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools

from pathlib import Path
from itertools import product


PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# +
RUNS = {
    "observed policy": Path("run/2022-02-02_14-19-24_observed_vac_policy"),
    "uniform": Path("run/2022-02-02_14-28-30_uniform_vac_policy"),
#     "elderly first": Path("run/2022-02-02_14-30-43_old_to_young"),
#     "young first": Path("run/2022-02-02_14-33-43_young_to_old"),
}

DATA_DIR = Path("../causal-covid-analysis/data/israel/israel_df.pkl")
OUTPUT_DIR = Path("run/comparisons/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATA = {}
for name, run_dir in RUNS.items():
    data = {}
    data["result"] = np.load(run_dir / "result.npy")

    data["week_dates"] = np.load(run_dir/ "parameters" / "week_dates.npy")
    data["P_t"] = np.load(run_dir/ "parameters" / "P_t.npy")
    data["D_a"] = np.load(run_dir/ "parameters" / "D_a.npy")
    data["P_a"] = np.load(run_dir/ "parameters" / "P_a.npy")
    data["age_groups"] = np.load(run_dir/ "parameters" / "age_groups.npy")
    data["age_group_names"] = np.load(run_dir/ "parameters" / "age_group_names.npy")
    data["weeks"] = np.load(run_dir/ "parameters" / "weeks.npy")
    data["weeks_extended"] = np.load(run_dir/ "parameters" / "weeks_extended.npy")
    data["vaccination_statuses"] = np.load(run_dir/ "parameters" / "vaccination_statuses.npy")

    data["g"] = np.load(run_dir/ "parameters" / "severity_factorisation" / "g.npy")
    data["f"] = np.load(run_dir/ "parameters" / "severity_factorisation" / "f.npy")
    data["h_params"] = np.load(run_dir/ "parameters" / "severity_factorisation" / "h_params.npy")

    data["U_2"] = np.load(run_dir/ "vaccination_policy" / "U_2.npy")
    data["u_3"] = np.load(run_dir/ "vaccination_policy" / "u_3.npy")
    RUN_DATA[name] = data


# + code_folding=[]
def plot():
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_HEIGHT), sharey=False, sharex=True)

    plt.subplots_adjust(hspace=0.15, wspace=0.3)
    plt.suptitle('Hospitalisations overall')

    # df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for name, data in RUN_DATA.items():
        axes[0].plot(
            data['week_dates'],
            data['D_a'].sum() * data['result'].sum(axis=(1, 2))/data['P_t'], label=name)
    axes[0].set_title("Weekly")
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].set_ylabel("weekly hospitalisations")
    axes[0].legend()
    axes[0].grid()

    for name, data in RUN_DATA.items():
        axes[1].plot(
            data['week_dates'],
            (data['D_a'].sum() * data['result'].sum(axis=(1, 2))/data['P_t']).cumsum(), label=name)
    axes[1].set_title("Cumulative")
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].set_ylabel("cumulative hospitalisations")
    axes[1].legend()
    axes[1].grid()

    plt.savefig(
        OUTPUT_DIR / 'scenarios_hospitalisations_overall.png',
        dpi=200, 
        bbox_inches='tight',
    )

    plt.show()

plot()


# +
def plot():
    fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    ax_flat = axes.flatten()
    ages = list(RUN_DATA.items())[0][1]['age_group_names']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by age group (all vac. statuses)')
    
    for i, (ax, ag) in enumerate(zip(ax_flat, ages)):
        for name, data in RUN_DATA.items():
            ax.plot(
                data['week_dates'],
                data['D_a'].sum() * data['result'].sum(axis=2)[:, i]/data['P_t'], 
                label=name,
            )
        ax.set_title(f"{ag}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid()
    ax_flat[0].legend()

    
    plt.savefig(
        OUTPUT_DIR / 'scenarios_hospitalisations_by_age_group.png',
        dpi=200, 
        bbox_inches='tight',
    )
    plt.show()

plot()
# -

list(RUN_DATA.items())[0][1]['age_group_names']


