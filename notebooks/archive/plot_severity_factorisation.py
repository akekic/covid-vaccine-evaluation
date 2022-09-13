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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

import seaborn as sns

from pathlib import Path

from common_functions import load_run
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
    VAC_COLORMAP,
    START_FIRST_WAVE,
    END_FIRST_WAVE,
    START_SECOND_WAVE,
    END_SECOND_WAVE,
    POLICY_NAME_MAP,
)

# plt.style.use("ggplot")
plt.rcParams.update(style_modifications)

LW_GLOBAL = 1
MS_GLOBAL = 3

SAVE_PLOTS = True

OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# RUN_DIR = Path("../run/2022-04-07_13-26-59_acc_exp/delta_rate_0.02")
# RUN_DIR = Path("../run/2022-04-20_13-39-40_acc_exp/delta_rate_0.02")
# RUN_DIR = Path("../run/2022-05-26_09-35-53.867483_acc_exp/global/delta_rate_-0.02")
# RUN_DIR_BASELINE = Path("../run/2022-05-26_09-35-53.867483_acc_exp/global/delta_rate_0.0")

RUN_DIR = Path("../run/2022-06-21_12-21-00.135372_acc_exp/global/delta_rate_0.01")
RUN_DIR_BASELINE = Path("../run/2022-06-21_12-21-00.135372_acc_exp/global/delta_rate_0.0")
# -

# ! ls -la ../run/2022-05-26_09-35-53.867483_acc_exp/global/

run = load_run(RUN_DIR)
run_baseline = load_run(RUN_DIR_BASELINE)
run.keys()


# +
def plot(save=False):
    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T

    fig = plt.figure(figsize=(0.25 * 0.8 * PAGE_WIDTH, 0.25 * 0.8 * PAGE_WIDTH), dpi=300)
    ax = plt.gca()
    df_g.plot.bar(ax=ax)
    plt.legend(
        title="vaccination status",
        loc="lower right",
        bbox_to_anchor=(1.02, 1.0),
        ncol=4,
        fancybox=False,
        shadow=False,
        fontsize="small"
    )
    plt.yscale("log")
    ax.set_ylabel(r"$g(V, A)$", loc="top", rotation="horizontal", usetex=True)
    ax.yaxis.set_label_coords(-0.05, 0.9)
    if save:
        plt.savefig(OUTPUT_DIR / "g", dpi=500)
    plt.show()

plot(save=SAVE_PLOTS)


# + code_folding=[]
def plot(save=False):
    fig = plt.figure(figsize=(0.25 * 0.8 * PAGE_WIDTH, 0.25 * 0.8 * PAGE_WIDTH), dpi=300)
    plt.plot(run['week_dates'], run['f_0'])
    plt.gca().tick_params(axis='x', labelrotation=45)
    plt.gca().tick_params(
            axis="y",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
    #         left=False,
#             labelleft=False,
        )
    plt.gca().ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.ylabel(r"$f^0(T)$", usetex=True)
    if save:
        plt.savefig(OUTPUT_DIR / "f_0", dpi=500)
    plt.show()

plot(save=SAVE_PLOTS)


# +
def plot(save=False):
    fig = plt.figure(figsize=(0.25 * 0.8 * PAGE_WIDTH, 0.25 * 0.8 * PAGE_WIDTH), dpi=300)
    colors_age = AGE_COLORMAP(np.linspace(0, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        plt.plot(
            run["week_dates"], 
            run["f_1"].T[:, i], 
            label=age_group_name,
            color=colors_age[i],
            lw=LW_GLOBAL,
        )
    plt.axhline(y=1, ls="--", c="black", lw=LW_GLOBAL, alpha=0.5)
    plt.gca().tick_params(axis="x", labelrotation=45)
    # plt.legend(title="age group")
    plt.legend(
        title="age group",
        loc="lower right",
        bbox_to_anchor=(1.02, 1.0),
        ncol=3,
        fancybox=False,
        shadow=False,
        fontsize="small",
    )
    plt.ylabel(r"$f^1(A, T)$", usetex=True)
    if save:
        plt.savefig(OUTPUT_DIR / "f_1", dpi=500)
    plt.show()

plot(save=SAVE_PLOTS)

# +
# def plot(save=False):
#     fig = plt.figure(figsize=(0.25 * 0.8 * PAGE_WIDTH, 0.25 * 0.8 * PAGE_WIDTH), dpi=300)
#     plt.plot(np.arange(len(run['h_params'])), run['h_params'], lw=LW_GLOBAL)
#     plt.yticks([1, run['h_params'][-1]], [1, r'$\frac{1}{1{-}\mathrm{VE}(0)}$'], usetex=True)
#     plt.axhline(y=run['h_params'][-1], ls='--', c='black', lw=LW_GLOBAL, alpha=0.5)
#     plt.xlabel('weeks since last dose')
#     plt.xlabel(r"$W$", usetex=True)
#     plt.ylabel(r"$h(W)$", usetex=True)
#     if save:
#         plt.savefig(OUTPUT_DIR / "h", dpi=500)
#     plt.show()

# plot(save=SAVE_PLOTS)
# -

run.keys()


# + code_folding=[0]
def plot(save=False):
    fig = plt.figure(
#         figsize=(0.8 * PAGE_WIDTH, 0.4 * PAGE_WIDTH),
        figsize=(DOUBLE_COLUMN, 0.5 * DOUBLE_COLUMN),
        dpi=500,
        tight_layout=True,
    )
    ax0 = plt.subplot(2, 2, 1)
    ax1 = plt.subplot(2, 2, 2)
    ax2 = plt.subplot(2, 2, 3)
    ax3 = plt.subplot(2, 2, 4, sharex=ax1)
    axes = [ax0, ax1, ax2, ax3]
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    import matplotlib.transforms as mtransforms

    trans = mtransforms.ScaledTranslation(10 / 72, -10 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
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

    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
    colors_vac = VAC_COLORMAP(np.linspace(0.4, 1, len(run["vaccination_statuses"])))
    df_g.plot.bar(ax=axes[0], color=colors_vac)
    axes[0].legend(
        title="Vaccination Status",
        loc="lower right",
        bbox_to_anchor=(1.02, 1.0),
        ncol=4,
        fancybox=False,
        shadow=False,
        fontsize="small",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$g(V, A)$", usetex=True)
    # axes[0].yaxis.set_label_coords(-0.05, 0.9)
    axes[0].tick_params(axis="x", labelrotation=30)

    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        axes[3].plot(
            run["week_dates"],
            run["f_1"].T[:, i],
            label=age_group_name,
            color=colors_age[i],
            lw=LW_GLOBAL,
        )
    axes[3].axhline(y=1, ls="--", c="black", lw=LW_GLOBAL, alpha=0.5)
    axes[3].tick_params(axis="x", labelrotation=30)
    axes[3].legend(
        title="Age Group",
#         loc="lower right",
#         bbox_to_anchor=(1.0, 0.98),
        ncol=2,
        fancybox=False,
        shadow=False,
        fontsize="small",
    )
    axes[3].set_ylabel(r"$f^1(A, T)$", usetex=True)

    ve = run["vaccine_efficacy_params"]
    h = (1 - ve) / (1 - ve[0])
    axes[2].plot(np.arange(len(h)), h, lw=LW_GLOBAL, color="black")
    axes[2].set_yticks([1, h[-1]], [1, r"$h_\mathrm{max}$"], usetex=True)
    axes[2].axhline(y=h[-1], ls="--", c="black", lw=LW_GLOBAL, alpha=0.5)
    axes[2].set_xlabel("Weeks Since Last Dose")
    axes[2].set_xlabel(r"$W$", usetex=True)
    axes[2].set_ylabel(r"$h(W)$", usetex=True)
    axes[2].yaxis.set_label_coords(-0.05, 0.5)

    axes[1].plot(run["week_dates"], run["f_0"], color="black")

    y_arrow = run["f_0"].max() * 1.2
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0

    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), 0.6 * y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), 0.6 * y_arrow
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
    axes[1].add_patch(arrow)
    axes[1].text(x_tail + 0.5 * dx, y_head * 1.04, "3rd Wave", ha="center", va="bottom")

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
    axes[1].add_patch(arrow)
    axes[1].text(x_tail + 0.5 * dx, y_head * 1.02, "4th Wave", ha="center", va="bottom")

    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].tick_params(
        axis="y",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        right=False,
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].set_ylabel(r"$f^0(T)$", usetex=True)
    locator = mdates.AutoDateLocator()
    axes[1].xaxis.set_major_locator(locator)
    locator_minor = mdates.AutoDateLocator(maxticks=0)
    axes[1].xaxis.set_minor_locator(locator_minor)

    axes[1].set_ylim([0, run["f_0"].max() * 1.4])

    # df_inf = run['infection_dynamics_df']
    # df_inf = df_inf[df_inf['Age_group'] == 'total']
    # ax_twin = axes[3].twinx()
    # ax_twin.plot(run_baseline['week_dates'], run_baseline['result'].sum(axis=(1,2)), c="black", ls="--")
    # # ax_twin.plot(run_baseline['week_dates'], run_baseline['result'].sum(axis=(1))[:, 0], c="black", ls="--")
    # ax_twin.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if save:
        plt.savefig(OUTPUT_DIR / "severity_factorisation.pdf")
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[111, 127]
def plot(save=False):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(SINGLE_COLUMN, 0.7 * SINGLE_COLUMN),
        dpi=500,
        tight_layout=True,
        gridspec_kw={'width_ratios': [2, 3], "wspace": 0.5, "hspace": 0.6},
    )
    axes = axes.flatten()
#     axes[0].get_shared_x_axes().join(axes[0], axes[2])
    plt.subplots_adjust(hspace=0.4, wspace=-30.0)

    import matplotlib.transforms as mtransforms

    trans = mtransforms.ScaledTranslation(7 / 72, -7 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
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

    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
    colors_vac = VAC_COLORMAP(np.linspace(0.4, 1, len(run["vaccination_statuses"])))
    df_g.plot.bar(ax=axes[3], color=colors_vac, width=0.7)
    axes[3].legend(
        title="Vaccination status",
        loc="upper right",
        bbox_to_anchor=(1.02, -0.6),
        ncol=2,
        fancybox=False,
        shadow=False,
        fontsize="small",
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"$g(V, A)$", usetex=True)
    # axes[0].yaxis.set_label_coords(-0.05, 0.9)
    axes[3].tick_params(axis="x", labelrotation=45)
    axes[3].yaxis.set_major_formatter(mticker.ScalarFormatter())
    

    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        axes[2].plot(
            run["week_dates"],
            run["f_1"].T[:, i],
            label=age_group_name,
            color=colors_age[i],
            lw=LW_GLOBAL,
            ls="-",
        )
    axes[2].axhline(y=1, ls="--", c="black", lw=1, alpha=0.5)
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=5)
    axes[2].xaxis.set_major_locator(locator)
    locator_minor = mdates.AutoDateLocator(maxticks=0)
#     axes[2].xaxis.set_minor_locator(locator_minor)
    
    myFmt = mdates.DateFormatter('%b')
    axes[2].xaxis.set_major_formatter(myFmt)
    
    axes[2].tick_params(axis="x", labelrotation=0)
    axes[2].set_ylabel(r"$f^1_{\tilde \pi}(A, T)$", usetex=True)
    axes[2].set_xlabel(2021)
    
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.0),
        title="Age group",
        fontsize="small",
    )

    ve = run["vaccine_efficacy_params"]
    h = (1 - ve) / (1 - ve[0])
    axes[1].plot(np.arange(len(h)), h, lw=2, color="black", ls="-")
    axes[1].set_yticks([1, h[-1]], [1, r"$h^V_\mathrm{max}$"], usetex=True)
    axes[1].axhline(y=h[-1], ls="--", c="black", lw=LW_GLOBAL, alpha=0.5)
    axes[1].set_xlabel("Weeks since last dose")
#     axes[1].set_xlabel(r"$W$", usetex=True)
    axes[1].set_ylabel(r"$h^V(W)$", usetex=True)
    axes[1].yaxis.set_label_coords(-0.05, 0.5)

    axes[0].plot(run["week_dates"], run["f_0"], color="black", lw=2, ls="-")

    y_arrow = run["f_0"].max() * 1.1
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0
    arrow_fontsize = "small"

    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), 0.6 * y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), 0.6 * y_arrow
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
    axes[0].text(x_tail + 0.65 * dx, y_head * 1.13, "3rd Wave", ha="center", va="bottom", fontsize=arrow_fontsize)

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
    axes[0].text(x_tail + 0.5 * dx, y_head * 1.09, "4th Wave", ha="center", va="bottom", fontsize=arrow_fontsize)

    axes[0].tick_params(axis="x", labelrotation=0)
    axes[0].tick_params(
        axis="y",
        which="both",
#         bottom=False,
#         top=False,
        labelbottom=False,
        right=False,
    )
#     axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel(r"$f^0(T)$", usetex=True)

    axes[0].set_ylim([0, run["f_0"].max() * 1.45])
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_minor_locator(locator_minor)
    axes[0].set_xlim(right=run["week_dates"].max())
    axes[0].set_xticklabels([])

    # df_inf = run['infection_dynamics_df']
    # df_inf = df_inf[df_inf['Age_group'] == 'total']
    # ax_twin = axes[3].twinx()
    # ax_twin.plot(run_baseline['week_dates'], run_baseline['result'].sum(axis=(1,2)), c="black", ls="--")
    # # ax_twin.plot(run_baseline['week_dates'], run_baseline['result'].sum(axis=(1))[:, 0], c="black", ls="--")
    # ax_twin.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if save:
        plt.savefig(OUTPUT_DIR / "severity_factorisation2.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
# +
def plot(save=False):
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(TEXT_WIDTH_WORKSHOP, 0.175 * TEXT_WIDTH_WORKSHOP),
        dpi=500,
#         tight_layout=True,
#         constrained_layout=True,
        gridspec_kw={
            'width_ratios': [3, 3, 3, 4], 
            "wspace": 0.5, 
#             "hspace": 0.6,
        },
    )
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.4, wspace=-30.0)

    import matplotlib.transforms as mtransforms

    trans = mtransforms.ScaledTranslation(7 / 72, -7 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
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

    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
    colors_vac = VAC_COLORMAP(np.linspace(0.4, 1, len(run["vaccination_statuses"])))
    df_g.plot.bar(ax=axes[3], color=colors_vac, width=0.7, legend=False)
    h, l = axes[3].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.90, 1.05),
        title="Vaccination status",
        fontsize="small",
    )
    axes[3].set_yscale("log")
    axes[3].set_title(r"$g(V, A)$", usetex=True)
    # axes[0].yaxis.set_label_coords(-0.05, 0.9)
    axes[3].tick_params(axis="x", labelrotation=60)
    axes[3].yaxis.set_major_formatter(mticker.ScalarFormatter())
    

    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        axes[2].plot(
            run["week_dates"],
            run["f_1"].T[:, i],
            label=age_group_name,
            color=colors_age[i],
            lw=LW_GLOBAL,
            ls="-",
        )
    axes[2].axhline(y=1, ls="--", c="black", lw=1, alpha=0.5)
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=5)
    axes[2].xaxis.set_major_locator(locator)
    locator_minor = mdates.AutoDateLocator(maxticks=0)
#     axes[2].xaxis.set_minor_locator(locator_minor)
    
    myFmt = mdates.DateFormatter('%b')
    axes[2].xaxis.set_major_formatter(myFmt)
    
    axes[2].tick_params(axis="x", labelrotation=0)
    axes[2].set_title(r"$f^1_{\tilde \pi}(A, T)$", usetex=True)
    axes[2].set_xlabel(2021)
    
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=5,
        loc="lower left",
        bbox_to_anchor=(0.175, 1.05),
        title="Age group",
        fontsize="small",
    )
    

    ve = run["vaccine_efficacy_params"]
    h = (1 - ve) / (1 - ve[0])
    axes[1].plot(np.arange(len(h)), h, lw=2, color="black", ls="-")
    axes[1].set_yticks([1, h[-1]], [1, r"$h^V_\mathrm{max}$"], usetex=True)
    axes[1].axhline(y=h[-1], ls="--", c="black", lw=LW_GLOBAL, alpha=0.5)
    axes[1].set_xlabel("Weeks since last dose")
    axes[1].set_title(r"$h^V(W)$", usetex=True)
    axes[1].yaxis.set_label_coords(-0.05, 0.5)

    axes[0].plot(run["week_dates"], run["f_0"], color="black", lw=2, ls="-")

    y_arrow = run["f_0"].max() * 1.1
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0
    arrow_fontsize = "x-small"

    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), 0.6 * y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), 0.6 * y_arrow
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
    axes[0].text(x_tail + 0.65 * dx, y_head * 1.13, "3rd Wave", ha="center", va="bottom", fontsize=arrow_fontsize)

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
    axes[0].text(x_tail + 0.5 * dx, y_head * 1.09, "4th Wave", ha="center", va="bottom", fontsize=arrow_fontsize)

    axes[0].tick_params(axis="x", labelrotation=0)
#     axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_title(r"$f^0(T)$", usetex=True)

    axes[0].set_ylim([0, run["f_0"].max() * 1.45])
    locator = mdates.AutoDateLocator(minticks=5, maxticks=5)
    axes[0].xaxis.set_major_locator(locator)
    locator_minor = mdates.AutoDateLocator(maxticks=0)
#     axes[0].xaxis.set_minor_locator(locator_minor)
    axes[0].set_xlim(right=run["week_dates"].max())
    
    myFmt = mdates.DateFormatter('%b')
    axes[0].xaxis.set_major_formatter(myFmt)
    axes[0].set_xlabel(2021)

    if save:
        fig.savefig(OUTPUT_DIR / "severity_factorisation_w.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


plot(save=SAVE_PLOTS)


# + code_folding=[0]
def plot(save=False):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXT_WIDTH_WORKSHOP, 0.4 * TEXT_WIDTH_WORKSHOP),
        dpi=500,
        tight_layout=True,
        gridspec_kw={'width_ratios': [2, 3], "wspace": 0.5, "hspace": 0.8},
    )
    axes = axes.flatten()
#     axes[0].get_shared_x_axes().join(axes[0], axes[2])
    plt.subplots_adjust(hspace=0.4, wspace=-30.0)

    import matplotlib.transforms as mtransforms

    trans = mtransforms.ScaledTranslation(7 / 72, -7 / 72, fig.dpi_scale_trans)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
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

    df_g = pd.DataFrame(run["g"], columns=run["age_group_names"]).T
    colors_vac = VAC_COLORMAP(np.linspace(0.4, 1, len(run["vaccination_statuses"])))
    df_g.plot.bar(ax=axes[3], color=colors_vac, width=0.7)
    axes[3].legend(
        title="Vaccination status",
        loc="upper right",
        bbox_to_anchor=(1.02, -0.6),
        ncol=2,
        fancybox=False,
        shadow=False,
        fontsize="small",
    )
    axes[3].set_yscale("log")
    axes[3].set_ylabel(r"$g(V, A)$", usetex=True)
    # axes[0].yaxis.set_label_coords(-0.05, 0.9)
    axes[3].tick_params(axis="x", labelrotation=45)
    axes[3].yaxis.set_major_formatter(mticker.ScalarFormatter())
    

    colors_age = AGE_COLORMAP(np.linspace(0.2, 1, len(run["age_group_names"])))
    for i, age_group_name in enumerate(run["age_group_names"]):
        axes[2].plot(
            run["week_dates"],
            run["f_1"].T[:, i],
            label=age_group_name,
            color=colors_age[i],
            lw=LW_GLOBAL,
            ls="-",
        )
    axes[2].axhline(y=1, ls="--", c="black", lw=1, alpha=0.5)
    
    locator = mdates.AutoDateLocator(minticks=5, maxticks=5)
    axes[2].xaxis.set_major_locator(locator)
    locator_minor = mdates.AutoDateLocator(maxticks=0)
#     axes[2].xaxis.set_minor_locator(locator_minor)
    
    myFmt = mdates.DateFormatter('%b')
    axes[2].xaxis.set_major_formatter(myFmt)
    
    axes[2].tick_params(axis="x", labelrotation=0)
    axes[2].set_ylabel(r"$f^1(A, T)$", usetex=True)
    axes[2].set_xlabel(2021)
    
    h, l = axes[2].get_legend_handles_labels()
    fig.legend(
        handles=h,
        labels=l,
        ncol=5,
        loc="upper left",
        bbox_to_anchor=(0.1, -0.025),
        title="Age group",
        fontsize="small",
    )

    ve = run["vaccine_efficacy_params"]
    h = (1 - ve) / (1 - ve[0])
    axes[1].plot(np.arange(len(h)), h, lw=2, color="black", ls="-")
    axes[1].set_yticks([1, h[-1]], [1, r"$h_\mathrm{max}$"], usetex=True)
    axes[1].axhline(y=h[-1], ls="--", c="black", lw=LW_GLOBAL, alpha=0.5)
    axes[1].set_xlabel("Weeks since last dose")
#     axes[1].set_xlabel(r"$W$", usetex=True)
    axes[1].set_ylabel(r"$h^V(W)$", usetex=True)
    axes[1].yaxis.set_label_coords(-0.05, 0.5)

    axes[0].plot(run["week_dates"], run["f_0"], color="black", lw=2, ls="-")

    y_arrow = run["f_0"].max() * 1.1
    arrow_scale = 3
    arrow_color = "black"
    arrow_lw = 1
    arrow_shrink = 0
    arrow_fontsize = "small"

    x_tail, y_tail = mdates.date2num(START_FIRST_WAVE), 0.6 * y_arrow
    x_head, y_head = mdates.date2num(END_FIRST_WAVE), 0.6 * y_arrow
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
    axes[0].text(x_tail + 0.5 * dx, y_head * 1.05, "3rd Wave", ha="center", va="bottom", fontsize=arrow_fontsize)

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
    axes[0].text(x_tail + 0.5 * dx, y_head * 1.09, "4th Wave", ha="center", va="bottom", fontsize=arrow_fontsize)

    axes[0].tick_params(axis="x", labelrotation=0)
    axes[0].tick_params(
        axis="y",
        which="both",
#         bottom=False,
#         top=False,
        labelbottom=False,
        right=False,
    )
#     axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set_ylabel(r"$f^0(T)$", usetex=True)

    axes[0].set_ylim([0, run["f_0"].max() * 1.55])
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_minor_locator(locator_minor)
    axes[0].set_xlim(right=run["week_dates"].max())
    axes[0].set_xticklabels([])

    # df_inf = run['infection_dynamics_df']
    # df_inf = df_inf[df_inf['Age_group'] == 'total']
    # ax_twin = axes[3].twinx()
    # ax_twin.plot(run_baseline['week_dates'], run_baseline['result'].sum(axis=(1,2)), c="black", ls="--")
    # # ax_twin.plot(run_baseline['week_dates'], run_baseline['result'].sum(axis=(1))[:, 0], c="black", ls="--")
    # ax_twin.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    if save:
        plt.savefig(OUTPUT_DIR / "severity_factorisation_w2.pdf")
    plt.show()


plot(save=SAVE_PLOTS)
# -


