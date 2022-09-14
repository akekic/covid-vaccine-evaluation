# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: causal-covid-analysis
#     language: python
#     name: causal-covid-analysis
# ---

# +
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import datetime
import itertools

from pathlib import Path

RERUN_DATA_PIPELINE = False

N_WANING_WEEKS = 104 # max. number of waning weeks to track, should be larger than total number of weeks
REFERENCE_AGE_GROUP = '60-69'
V1_eff = 70
V2_eff = 90
V3_eff = 95

INPUT_DATA_DIR = Path("../data/preprocessed-data")
OUTPUT_DATA_DIR = Path("../data")



pd.set_option('display.max_rows', 1000)

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4

PLOT = False

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# -

# ## Load Data

# +
# if RERUN_DATA_PIPELINE:
#     %run ./israel_data_processing.ipynb
# -

df = pd.read_pickle(INPUT_DATA_DIR / 'israel_df.pkl')
df_constants = pd.read_pickle(INPUT_DATA_DIR / 'israel_constants.pkl')
df.head()

df.columns

df_constants

# + code_folding=[4]
# vaccine efficacy curve
# <1, 1-<2, 2-<3, 3-<4, 4-<5, 5+ months
vaccine_efficacy_values_months = np.array([0.87, 0.85, 0.78, 0.67, 0.61, 0.50])

vaccine_efficacy_values_weeks = np.array(
    4 * [vaccine_efficacy_values_months[0]]
    + 4 * [vaccine_efficacy_values_months[1]]
    + 4 * [vaccine_efficacy_values_months[2]]
    + 4 * [vaccine_efficacy_values_months[3]]
    + 4 * [vaccine_efficacy_values_months[4]]
    + 1 * [vaccine_efficacy_values_months[5]]
)


def vaccine_efficacy_curve_naive(w):
    # TODO: 4 weeks ≠ 1 month
    return vaccine_efficacy_values_weeks[np.clip(w, a_min=0, a_max=len(vaccine_efficacy_values_weeks) - 1)]
#     return vaccine_efficacy_values_months[np.clip(w, a_min=0, a_max=len(vaccine_efficacy_values_months) - 1)]


# -

vaccine_efficacy_values_weeks

if PLOT:
    plt.figure()
    w = np.arange(0, 30, dtype=int)
    plt.plot(w, vaccine_efficacy_curve_naive(w))
    plt.xlabel('weeks since second dose')
    plt.ylabel('vaccine efficacy vs. infection')
    plt.title('vaccine efficacy waning against infections')
    plt.show()

if PLOT:
    plt.figure()
    w = np.arange(0, 30, dtype=int)
    plt.plot(
        w, 
        (1 - vaccine_efficacy_curve_naive(w)) / (1 - vaccine_efficacy_curve_naive(0))
    )
    plt.xlabel('weeks since second dose')
    plt.ylabel('waning factor h(w)')
    plt.title('waning curve')
    plt.show()

if PLOT:
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 1.5*4.8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    w = np.arange(0, 30, dtype=int)

    axes[0].plot(w, vaccine_efficacy_curve_naive(w))
    axes[0].set_ylabel('VE(w)')
    axes[0].set_title('vaccine efficacy')

    axes[1].plot(
        w, 
        (1 - vaccine_efficacy_curve_naive(w)) / (1 - vaccine_efficacy_curve_naive(0))
    )
    axes[1].set_xlabel('weeks since second dose')
    axes[1].set_ylabel('h(w)')
    axes[1].set_title('waning function')

    plt.show()


# +
# fit waning curve
x_max = 24
x_min = 0
y = np.array([vaccine_efficacy_values_months[0]] + list(vaccine_efficacy_values_months) + [0])
x = np.array([x_min] + list(np.arange(6) + 0.5) + [x_max])

y = np.array(list(vaccine_efficacy_values_months) + [0])
x = np.array(list(np.arange(6) + 0.5) + [x_max])

if PLOT: print(f"x: {x}, y: {y}")

def f_ve(x, a, b, c, d):
    return (a / (1. + np.exp(-c * (x - d)))) + b


popt, pcov = opt.curve_fit(f_ve, x, y, method="trf")
x_fit = np.linspace(x_min, x_max, num=100)
y_fit = f_ve(x_fit, *popt)

if PLOT: print(f"a = {popt[0]}, b = {popt[1]}, c = {popt[2]}, d = {popt[3]}")

if PLOT:
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, '.', label='data')
    plt.plot(x_fit, y_fit, '-', label='logistic fit')
    plt.ylabel("vaccine efficacy vs. infection")
    plt.xlabel("months since second shot")
    plt.grid()
    plt.legend()
    # plt.savefig('figs/waning_curve_fit.png', dpi=200, bbox_inches='tight')
    plt.show()

def vaccine_efficacy_curve_fit(w):
    week_to_month_factor = 365 / (12 * 7)
    m = w / week_to_month_factor
    return f_ve(m, *popt)


# -

if PLOT:
    plt.figure()
    # plt.plot(x, (1-y)/(1-y[0]), '.', label='data')
    plt.plot(x_fit, (1 - y_fit)/(1-y_fit[0]), '-', label='fit', c='#ff7f0e')
    plt.ylabel("waning factor")
    plt.xlabel("months since second shot")
    plt.grid()
    plt.legend()
    plt.show()

if PLOT:
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_HEIGHT), sharex=False)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.suptitle('Waning curve fit')

    df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axes[0].plot(x, y, '.', label='data')
    axes[0].plot(x_fit, y_fit, '-', label='logistic fit')
    axes[0].set_ylabel("vaccine efficacy")
    axes[0].set_xlabel("months since second shot")
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(x_fit, (1 - y_fit)/(1-y_fit[0]), '-', label='fit', c='#ff7f0e')
    axes[1].set_ylabel("waning factor")
    axes[1].set_xlabel("months since second shot")
    axes[1].grid()
    axes[1].legend()

    plt.show()

# +
# fit waning curve fast
x_max = 24
x_min = 0
y = np.array([vaccine_efficacy_values_months[0]] + list(vaccine_efficacy_values_months) + [0])
x = np.array([x_min] + list(np.arange(6) + 0.5) + [x_max])

y = np.array(list(vaccine_efficacy_values_months) + [0])
x = 0.75*np.array(list(np.arange(6) + 0.5) + [x_max])

if PLOT: print(f"x: {x}, y: {y}")

def f_ve(x, a, b, c, d):
    return (a / (1. + np.exp(-c * (x - d)))) + b


popt_fast, pcov_fast = opt.curve_fit(f_ve, x, y, method="trf")
x_fit = np.linspace(x_min, x_max, num=100)
y_fit = f_ve(x_fit, *popt_fast)

if PLOT: print(f"a = {popt_fast[0]}, b = {popt_fast[1]}, c = {popt_fast[2]}, d = {popt_fast[3]}")

if PLOT:
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, '.', label='data')
    plt.plot(x_fit, y_fit, '-', label='logistic fit')
    plt.ylabel("vaccine efficacy vs. infection")
    plt.xlabel("months since second shot")
    plt.grid()
    plt.legend()
    # plt.savefig('figs/waning_curve_fit_fast.png', dpi=200, bbox_inches='tight')
    plt.show()

def vaccine_efficacy_curve_fit_fast(w):
    week_to_month_factor = 365 / (12 * 7)
    m = w / week_to_month_factor
    return f_ve(m, *popt_fast)


# -

def vaccine_efficacy_curve_no_waning(w):
    return vaccine_efficacy_curve_fit(0)


# +
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# colors
# -

# ## Compute Waning Time Distribution

# + code_folding=[5, 17, 29]
def f_weeks(x, week=0):
    w_in_weeks = x[::-1] / x.sum()
    return w_in_weeks[week] if week < len(w_in_weeks) else 0

# waning times 1st dose
for i in range(N_WANING_WEEKS):
    df_tmp = df.groupby('Age_group')[['1st_dose']].expanding(axis=0).apply(
        f_weeks, raw=True, args=(i,)
    ).rename(columns={'1st_dose': f'w1_{i}'})
    if f'w1_{i}' in df.columns:
        df = df.drop(columns=[f'w1_{i}'])
    df = df.join(df_tmp.set_index(df_tmp.index.get_level_values(1)))

w1_cols = [f"w1_{i}" for i in range(N_WANING_WEEKS)]
df.loc[df[w1_cols].sum(axis=1) == 0, w1_cols] = np.nan

# waning times 2nd dose
for i in range(N_WANING_WEEKS):
    df_tmp = df.groupby('Age_group')[['2nd_dose']].expanding(axis=0).apply(
        f_weeks, raw=True, args=(i,)
    ).rename(columns={'2nd_dose': f'w2_{i}'})
    if f'w2_{i}' in df.columns:
        df = df.drop(columns=[f'w2_{i}'])
    df = df.join(df_tmp.set_index(df_tmp.index.get_level_values(1)))

w2_cols = [f"w2_{i}" for i in range(N_WANING_WEEKS)]
df.loc[df[w2_cols].sum(axis=1) == 0, w2_cols] = np.nan

# waning times 3rd dose
for i in range(N_WANING_WEEKS):
    df_tmp = df.groupby('Age_group')[['3rd_dose']].expanding(axis=0).apply(
        f_weeks, raw=True, args=(i,)
    ).rename(columns={'3rd_dose': f'w3_{i}'})
    if f'w3_{i}' in df.columns:
        df = df.drop(columns=[f'w3_{i}'])
    df = df.join(df_tmp.set_index(df_tmp.index.get_level_values(1)))

w3_cols = [f"w3_{i}" for i in range(N_WANING_WEEKS)]
df.loc[df[w3_cols].sum(axis=1) == 0, w3_cols] = np.nan

df.head()
# -
# ## Pipeline

# remove total age category
df = df[df['Age_group'] != 'total']
df.head()


# +
# input: df

# estimate g(0,a) -> df_g

# estimate g(1,a), g(3,a) -> df_g

# estimate g(2, a) -> df_g

# estimate f_(v,a)(t) -> df_f

# aggregate f_(v,a) (t) to f (t) by optimisation -> df_f

# evaluate reconstruction error

# + code_folding=[0, 7, 36, 70]
def estimate_g_unvaccinated(df):
    return pd.DataFrame((
        df.rename(columns={'hosp_unvaccinated_rel':'g'}).groupby('Age_group')['g'].sum()
        / df.groupby('Age_group')['hosp_unvaccinated_rel'].sum()[REFERENCE_AGE_GROUP]
    ).rename('g0'))


def estimate_g_after_1st_dose(df, df_g):
    results = {}
    for ag in df['Age_group'].unique():
        cols = ['Sunday_date', 'hosp_after_1st_dose_rel', 'waning_correction1']
        df_tmp = df[df['Age_group'] == ag][cols].set_index('Sunday_date')

        df_tmp_baseline = df[df['Age_group'] == REFERENCE_AGE_GROUP][
            ['Sunday_date', 'hosp_unvaccinated_rel']
        ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})
#         df_tmp_baseline = df[df['Age_group'] == ag][
#             ['Sunday_date', 'hosp_unvaccinated_rel']
#         ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})

        df_tmp = df_tmp.join(df_tmp_baseline)
        df_tmp = df_tmp[df_tmp['hosp_after_1st_dose_rel'].notnull()]
#         results[ag] = df_tmp['hosp_after_1st_dose_rel'].sum()/df_tmp['hosp_baseline_rel'].sum()
        # TODO: check
        results[ag] = (
            (
                df_g.loc[REFERENCE_AGE_GROUP, 'g0']
                * (
                    df_tmp['hosp_after_1st_dose_rel'] / df_tmp['waning_correction1']
                ).sum()
                / df_tmp['hosp_baseline_rel'].sum()
            )
        )
    return df_g.join(pd.Series(results).rename('g1'))


def estimate_g_after_2nd_dose(df, df_g):
    results = {}
    for ag in df['Age_group'].unique():
        df_tmp = df[df['Age_group'] == ag].set_index('Sunday_date')

        df_tmp_baseline = df[df['Age_group'] == ag][
            ['Sunday_date', 'hosp_unvaccinated_rel']
        ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})

        df_tmp = df_tmp.join(df_tmp_baseline)
        df_tmp = df_tmp[df_tmp['hosp_after_2nd_dose_rel'].notnull()]
        df_tmp = df_tmp[(df_tmp['build_up_factor1'] + df_tmp['build_up_factor2']) != 0]
        

        results[ag] = (
            (
                df_g.loc[ag, 'g0']
                * (
                    df_tmp['hosp_after_2nd_dose_rel'] 
                    / (
                        df_tmp['waning_correction2'] * (df_tmp['build_up_factor1'] + df_tmp['build_up_factor2'])
                    )
                ).sum()
                / df_tmp['hosp_baseline_rel'].sum()
            )
            - (
                (df_tmp['build_up_factor0'] + df_tmp['build_up_factor1']).sum()
                /(df_tmp['build_up_factor1'] + df_tmp['build_up_factor2']).sum()
            ) 
            * df_g.loc[ag, 'g1']
        )
    return df_g.join(pd.Series(results).rename('g2'))


def estimate_g_after_3rd_dose(df, df_g):
    results = {}
    for ag in df['Age_group'].unique():
        cols = ['Sunday_date', 'hosp_after_3rd_dose_rel', 'waning_correction3']
        df_tmp = df[df['Age_group'] == ag][cols].set_index('Sunday_date')

        df_tmp_baseline = df[df['Age_group'] == REFERENCE_AGE_GROUP][
            ['Sunday_date', 'hosp_unvaccinated_rel']
        ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})

        df_tmp = df_tmp.join(df_tmp_baseline)
        df_tmp = df_tmp[df_tmp['hosp_after_3rd_dose_rel'].notnull()]
#         results[ag] = df_tmp['hosp_after_3rd_dose_rel'].sum()/df_tmp['hosp_baseline_rel'].sum()
        # TODO: check
        results[ag] = (
            (
                df_g.loc[REFERENCE_AGE_GROUP, 'g0']
                * (
                    df_tmp['hosp_after_3rd_dose_rel'] / df_tmp['waning_correction3']
                ).sum()
                / df_tmp['hosp_baseline_rel'].sum()
            )
        )
    return df_g.join(pd.Series(results).rename('g3'))


# + code_folding=[0, 6, 12, 18]
delta_split_date = datetime.datetime(year=2021, month=7, day=1)

def estimate_g_unvaccinated_delta_split(df, delta_split_date=delta_split_date):
    df_g_delta0 = estimate_g_unvaccinated(df[df['Sunday_date'] <= delta_split_date])
    df_g_delta1 = estimate_g_unvaccinated(df[df['Sunday_date'] > delta_split_date])
    return df_g_delta0, df_g_delta1


def estimate_g_after_1st_dose_delta_split(df, df_g_delta0, df_g_delta1, delta_split_date=delta_split_date):
    df_g_delta0 = estimate_g_after_1st_dose(df[df['Sunday_date'] <= delta_split_date], df_g_delta0)
    df_g_delta1 = estimate_g_after_1st_dose(df[df['Sunday_date'] > delta_split_date], df_g_delta1)
    return df_g_delta0, df_g_delta1


def estimate_g_after_2nd_dose_delta_split(df, df_g_delta0, df_g_delta1, delta_split_date=delta_split_date):
    df_g_delta0 = estimate_g_after_2nd_dose(df[df['Sunday_date'] <= delta_split_date], df_g_delta0)
    df_g_delta1 = estimate_g_after_2nd_dose(df[df['Sunday_date'] > delta_split_date], df_g_delta1)
    return df_g_delta0, df_g_delta1


def estimate_g_after_3rd_dose_delta_split(df, df_g_delta0, df_g_delta1, delta_split_date=delta_split_date):
    df_g_delta0 = estimate_g_after_3rd_dose(df[df['Sunday_date'] <= delta_split_date], df_g_delta0)
    df_g_delta1 = estimate_g_after_3rd_dose(df[df['Sunday_date'] > delta_split_date], df_g_delta1)
    return df_g_delta0, df_g_delta1


# + code_folding=[0, 29, 63]
def estimate_g_after_1st_dose_cons(df, df_g):
    results = {}
    for ag in df['Age_group'].unique():
        cols = ['Sunday_date', 'hosp_after_1st_dose_rel', 'waning_correction1', 'hosp_unvaccinated_rel']
        df_tmp = df[df['Age_group'] == ag][cols].set_index('Sunday_date')

        df_tmp_baseline = df[df['Age_group'] == REFERENCE_AGE_GROUP][
            ['Sunday_date', 'hosp_unvaccinated_rel']
        ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})
#         df_tmp_baseline = df[df['Age_group'] == ag][
#             ['Sunday_date', 'hosp_unvaccinated_rel']
#         ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})

        df_tmp = df_tmp.join(df_tmp_baseline)
        df_tmp = df_tmp[df_tmp['hosp_after_1st_dose_rel'].notnull()]
#         results[ag] = df_tmp['hosp_after_1st_dose_rel'].sum()/df_tmp['hosp_baseline_rel'].sum()
        # TODO: check
        results[ag] = (
            (
                df_g.loc[REFERENCE_AGE_GROUP, 'g0']
                * (
                    df_tmp['hosp_after_1st_dose_rel'] / df_tmp['waning_correction1']
                ).sum()
                / df_tmp['hosp_baseline_rel'].sum()
            )
        )
    return df_g.join(pd.Series(results).rename('g1'))


def estimate_g_after_2nd_dose_cons(df, df_g):
    results = {}
    for ag in df['Age_group'].unique():
        df_tmp = df[df['Age_group'] == ag].set_index('Sunday_date')

        df_tmp_baseline = df[df['Age_group'] == ag][
            ['Sunday_date', 'hosp_unvaccinated_rel']
        ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})

        df_tmp = df_tmp.join(df_tmp_baseline)
        df_tmp = df_tmp[df_tmp['hosp_after_2nd_dose_rel'].notnull()]
        df_tmp = df_tmp[(df_tmp['build_up_factor1'] + df_tmp['build_up_factor2']) != 0]
        

        results[ag] = (
            (
                df_g.loc[ag, 'g0']
                * (
                    df_tmp['hosp_after_2nd_dose_rel'] 
                    / (
                        df_tmp['waning_correction2'] * (df_tmp['build_up_factor1'] + df_tmp['build_up_factor2'])
                    )
                ).sum()
                / df_tmp['hosp_unvaccinated_rel'].sum()
            )
            - (
                (df_tmp['build_up_factor0'] + df_tmp['build_up_factor1']).sum()
                /(df_tmp['build_up_factor1'] + df_tmp['build_up_factor2']).sum()
            ) 
            * df_g.loc[ag, 'g1']
        )
    return df_g.join(pd.Series(results).rename('g2'))


def estimate_g_after_3rd_dose_cons(df, df_g):
    results = {}
    for ag in df['Age_group'].unique():
        cols = ['Sunday_date', 'hosp_after_3rd_dose_rel', 'waning_correction3', 'hosp_unvaccinated_rel']
        df_tmp = df[df['Age_group'] == ag][cols].set_index('Sunday_date')

        df_tmp_baseline = df[df['Age_group'] == REFERENCE_AGE_GROUP][
            ['Sunday_date', 'hosp_unvaccinated_rel']
        ].set_index('Sunday_date').rename(columns={'hosp_unvaccinated_rel': 'hosp_baseline_rel'})

        df_tmp = df_tmp.join(df_tmp_baseline)
        df_tmp = df_tmp[df_tmp['hosp_after_3rd_dose_rel'].notnull()]
#         results[ag] = df_tmp['hosp_after_3rd_dose_rel'].sum()/df_tmp['hosp_baseline_rel'].sum()
        # TODO: check
        results[ag] = (
            (
                df_g.loc[REFERENCE_AGE_GROUP, 'g0']
                * (
                    df_tmp['hosp_after_3rd_dose_rel'] / df_tmp['waning_correction3']
                ).sum()
                / df_tmp['hosp_unvaccinated_rel'].sum()
            )
        )
    return df_g.join(pd.Series(results).rename('g3'))


# + code_folding=[0, 6, 11]
def estimate_f_unvaccinated(df):
    df_f = df.loc[:, ['Sunday_date', 'Age_group']]
    df_f['f0'] = df['hosp_unvaccinated_rel'] / df['g0']
    return df_f


def estimate_f_after_1st_dose(df, df_f):
    df_f['f1'] = df['hosp_after_1st_dose_rel'] / (df['g1'] * df['waning_correction1'])
    return df_f


def estimate_f_after_2nd_dose(df, df_f):
    df_f['f2'] = (
        df['hosp_after_2nd_dose_rel'] 
        / (
            df['g1'] * (df['build_up_factor0'] + df['build_up_factor1'])
            + df['g2'] * (df['waning_correction2'] - df['build_up_factor0'] - df['build_up_factor1'] )
        )
    )
    # TODO: here the waning correction with build up factor is overestimated due to double counting
    return df_f


def estimate_f_after_3rd_dose(df, df_f):
    df_f['f3'] = df['hosp_after_3rd_dose_rel'] / (df['g3'] * df['waning_correction3'])
    return df_f


# + code_folding=[45, 50, 88, 93]
# waning factor
def compute_waning_correction(
    df,
    waning_states=(2,),
    vaccine_efficacy_curve=vaccine_efficacy_curve_naive,
):
    if 1 in waning_states:
        waning_correction1 = 0
        expected_ve_0 = float(V1_eff) / 100
        s = expected_ve_0 / vaccine_efficacy_curve(0)
        for i in range(N_WANING_WEEKS):
            waning_correction1 += (
                df[f'w1_{i}'] 
                * (1 - s * vaccine_efficacy_curve(i))
                / (1 - s * vaccine_efficacy_curve(0))
            )
        waning_correction1 = waning_correction1.rename('waning_correction1').fillna(1)
    else:
        waning_correction1 = 1
    
    if 2 in waning_states:
        waning_correction2 = 0
        expected_ve_0 = float(V2_eff) / 100
        s = expected_ve_0 / vaccine_efficacy_curve(0)
        for i in range(N_WANING_WEEKS):
            waning_correction2 += (
                df[f'w2_{i}'] 
                * (1 - s * vaccine_efficacy_curve(i))
                / (1 - s * vaccine_efficacy_curve(0))
            )
        waning_correction2 = waning_correction2.rename('waning_correction2').fillna(1)
    else:
        waning_correction2 = 1
    
    if 3 in waning_states:
        waning_correction3 = 0
        expected_ve_0 = float(V3_eff) / 100
        s = expected_ve_0 / vaccine_efficacy_curve(0)
        for i in range(N_WANING_WEEKS):
            waning_correction3 += (
                df[f'w3_{i}'] 
                * (1 - s * vaccine_efficacy_curve(i))
                / (1 - s * vaccine_efficacy_curve(0))
            )
        waning_correction3 = waning_correction3.rename('waning_correction3').fillna(1)
    else:
        waning_correction3 = 1
    
    return (waning_correction1, waning_correction2, waning_correction3)


def compute_waning_correction_cons(
    df,
    waning_states=(2,),
    vaccine_efficacy_curve=vaccine_efficacy_curve_naive,
    alpha=None
):
    
    if 1 in waning_states:
        waning_correction1 = 0
        for i in range(N_WANING_WEEKS):
            al = 1 if alpha is None else df['Age_group'].map(alpha['g1'])
            waning_correction1 += (
                df[f'w1_{i}'] 
                * (1 - al * vaccine_efficacy_curve(i))
                / (1 - al * vaccine_efficacy_curve(0))
            )
        waning_correction1 = waning_correction1.rename('waning_correction1').fillna(1)
    else:
        waning_correction1 = 1
    
    if 2 in waning_states:
        waning_correction2 = 0
        for i in range(N_WANING_WEEKS):
            al = 1 if alpha is None else df['Age_group'].map(alpha['g2'])
            waning_correction2 += (
                df[f'w2_{i}'] 
                * (1 - al * vaccine_efficacy_curve(i))
                / (1 - al * vaccine_efficacy_curve(0))
            )
        waning_correction2 = waning_correction2.rename('waning_correction2').fillna(1)
    else:
        waning_correction2 = 1
    
    if 3 in waning_states:
        waning_correction3 = 0
        for i in range(N_WANING_WEEKS):
            al = 1 if alpha is None else df['Age_group'].map(alpha['g3'])
            waning_correction3 += (
                df[f'w3_{i}'] 
                * (1 - al * vaccine_efficacy_curve(i))
                / (1 - al * vaccine_efficacy_curve(0))
            )
        waning_correction3 = waning_correction3.rename('waning_correction3').fillna(1)
    else:
        waning_correction3 = 1
    
    return (waning_correction1, waning_correction2, waning_correction3)

def compute_build_up_correction(df, apply_build_up_correction=True):
    if apply_build_up_correction:
        build_up_factor0 = df['w2_0']
        build_up_factor1 = 0.5 * df['w2_1']
        build_up_factor2 = 0
        for i in range(2, N_WANING_WEEKS):
            build_up_factor2 += df[f'w2_{i}']
        return build_up_factor0, build_up_factor1, build_up_factor2
    else:
        build_up_factor0 = 0
        build_up_factor1 = 0
        build_up_factor2 = 1
        return build_up_factor0, build_up_factor1, build_up_factor2


# + code_folding=[0]
def compute_reconstruction_error(df):
    df_tmp = df.copy()
    
    hosp_rates = [
        'hosp_unvaccinated_rel', 
        'hosp_after_1st_dose_rel', 
        'hosp_after_2nd_dose_rel',
        'hosp_after_3rd_dose_rel',
    ]
    
    for i, hosp_rate in enumerate(hosp_rates):
        df_tmp[f'recon_error{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])**2
        df_tmp[f'recon_bias{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])

    n_time_steps = len(df_tmp['Sunday_date'].unique())

    df_tmp['weighting_factor0'] = df_tmp['unvaccinated_cum_rel'] * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor1'] = (
        df_tmp['1st_dose_cum_rel'] -  df_tmp['2nd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor2'] = (
        df_tmp['2nd_dose_cum_rel'] -  df_tmp['3rd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor3'] = df_tmp['3rd_dose_cum_rel'] * df_tmp['Population_share'] / n_time_steps
    
    for i, hosp_rate in enumerate(hosp_rates):
        df_tmp[f'recon_error_weighted{i}'] = (df_tmp[f'recon_error{i}'] * df_tmp[f'weighting_factor{i}'])
        df_tmp[f'recon_bias_weighted{i}'] = (df_tmp[f'recon_bias{i}'] * df_tmp[f'weighting_factor{i}'])
        df_tmp[f'normalisation_squared{i}'] = df_tmp[hosp_rate]**2 * df_tmp[f'weighting_factor{i}']
        df_tmp[f'normalisation_linear{i}'] = df_tmp[hosp_rate] * df_tmp[f'weighting_factor{i}']
    
    overall_error_nominator = (
        df_tmp['recon_error_weighted0']
        + df_tmp['recon_error_weighted1']
        + df_tmp['recon_error_weighted2']
        + df_tmp['recon_error_weighted3']
    ).agg('sum')
    
    overall_error_denominator = (
        df_tmp['normalisation_squared0']
        + df_tmp['normalisation_squared1']
        + df_tmp['normalisation_squared2']
        + df_tmp['normalisation_squared3']
    ).agg('sum')
    
    overall_bias_nominator = (
        df_tmp['recon_bias_weighted0']
        + df_tmp['recon_bias_weighted1']
        + df_tmp['recon_bias_weighted2']
        + df_tmp['recon_bias_weighted3']
    ).agg('sum')
    
    overall_bias_denominator = (
        df_tmp['normalisation_linear0']
        + df_tmp['normalisation_linear1']
        + df_tmp['normalisation_linear2']
        + df_tmp['normalisation_linear3']
    ).agg('sum')
    
    output = {}
    output['overall_error'] = overall_error_nominator/overall_error_denominator
    output['overall_bias'] = overall_bias_nominator/overall_bias_denominator
    
    for i in range(4):
        nominator = df_tmp[f'recon_error_weighted{i}'].agg('sum')
        denominator = df_tmp[f'normalisation_squared{i}'].agg('sum')
        output[f'error{i}'] = nominator/denominator
    
    for i in range(4):
        nominator = df_tmp[f'recon_bias_weighted{i}'].agg('sum')
        denominator = df_tmp[f'normalisation_linear{i}'].agg('sum')
        output[f'bias{i}'] = nominator/denominator
    
    return output


# + code_folding=[0]
def collect_f(df, df_f, how='unvaccinated'):
    if how == 'unvaccinated':
        df_f = df_f.merge(df_f.groupby(['Sunday_date'])['f0'].mean().rename('f'), on='Sunday_date')
        return df_f
    elif how == 'weighted_sum':
        cols = [
            'Sunday_date',
            'Age_group',
            'unvaccinated_cum_rel',
            '1st_dose_cum_rel',
            '2nd_dose_cum_rel',
            '3rd_dose_cum_rel',
            'Population_share',
        ]
        df_tmp = df.loc[:, cols]
        df_tmp['weight0'] = df_tmp['unvaccinated_cum_rel'] * df_tmp['Population_share']
        df_tmp['weight1'] = (df_tmp['1st_dose_cum_rel'] - df_tmp['2nd_dose_cum_rel']) * df_tmp['Population_share']
        df_tmp['weight2'] = (df_tmp['2nd_dose_cum_rel'] - df_tmp['3rd_dose_cum_rel']) * df_tmp['Population_share']
        df_tmp['weight3'] = df_tmp['3rd_dose_cum_rel'] * df_tmp['Population_share']
        
        df_f = df_f.merge(
            df_tmp[['Sunday_date', 'Age_group', 'weight0', 'weight1', 'weight2', 'weight3']],
            on=['Sunday_date', 'Age_group']
        )
        
        df_f['f_pre_aggregation'] = (
            df_f['weight0'].fillna(0) * df_f['f0'].fillna(0)
            + df_f['weight1'].fillna(0) * df_f['f1'].fillna(0)
            + df_f['weight2'].fillna(0) * df_f['f2'].fillna(0)
            + df_f['weight3'].fillna(0) * df_f['f3'].fillna(0)
        )
        
        df_f = df_f.merge(
            df_f[
                ['Sunday_date', 'f_pre_aggregation']
            ].groupby('Sunday_date').sum().rename(columns={'f_pre_aggregation':'f'}),
            on='Sunday_date'
        )
        
        df_f = df_f.drop(columns=['f_pre_aggregation'])
        return df_f
    else:
        raise ValueError(f"Invalid option for how: {how}. Valid opptions: 'unvaccinated', 'weighted_sum'.")


# + code_folding=[0]
def collect_output(df, df_f):
    
    # store f
    df = df.merge(df_f, on=['Sunday_date', 'Age_group'])
    
    # reconstruction
    df['recon0'] = df['f'] * df['g0']
    df['recon1'] = df['f'] * df['g1']
    df['recon2'] = df['f'] * (
        df['g1'] * (df['build_up_factor0'] + df['build_up_factor1'])
        + df['g2'] * df['build_up_factor1']
        + df['g2'] * df['waning_correction2']
    )
    df['recon3'] = df['f'] * df['g3']
    
    df['recon_norm0'] = df['f']
    df['recon_norm1'] = df['f']
    df['recon_norm2'] = df['f'] * (
        df['g1'] * (df['build_up_factor0'] + df['build_up_factor1'])
        + df['g2'] * df['build_up_factor1']
        + df['g2'] * df['waning_correction2']
    ) / df['g2']
    df['recon_norm3'] = df['f']
    
    return df


# + code_folding=[0]
def plot_residuals(df):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, sharex=True)
    
    cols = [
        'hosp_unvaccinated_rel',
        'hosp_after_1st_dose_rel',
        'hosp_after_2nd_dose_rel',
        'hosp_after_3rd_dose_rel',
    ]
    title = [
        'unvaccinated',
        'after_1st_dose',
        'after_2nd_dose',
        'after_3rd_dose',
    ]
    
    
    plt.subplots_adjust(hspace=0.2)
    plt.suptitle('Time dependence residuals')
    
    for j, (col, t, ax) in enumerate(zip(cols, title, axes.flatten())):
        df_avg = pd.DataFrame(index=df[df['Age_group'] == '0-19'].set_index('Sunday_date').index)
        df_avg['f'] = 0

        ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
        for ag in ages:
            df_tmp = df[df['Age_group'] == ag].set_index('Sunday_date')
            if col == 'hosp_after_2nd_dose_rel':
                waning_correction = df_tmp['waning_correction2']
            else:
                waning_correction = 1
            df_avg['f'] += (
                (1.0/len(df['Age_group'].unique()))
                * (df_tmp[col] / (df_tmp[f'g{j}'] * waning_correction) ).fillna(0)
            )
            ax.plot(
                df_tmp.index,
                df_tmp[col] / (df_tmp[f'g{j}'] * waning_correction),
                label=f"{ag}",
                alpha=0.3,
            )
        ax.plot(
            df_avg.index,
            df_avg['f'],
            label="avg"
        )
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(t)
        ax.set_ylabel('hospitalisation rate / g(V,A)')
        ax.grid()
        ax.set_ylim(top=0.015)
    ax.legend(loc='center left')
    plt.show()


# + code_folding=[0]
def plot_residual_means(df, title='Mean residuals'):
    fig= plt.figure()
    ax = plt.gca()
    
    cols = [
        'hosp_unvaccinated_rel',
        'hosp_after_1st_dose_rel',
        'hosp_after_2nd_dose_rel',
        'hosp_after_3rd_dose_rel',
    ]
    
    
    plt.subplots_adjust(hspace=0.2)
    plt.title(title)
    
    for j, col in enumerate(cols):
        df_avg = pd.DataFrame(index=df[df['Age_group'] == '0-19'].set_index('Sunday_date').index)
        df_avg['f'] = 0

        ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
        for ag in ages:
            df_tmp = df[df['Age_group'] == ag].set_index('Sunday_date')
            if col == 'hosp_after_2nd_dose_rel':
                waning_correction = df_tmp['waning_correction2']
            else:
                waning_correction = 1
            df_avg['f'] += (
                (1.0/len(df['Age_group'].unique()))
                * (df_tmp[col] / (df_tmp[f'g{j}'] * waning_correction) ).fillna(0)
            )
        ax.plot(
            df_avg.index,
            df_avg['f'],
            label=f"mean residuals {j}"
        )
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel('hospitalisation rate / g(V,A)')
        ax.grid()
#         ax.set_ylim(top=0.015)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid()
    plt.show()


# + code_folding=[0]
def plot_reconstruction(df):
    df_tmp = df.copy()
    
    hosp_rates = [
        'hosp_unvaccinated_rel', 
        'hosp_after_1st_dose_rel', 
        'hosp_after_2nd_dose_rel',
        'hosp_after_3rd_dose_rel',
    ]
    
    for i, hosp_rate in enumerate(hosp_rates):
        df_tmp[f'recon_error{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])**2
        df_tmp[f'recon_bias{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])

    n_time_steps = len(df_tmp['Sunday_date'].unique())

    df_tmp['weighting_factor0'] = df_tmp['unvaccinated_cum_rel'] * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor1'] = (
        df_tmp['1st_dose_cum_rel'] -  df_tmp['2nd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor2'] = (
        df_tmp['2nd_dose_cum_rel'] -  df_tmp['3rd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor3'] = df_tmp['3rd_dose_cum_rel'] * df_tmp['Population_share'] / n_time_steps
    
    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, hosp_rate in enumerate(hosp_rates):
        df_tmp[f'hosp_rate_norm{i}'] = df_tmp[hosp_rate] / df_tmp[f'g{i}']
        plt.plot(
            df_tmp.groupby(['Sunday_date'])[f"recon_norm{i}"].mean(),
            label=f"recon {i}",
            color=colors[i],
        )
        plt.plot(
            df_tmp.groupby(['Sunday_date'])[f'hosp_rate_norm{i}'].mean(),
            '--',
            label=f"hosp rate {i}",
            color=colors[i],
        )
        
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.tick_params(axis='x', labelrotation=45)
    plt.grid()
    plt.title('Normalised severity conditional reconstruction')
    plt.ylabel('P(S=1)/g(V,A)')
    plt.show()


# + code_folding=[0]
def plot_reconstruction2(df, title='Mean residuals'):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), sharey=True, sharex=True)
    
    cols = [
        'hosp_unvaccinated_rel',
        'hosp_after_1st_dose_rel',
        'hosp_after_2nd_dose_rel',
        'hosp_after_3rd_dose_rel',
    ]
    title = [
        'unvaccinated',
        'after_1st_dose',
        'after_2nd_dose',
        'after_3rd_dose',
    ]
    
    
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    plt.suptitle('Reconstruction vs. observed severity rate')
    
    for j, (col, t, ax) in enumerate(zip(cols, title, axes.flatten())):
        df_avg = pd.DataFrame(index=df[df['Age_group'] == '0-19'].set_index('Sunday_date').index)
        df_avg['f'] = 0

        ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
        for ag in ages:
            df_tmp = df[df['Age_group'] == ag].set_index('Sunday_date')
            df_avg['f'] += (
                (1.0/len(df['Age_group'].unique()))
                * (df_tmp[col] / (df_tmp[f'g{j}']) ).fillna(0)
            )
            ax.plot(
                df_tmp.index,
                df_tmp[col] / (df_tmp[f'g{j}']),
                label=f"{ag}",
                alpha=0.3,
            )
        ax.plot(
            df_tmp.index,
            df_tmp[f"recon_norm{j}"],
            label="normalised reconstruction",
            alpha=1.0,
            color='black',
        )
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_title(t)
        ax.set_ylabel('P(S=1|v, a, t) / g(v, a)')
        ax.grid()
#         ax.set_ylim(top=0.015)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

# plot_reconstruction2(df_wc)

# + code_folding=[0]
def plot_risk_factors(df_g):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 1.5*9.6), sharex=False)
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    plt.suptitle('Risk factors')
    
    df_g.plot.bar(ax=axes[0,0])
    axes[0,0].set_title('overview')
    
    df_g.plot.bar(ax=axes[0,1])
    axes[0,1].set_yscale('log')
    axes[0,1].set_title('overview (log)')
    
    df_g['g0'].plot.bar(ax=axes[1,0], color=colors[0])
    axes[1,0].set_title('unvaccinated')
    
    df_g['g1'].plot.bar(ax=axes[1,1], color=colors[1])
    axes[1,1].set_title('after 1st dose')
    
    df_g['g2'].plot.bar(ax=axes[2,0], color=colors[2])
    axes[2,0].set_title('after 2nd dose')
    
    df_g['g3'].plot.bar(ax=axes[2,1], color=colors[3])
    axes[2,1].set_title('after 3rd dose')
    
    plt.show()


# + code_folding=[]
def save_risk_factors(df_g, path):
    path.mkdir(parents=True, exist_ok=True)
    df_g.to_csv(path / "risk_factors.csv")


# + code_folding=[0]
def save_time_dependence(df_f, path):
    path.mkdir(parents=True, exist_ok=True)
    df_tmp = df_f[df_f['Age_group'] == '0-19'][['Sunday_date', 'f']]
    df_tmp.to_csv(path / "time_dependence.csv", index=False)


# + code_folding=[0]
def save_waning_curve(vaccine_efficacy_curve, path):
    path.mkdir(parents=True, exist_ok=True)
    
    waning_weeks = np.arange(104)
    waning_function = (1.0 - vaccine_efficacy_curve(waning_weeks)) / (1.0 - vaccine_efficacy_curve(0))

    pd.DataFrame({
        'week_since_last_dose': waning_weeks,
        'h': waning_function,
    }).to_csv(path / "waning_curve.csv", index=False)
    
    pd.DataFrame({
        'week_since_last_dose': waning_weeks,
        'vaccine_efficacy': vaccine_efficacy_curve(waning_weeks),
    }).to_csv(path / "vaccine_efficacy_waning_data.csv", index=False)


# + code_folding=[0]
def save_population_data(df, path):
    path.mkdir(parents=True, exist_ok=True)
    df_tmp = df[
        df['Sunday_date'] == df['Sunday_date'].unique()[0]
    ][
        ['Age_group', 'Population_size', 'Population_share']
    ]
    df_tmp.to_csv(path / "population_data.csv", index=False)


# + code_folding=[0]
def save_vaccination_data(df, path):
    path.mkdir(parents=True, exist_ok=True)
    df_tmp = df[['Sunday_date', 'Age_group', '1st_dose', '2nd_dose', '3rd_dose']]
    df_tmp.to_csv(path / "vaccination_data.csv", index=False)


# + code_folding=[0]
def save_observed_severity_data(df, path):
    path.mkdir(parents=True, exist_ok=True)
    df_tmp = df[[
        'Sunday_date',
        'Age_group',
        'hosp_unvaccinated_rel',
        'hosp_after_1st_dose_rel',
        'hosp_after_2nd_dose_rel',
       'hosp_after_3rd_dose_rel',
    ]]
    df_tmp.to_csv(path / "observed_severity_data.csv", index=False)


# + code_folding=[0]
def save_vaccine_acceptance_data(df, path):
    path.mkdir(parents=True, exist_ok=True)
    df_tmp = df[df['Sunday_date'] == df['Sunday_date'].max()][
        ['Age_group', '1st_dose_cum_rel', '2nd_dose_cum_rel', '3rd_dose_cum_rel']
    ].rename(columns={
        '1st_dose_cum_rel': '1st_dose_acceptance_rate',
        '2nd_dose_cum_rel': '2nd_dose_acceptance_rate',
        '3rd_dose_cum_rel': '3rd_dose_acceptance_rate',
    })
    df_tmp.to_csv(path / "vaccine_acceptance_data.csv", index=False)


# + code_folding=[0]
def save_observed_infection_data(df, path):
    path.mkdir(parents=True, exist_ok=True)
    infection_cols = [
        'positive_unvaccinated',
        'positive_after_1st_dose',
        'positive_after_2nd_dose',
        'positive_after_3rd_dose',
    ]
    df_tmp = df[[
        'Sunday_date',
        'Age_group',
        *infection_cols,
    ]]
    df_total = df_tmp.groupby('Sunday_date')[infection_cols].sum().reset_index()
    df_total['Age_group'] = 'total'
    df_tmp = df_tmp.append(df_total).sort_values(['Sunday_date', 'Age_group'])
    df_tmp.to_csv(path / "observed_infection_data.csv", index=False)


# -

# ## No Waning Correction

# +
# without waning correction
df_no_wc = df.copy()
(
    df_no_wc['waning_correction1'], 
    df_no_wc['waning_correction2'], 
    df_no_wc['waning_correction3'],
) = compute_waning_correction(df_no_wc, waning_states=[])
df_no_wc['build_up_factor0'] = compute_build_up_correction(df_no_wc, apply_build_up_correction=False)[0]
df_no_wc['build_up_factor1'] = compute_build_up_correction(df_no_wc, apply_build_up_correction=False)[1]
df_no_wc['build_up_factor2'] = compute_build_up_correction(df_no_wc, apply_build_up_correction=False)[2]

df_g_no_wc = estimate_g_unvaccinated(df_no_wc)
df_g_no_wc = estimate_g_after_1st_dose(df_no_wc, df_g_no_wc)
df_g_no_wc = estimate_g_after_2nd_dose(df_no_wc, df_g_no_wc)
df_g_no_wc = estimate_g_after_3rd_dose(df_no_wc, df_g_no_wc)

df_no_wc = df_no_wc.merge(df_g_no_wc, on=['Age_group'])

df_f_no_wc = estimate_f_unvaccinated(df_no_wc)
df_f_no_wc = estimate_f_after_1st_dose(df_no_wc, df_f_no_wc)
df_f_no_wc = estimate_f_after_2nd_dose(df_no_wc, df_f_no_wc)
df_f_no_wc = estimate_f_after_3rd_dose(df_no_wc, df_f_no_wc)
df_f_no_wc = collect_f(df_no_wc, df_f_no_wc)

df_no_wc = collect_output(df_no_wc, df_f_no_wc)

df_no_wc.head()
# -

if PLOT: plot_risk_factors(df_g_no_wc)

compute_reconstruction_error(df_no_wc)

if PLOT: plot_residuals(df_no_wc)

if PLOT: plot_residual_means(df_no_wc)

if PLOT: plot_reconstruction(df_no_wc)

if PLOT: plot_reconstruction2(df_no_wc)

# ## With Naive Waning Correction

# +
# with waning correction
df_wc = df.copy()
(
    df_wc['waning_correction1'], 
    df_wc['waning_correction2'], 
    df_wc['waning_correction3'],
) = compute_waning_correction(
    df_wc, waning_states=[2,], vaccine_efficacy_curve=vaccine_efficacy_curve_naive
)
df_wc['build_up_factor0'] = compute_build_up_correction(df_wc, apply_build_up_correction=False)[0]
df_wc['build_up_factor1'] = compute_build_up_correction(df_wc, apply_build_up_correction=False)[1]
df_wc['build_up_factor2'] = compute_build_up_correction(df_wc, apply_build_up_correction=False)[2]

df_g_wc = estimate_g_unvaccinated(df_wc)
df_g_wc = estimate_g_after_1st_dose(df_wc, df_g_wc)
df_g_wc = estimate_g_after_2nd_dose(df_wc, df_g_wc)
df_g_wc = estimate_g_after_3rd_dose(df_wc, df_g_wc)

df_wc = df_wc.merge(df_g_wc, on=['Age_group'])

df_f_wc = estimate_f_unvaccinated(df_wc)
df_f_wc = estimate_f_after_1st_dose(df_wc, df_f_wc)
df_f_wc = estimate_f_after_2nd_dose(df_wc, df_f_wc)
df_f_wc = estimate_f_after_3rd_dose(df_wc, df_f_wc)

df_f_wc = collect_f(df_wc, df_f_wc, how='weighted_sum')

df_wc = collect_output(df_wc, df_f_wc)

df_wc.head()
# -

metrics = compute_reconstruction_error(df_wc)
metrics

if PLOT: plot_risk_factors(df_g_wc)

if PLOT: plot_residuals(df_wc)

if PLOT: plot_residual_means(df_wc)

if PLOT: plot_reconstruction2(df_wc)

if PLOT: plot_reconstruction(df_wc)

dir_name = Path("initial-factorisation")
path = OUTPUT_DATA_DIR / dir_name
save_risk_factors(df_g_wc, path)
save_time_dependence(df_f_wc, path)
save_waning_curve(vaccine_efficacy_curve_naive, path)
save_population_data(df, path)
save_vaccination_data(df, path)
save_observed_severity_data(df, path)
save_observed_infection_data(df, path)
save_vaccine_acceptance_data(df, path)

df[['Sunday_date', 'Age_group', 'hosp_unvaccinated_rel',
       'hosp_after_1st_dose_rel', 'hosp_after_2nd_dose_rel',
       'hosp_after_3rd_dose_rel']]

# ## With Waning Correction from Fit

# +
# with waning correction
df_wc_fit = df.copy()
(
    df_wc_fit['waning_correction1'], 
    df_wc_fit['waning_correction2'], 
    df_wc_fit['waning_correction3'],
) = compute_waning_correction(
    df_wc_fit, waning_states=[1, 2, 3], vaccine_efficacy_curve=vaccine_efficacy_curve_fit
)
df_wc_fit['build_up_factor0'] = compute_build_up_correction(df_wc_fit, apply_build_up_correction=False)[0]
df_wc_fit['build_up_factor1'] = compute_build_up_correction(df_wc_fit, apply_build_up_correction=False)[1]
df_wc_fit['build_up_factor2'] = compute_build_up_correction(df_wc_fit, apply_build_up_correction=False)[2]

df_g_wc_fit = estimate_g_unvaccinated(df_wc_fit)
df_g_wc_fit = estimate_g_after_1st_dose(df_wc_fit, df_g_wc_fit)
df_g_wc_fit = estimate_g_after_2nd_dose(df_wc_fit, df_g_wc_fit)
df_g_wc_fit = estimate_g_after_3rd_dose(df_wc_fit, df_g_wc_fit)

df_wc_fit = df_wc_fit.merge(df_g_wc_fit, on=['Age_group'])

df_f_wc_fit = estimate_f_unvaccinated(df_wc_fit)
df_f_wc_fit = estimate_f_after_1st_dose(df_wc_fit, df_f_wc_fit)
df_f_wc_fit = estimate_f_after_2nd_dose(df_wc_fit, df_f_wc_fit)
df_f_wc_fit = estimate_f_after_3rd_dose(df_wc_fit, df_f_wc_fit)

df_f_wc_fit = collect_f(df_wc_fit, df_f_wc_fit, how='weighted_sum')

df_wc_fit = collect_output(df_wc_fit, df_f_wc_fit)

df_wc_fit.head()
# -

metrics = compute_reconstruction_error(df_wc_fit)
metrics

{'overall_error': 0.09235087866621548,
 'overall_bias': -0.05844519610341493,
 'error0': 0.07790002670439677,
 'error1': 0.395434333866452,
 'error2': 0.160572842270214,
 'error3': 0.2120981488483815,
 'bias0': 0.08430723251588951,
 'bias1': -0.24138181043159931,
 'bias2': -0.09501704708533106,
 'bias3': 0.02792538220921726}

if PLOT: plot_risk_factors(df_g_wc_fit)

if PLOT: plot_residuals(df_wc_fit)

if PLOT: plot_residual_means(df_wc_fit)

if PLOT: plot_reconstruction2(df_wc_fit)

if PLOT: plot_reconstruction(df_wc_fit)


# + code_folding=[0]
def plot():

    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_HEIGHT), sharex=False)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    plt.suptitle('Impact of waning curve fit on reconstruction')

    # df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    df_tmp = df_wc.copy()

    hosp_rates = [ 
        'hosp_after_2nd_dose_rel',
    ]

    for i, hosp_rate in zip([2], hosp_rates):
        df_tmp[f'recon_error{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])**2
        df_tmp[f'recon_bias{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])

    n_time_steps = len(df_tmp['Sunday_date'].unique())

    df_tmp['weighting_factor0'] = df_tmp['unvaccinated_cum_rel'] * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor1'] = (
        df_tmp['1st_dose_cum_rel'] -  df_tmp['2nd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor2'] = (
        df_tmp['2nd_dose_cum_rel'] -  df_tmp['3rd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor3'] = df_tmp['3rd_dose_cum_rel'] * df_tmp['Population_share'] / n_time_steps

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, hosp_rate in zip([2], hosp_rates):
        df_tmp[f'hosp_rate_norm{i}'] = df_tmp[hosp_rate] / df_tmp[f'g{i}']
        axes[0].plot(
            df_tmp.groupby(['Sunday_date'])[f"recon_norm{i}"].mean(),
            label=f"recon {i}",
            color=colors[i],
        )
        axes[0].plot(
            df_tmp.groupby(['Sunday_date'])[f'hosp_rate_norm{i}'].mean(),
            '--',
            label=f"hosp rate {i}",
            color=colors[i],
        )
        axes[0].tick_params(axis='x', labelrotation=45)
        axes[0].grid()
        axes[0].set_title('Naive waning curve')
        axes[0].set_ylabel('P(S=1)/g(V,A)')

    df_tmp = df_wc_fit.copy()

    hosp_rates = [ 
        'hosp_after_2nd_dose_rel',
    ]

    for i, hosp_rate in zip([2], hosp_rates):
        df_tmp[f'recon_error{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])**2
        df_tmp[f'recon_bias{i}'] = (df_tmp[f'recon{i}'] - df_tmp[hosp_rate])

    n_time_steps = len(df_tmp['Sunday_date'].unique())

    df_tmp['weighting_factor0'] = df_tmp['unvaccinated_cum_rel'] * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor1'] = (
        df_tmp['1st_dose_cum_rel'] -  df_tmp['2nd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor2'] = (
        df_tmp['2nd_dose_cum_rel'] -  df_tmp['3rd_dose_cum_rel']
    ) * df_tmp['Population_share'] / n_time_steps
    df_tmp['weighting_factor3'] = df_tmp['3rd_dose_cum_rel'] * df_tmp['Population_share'] / n_time_steps

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, hosp_rate in zip([2], hosp_rates):
        df_tmp[f'hosp_rate_norm{i}'] = df_tmp[hosp_rate] / df_tmp[f'g{i}']
        axes[1].plot(
            df_tmp.groupby(['Sunday_date'])[f"recon_norm{i}"].mean(),
            label=f"recon {i}",
            color=colors[i],
        )
        axes[1].plot(
            df_tmp.groupby(['Sunday_date'])[f'hosp_rate_norm{i}'].mean(),
            '--',
            label=f"hosp rate {i}",
            color=colors[i],
        )
        axes[1].tick_params(axis='x', labelrotation=45)
        axes[1].grid()
        axes[1].set_title('Waning curve fit')
        axes[1].set_ylabel('P(S=1)/g(V,A)')
    
#     plt.savefig('figs/reconstruction_and_waning_fit.png', dpi=200, bbox_inches='tight')
    plt.show()

if PLOT: plot()

# +
dir_name = Path("factorisation-with-fit")
path = OUTPUT_DATA_DIR / dir_name
save_risk_factors(df_g_wc_fit, path)
save_time_dependence(df_f_wc_fit, path)
save_waning_curve(vaccine_efficacy_curve_fit, path)
save_population_data(df, path)
save_vaccination_data(df, path)
save_observed_severity_data(df, path)
save_observed_infection_data(df, path)
save_vaccine_acceptance_data(df, path)

# path = Path("/Users/akekic/Repos/causal-covid-analysis/output/factorisation-with-fit-fast-waning")
dir_name = Path("factorisation-with-fit-fast-waning")
path = OUTPUT_DATA_DIR / dir_name
save_risk_factors(df_g_wc_fit, path)
save_time_dependence(df_f_wc_fit, path)
save_waning_curve(vaccine_efficacy_curve_fit_fast, path)
save_population_data(df, path)
save_vaccination_data(df, path)
save_observed_severity_data(df, path)
save_observed_infection_data(df, path)
save_vaccine_acceptance_data(df, path)

# path = Path("/Users/akekic/Repos/causal-covid-analysis/output/factorisation-with-fit-no-waning")
dir_name = Path("factorisation-with-fit-no-waning")
path = OUTPUT_DATA_DIR / dir_name
save_risk_factors(df_g_wc_fit, path)
save_time_dependence(df_f_wc_fit, path)
save_waning_curve(vaccine_efficacy_curve_no_waning, path)
save_population_data(df, path)
save_vaccination_data(df, path)
save_observed_severity_data(df, path)
save_observed_infection_data(df, path)
save_vaccine_acceptance_data(df, path)