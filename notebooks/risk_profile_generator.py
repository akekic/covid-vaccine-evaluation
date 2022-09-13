# -*- coding: utf-8 -*-
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

# # Risk profile generator

# ## 1. Imports

# +
import os
import shutil

from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4

PLOT = False
# -

# ## 2. Input

FACTORISATION_PATH = Path("../data/factorisation-with-fit")
OUTPUT_PATH = Path("../data/risk-profile-experiment")


# ## 3. Uniform risk

def save_risk_factors(df_g, path):
    df_g.to_csv(path / "risk_factors.csv")


df_g = pd.read_csv(FACTORISATION_PATH / "risk_factors.csv").set_index('Age_group')
df_g

if PLOT:
    df_g.plot()
    plt.title("Observed Risk Profile")
    plt.ylabel("risk")
    plt.show()

if PLOT:
    plt.plot(1 - df_g['g1']/df_g['g0'], label="VE 1")
    plt.plot(1 - df_g['g2']/df_g['g0'], label="VE 2")
    plt.plot(1 - df_g['g3']/df_g['g0'], label="VE 3")
    plt.title("Vaccine Efficacy by Number of Doses")
    plt.ylabel("vaccine efficacy")
    plt.xlabel("age group")
    plt.legend()
    plt.show()

ratio1 = (df_g['g1']/df_g['g0']).mean()
ratio2 = (df_g['g2']/df_g['g0']).mean()
ratio3 = (df_g['g3']/df_g['g0']).mean()
print(f"Mean risk ratios: 1st dose: {ratio1},  2nd dose: {ratio2},  3rd dose: {ratio3}")

# +
# uniform risk
scaling_factor_uniform = 0.721330940852634 * 1.0225772435336615 * 0.9986823046905743
df_g_uniform = df_g.copy()
df_g_uniform['g0'] = scaling_factor_uniform * 1.0
df_g_uniform['g1'] = scaling_factor_uniform * ratio1 * df_g_uniform['g0']
df_g_uniform['g2'] = scaling_factor_uniform * ratio2 * df_g_uniform['g0']
df_g_uniform['g3'] = scaling_factor_uniform * ratio3 * df_g_uniform['g0']

df_g_uniform
# -

if PLOT:
    df_g_uniform.plot()
    plt.title("Uniform Risk Profile")
    plt.ylabel("risk")
    plt.show()

# ## 4. Spanish Flu

# +
# Spanish Flu
excess_mortality_rate = np.array([
    110,
    87, 67, 52, 38, 26, 21, 19, 18, 17, 17,
    17, 18, 19, 21, 26, 37, 42, 48, 56, 60,
    70, 75, 77, 79, 79, 80, 79, 78, 74, 70, 
    68, 66, 64, 60, 55, 50, 43, 40, 35, 32, # age 40
    30, 29, 27, 25, 22, 20, 19, 17, 15, 15,
    14, 14, 13, 12, 10,  9,  7,  7,  8, 10, # age 60
    11, 11, 12, 12, 14, 15, 17, 18, 18, 19, 
    17, 15, 12,  8,  6,  3,  2,  3,  6, 11, # age 80
])

if PLOT:
    plt.plot(excess_mortality_rate)
    plt.title("Spanish Excess Mortality Rate")
    plt.xlabel("age")
    plt.ylabel("excess death rate per 10000")
    plt.show()
# -

# Source: [Age- and Sex-Specific Mortality Associated With the 1918–1919 Influenza Pandemic in Kentucky](https://doi.org/10.1093/infdis/jis745)

# +
# Spanish Flu
scaling_factor_spanish = 0.020794936228115346 * 1.0225036876370608 * 0.9986865920589011
df_g_spanish = df_g.copy()
df_g_spanish['g0']['0-19'] = scaling_factor_spanish * excess_mortality_rate[:20].mean()
df_g_spanish['g0']['20-29'] = scaling_factor_spanish * excess_mortality_rate[20:30].mean()
df_g_spanish['g0']['30-39'] = scaling_factor_spanish * excess_mortality_rate[30:40].mean()
df_g_spanish['g0']['40-49'] = scaling_factor_spanish * excess_mortality_rate[40:50].mean()
df_g_spanish['g0']['50-59'] = scaling_factor_spanish * excess_mortality_rate[50:60].mean()
df_g_spanish['g0']['60-69'] = scaling_factor_spanish * excess_mortality_rate[60:70].mean()
df_g_spanish['g0']['70-79'] = scaling_factor_spanish * excess_mortality_rate[70:].mean()
df_g_spanish['g0']['80-89'] = scaling_factor_spanish * excess_mortality_rate[70:].mean()
df_g_spanish['g0']['90+'] = scaling_factor_spanish * excess_mortality_rate[70:].mean()

df_g_spanish['g1'] = scaling_factor_spanish * ratio1 * df_g_spanish['g0']
df_g_spanish['g2'] = scaling_factor_spanish * ratio2 * df_g_spanish['g0']
df_g_spanish['g3'] = scaling_factor_spanish * ratio3 * df_g_spanish['g0']

df_g_spanish
# -

if PLOT:
    df_g_spanish.plot()
    plt.title("Spanish Flu Risk Profile")
    plt.ylabel("risk")
    plt.show()

# ## 5. Save output

# +
shutil.copytree(FACTORISATION_PATH, OUTPUT_PATH / FACTORISATION_PATH.name)

shutil.copytree(FACTORISATION_PATH, OUTPUT_PATH / "uniform")
os.remove(OUTPUT_PATH / "uniform" / "risk_factors.csv")
save_risk_factors(df_g_uniform, path=OUTPUT_PATH / "uniform")

shutil.copytree(FACTORISATION_PATH, OUTPUT_PATH / "spanish")
os.remove(OUTPUT_PATH / "spanish" / "risk_factors.csv")
save_risk_factors(df_g_spanish, path=OUTPUT_PATH / "spanish")
