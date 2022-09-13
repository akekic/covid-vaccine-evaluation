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
import scipy.optimize as opt
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import itertools

from pathlib import Path
from itertools import product

# RUN_NAME, RUN_DIR = "observed policy", Path("run/2022-01-27_14-37-30_observed_vac_policy")
# RUN_NAME, RUN_DIR = "uniform", Path("run/2022-01-27_16-41-27_uniform_vac_policy")
# RUN_NAME, RUN_DIR = "observed severity", Path("run/2022-01-27_16-44-51_observed_severity")
# RUN_NAME, RUN_DIR = "young to old", Path("run/2022-01-28_11-32-59_young_to_old")
# RUN_NAME, RUN_DIR = "old to young", Path("run/2022-02-01_14-02-57_old_to_young")

# RUN_NAME, RUN_DIR = "observed policy", Path("run/2022-02-02_14-19-24_observed_vac_policy")
# RUN_NAME, RUN_DIR = "uniform", Path("run/2022-02-02_14-28-30_uniform_vac_policy")
# RUN_NAME, RUN_DIR = "old to young", Path("run/2022-02-02_14-30-43_old_to_young")
# RUN_NAME, RUN_DIR = "young to old", Path("run/2022-02-02_14-33-43_young_to_old")
# RUN_NAME, RUN_DIR = "observed severity", Path("run/2022-01-27_16-44-51_observed_severity")

# DELETE
# RUN_NAME, RUN_DIR = "young to old", Path("run/2022-02-09_11-45-56")
# RUN_NAME, RUN_DIR = "young to old", Path("run/2022-02-09_11-58-10")
# RUN_NAME, RUN_DIR = "young to old", Path("run/2022-02-09_15-12-04")
RUN_NAME, RUN_DIR = "young to old", Path("run/2022-02-09_16-39-19_young_to_old_cap")
# RUN_NAME, RUN_DIR = "old to young", Path("run/2022-02-09_16-47-10")

DATA_DIR = Path("../causal-covid-analysis/data/israel/israel_df.pkl")

OUTPUT_DIR = RUN_DIR / "scenario_export"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4

# +
result = np.load(RUN_DIR / "result.npy")

week_dates = np.load(RUN_DIR/ "parameters" / "week_dates.npy")
P_t = np.load(RUN_DIR/ "parameters" / "P_t.npy")
D_a = np.load(RUN_DIR/ "parameters" / "D_a.npy")
P_a = np.load(RUN_DIR/ "parameters" / "P_a.npy")
age_groups = np.load(RUN_DIR/ "parameters" / "age_groups.npy")
age_group_names = np.load(RUN_DIR/ "parameters" / "age_group_names.npy")
weeks = np.load(RUN_DIR/ "parameters" / "weeks.npy")
weeks_extended = np.load(RUN_DIR/ "parameters" / "weeks_extended.npy")
vaccination_statuses = np.load(RUN_DIR/ "parameters" / "vaccination_statuses.npy")

g = np.load(RUN_DIR/ "parameters" / "severity_factorisation" / "g.npy")
f = np.load(RUN_DIR/ "parameters" / "severity_factorisation" / "f.npy")
h_params = np.load(RUN_DIR/ "parameters" / "severity_factorisation" / "h_params.npy")

U_2 = np.load(RUN_DIR/ "vaccination_policy" / "U_2.npy")
u_3 = np.load(RUN_DIR/ "vaccination_policy" / "u_3.npy")

result.shape
# -

df = pd.read_pickle(DATA_DIR)
df.head()

df.columns

# population data 
df_tmp = df[df['Sunday_date'] == df['Sunday_date'].unique()[0]][['Age_group', 'Population_size']]
df_tmp.to_csv(OUTPUT_DIR / "population_data.csv", index=False)
df_tmp.head()

# observed vaccination data
cols = [
    'Sunday_date',
    'Age_group',
    'unvaccinated_cum_rel',
    '1st_dose_cum_rel',
    '2nd_dose_cum_rel',
    '3rd_dose_cum_rel',
]
cols_save = [
    'Sunday_date',
    'Age_group',
    'unvaccinated_share',
    '1st_dose_share',
    '2nd_dose_share',
    '3rd_dose_share',
]
df_tmp = df.loc[df['Age_group'] != 'total', cols]
df_tmp['unvaccinated_share'] = df_tmp['unvaccinated_cum_rel']
df_tmp['1st_dose_share'] = df_tmp['1st_dose_cum_rel'] - df_tmp['2nd_dose_cum_rel']
df_tmp['2nd_dose_share'] = df_tmp['2nd_dose_cum_rel'] - df_tmp['3rd_dose_cum_rel']
df_tmp['3rd_dose_share'] = df_tmp['3rd_dose_cum_rel']
df_tmp[cols_save].to_csv(OUTPUT_DIR / "observed_vaccination_data.csv", index=False)
df_tmp

# observed cases
cols = [
    'Sunday_date',
    'Age_group',
    'positive_unvaccinated',
    'positive_after_1st_dose',
    'positive_after_2nd_dose',
    'positive_after_3rd_dose',
]
df_tmp = df[cols]
df_tmp.to_csv(OUTPUT_DIR / "observed_infection_data.csv", index=False)
df_tmp.head()

# +
# compute vaccinations status from vaccination policy parametrisation
u_product = np.zeros((len(weeks), len(age_groups), 4))  # [t, a, v]
waning_time_distr = np.zeros((len(age_groups), len(weeks), len(weeks)))  # [a, t, w]

for t, a in product(weeks, age_groups):
    # unvaccinated
    tmp_0 = 0
    for t1, t2, t3 in product(
        weeks_extended[t + 1 :],
        weeks_extended[t + 1 :],
        weeks_extended[:],
    ):
        tmp_0 += U_2[a, t1, t2] * u_3[a, t2, t3]
    u_product[t, a, 0] = tmp_0 / D_a[a]

    # after 1st dose
    tmp_1 = 0
    for t1, t2, t3 in product(
        weeks_extended[: t + 1],
        weeks_extended[t + 1 :],
        weeks_extended[:],
    ):
#     for t1, t2 in product(
#         weeks_extended[: t + 1],
#         weeks_extended[t + 1 :],
#     ):
        tmp_1 += U_2[a, t1, t2] * u_3[a, t2, t3]
#         tmp_1 += U_2[a, t1, t2]
    u_product[t, a, 1] = tmp_1 / D_a[a]

    # after 2nd dose
    tmp_2 = 0
    for t1, t2, t3 in product(
        weeks_extended[: t + 1],
        weeks_extended[: t + 1],
        weeks_extended[t + 1 :],
    ):
        tmp_2 += U_2[a, t1, t2] * u_3[a, t2, t3]
        waning_time_distr[a, t, t - t2] += U_2[a, t1, t2] * u_3[a, t2, t3]
    if tmp_2 != 0:
        waning_time_distr[a, t, :] /= tmp_2  # normalisation
    u_product[t, a, 2] = tmp_2 / D_a[a]

    # after 3rd dose
    tmp_3 = 0
    for t1, t2, t3 in product(
        weeks_extended[: t + 1],
        weeks_extended[: t + 1],
        weeks_extended[: t + 1],
    ):
        tmp_3 += U_2[a, t1, t2] * u_3[a, t2, t3]
    u_product[t, a, 3] = tmp_3 / D_a[a]
u_product.shape

# +
df_tmp = None
Age_group = []
Sunday_date = []
data_unvaccinated_share = []
data_1st_dose_share = []
data_2nd_dose_share = []
data_3rd_dose_share = []
for a, name in zip(age_groups, age_group_names):
    Age_group.extend(len(weeks) * [name])
    Sunday_date.extend(week_dates)
    data_unvaccinated_share.extend(u_product[:, a, 0])
    data_1st_dose_share.extend(u_product[:, a, 1])
    data_2nd_dose_share.extend(u_product[:, a, 2])
    data_3rd_dose_share.extend(u_product[:, a, 3])

data = {
    'Sunday_date': Sunday_date,
    'Age_group': Age_group,
    'unvaccinated_share': data_unvaccinated_share,
    '1st_dose_share': data_1st_dose_share,
    '2nd_dose_share': data_2nd_dose_share,
    '3rd_dose_share': data_3rd_dose_share,
}
df_tmp = pd.DataFrame(data).sort_values(['Sunday_date', 'Age_group'])
df_tmp[cols_save].to_csv(OUTPUT_DIR / "scenario_vaccination_data.csv", index=False)
df_tmp.head()
    
    
    
# -

df_tmp['sum'] = (
    df_tmp['unvaccinated_share'] + df_tmp['1st_dose_share'] + df_tmp['2nd_dose_share'] + df_tmp['3rd_dose_share']
)
df_tmp

g.shape

# +
df_g = pd.DataFrame(g).melt(
    ignore_index=False, var_name='Age_group', value_name='risk_factor'
).reset_index().rename(columns={'index':'vaccination_status'})
df_g['Age_group'] = df_g['Age_group'].map(lambda x: age_group_names[x])

df_g.to_csv(OUTPUT_DIR / "risk_factor_data.csv", index=False)
df_g.head()

# +
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


plt.figure(figsize=(6.4, 4.8))
plt.plot(x, y, '.', label='data')
plt.plot(x_fit, y_fit, '-', label='logistic fit')
plt.ylabel("vaccine efficacy vs. infection")
plt.xlabel("months since second shot")
plt.grid()
plt.legend()
plt.show()

def vaccine_efficacy_curve_fit(w):
    week_to_month_factor = 365 / (12 * 7)
    m = w / week_to_month_factor
    return f_ve(m, *popt)


# -

weeks_eval = np.arange(104)
efficacy_eval = vaccine_efficacy_curve_fit(weeks_eval)
df_ve = pd.DataFrame(
    {
        'week_since_last_dose': weeks_eval,
        'vaccine_efficacy': efficacy_eval,
    }
)
df_ve.to_csv(OUTPUT_DIR / "vaccine_efficacy_waning_data.csv", index=False)
df_ve.head()


