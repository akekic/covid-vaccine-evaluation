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
# RUN_NAME, RUN_DIR = "young to old", Path("run/2022-02-09_16-39-19")
# RUN_NAME, RUN_DIR = "old to young", Path("run/2022-02-09_16-47-10")
# RUN_NAME, RUN_DIR = "old to young", Path("run/2022-02-11_15-28-34")  # old logic
# RUN_NAME, RUN_DIR = "old to young", Path("run/2022-02-11_15-33-39")  # new logic

# RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-07_15-39-40_elderly_first")
# RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-07_16-35-15_elderly_first")  # constraints per dose
# RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-07_17-14-44_elderly_first")  # higher acceptance rate (0.9 instead of 0.8)
# RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-07_17-22-09_elderly_first")  # obs acceptance rate

# RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-20_14-30-18_uniform")
# RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-20_14-34-58_observed")
RUN_NAME, RUN_DIR = "elderly first", Path("../run/2022-04-20_14-42-09_young_first")

DATA_DIR = Path("../../causal-covid-analysis/data/israel/israel_df.pkl")

FIG_DIR = RUN_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4
# -

# ## Load Data

df = pd.read_pickle(DATA_DIR)
df.head()

df.columns

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

try:
    infection_dynamics_df = pd.read_csv(RUN_DIR / "severity_factorisation" / "infection_dynamics.csv")
except FileNotFoundError:
    print(f"Infection dynamics data not found under {RUN_DIR / 'severity_factorisation' / 'infection_dynamics.csv'}")
    infection_dynamics_df = None

g = np.load(RUN_DIR / "severity_factorisation" / "g.npy")
f_0 = np.load(RUN_DIR / "severity_factorisation" / "f_0.npy")
h_params = np.load(RUN_DIR / "severity_factorisation" / "h_params.npy")

U_2 = np.load(RUN_DIR/ "vaccination_policy" / "U_2.npy")
u_3 = np.load(RUN_DIR/ "vaccination_policy" / "u_3.npy")

result.shape
# -

age_group_names

# ## Infection Dynamics

# +
plt.figure()

plt.plot(
    week_dates,
    infection_dynamics_df[infection_dynamics_df['Age_group'] == 'total']['total_infections_observed'], 
    label='observed',
)
plt.plot(
    week_dates,
    infection_dynamics_df[infection_dynamics_df['Age_group'] == 'total']['total_infections_scenario'],
    "--",
    label='scenario',
)
plt.ylabel("Number of infections")
plt.title("Infections overall")
plt.legend()
plt.grid()
plt.show()

# +


fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
ax_flat = axes.flatten()

plt.subplots_adjust(hspace=0.15, wspace=0.1)
plt.suptitle("Infections per age group")

for ax, an in zip(ax_flat, age_group_names):
    ax.plot(
        week_dates,
        infection_dynamics_df[infection_dynamics_df['Age_group'] == an]['total_infections_observed'], 
        label='observed',
    )
    ax.plot(
        week_dates,
        infection_dynamics_df[infection_dynamics_df['Age_group'] == an]['total_infections_scenario'],
        "--",
        label='scenario',
    )
    ax.tick_params(axis='x', labelrotation=45)
    ax.grid()
    ax.set_title(an)
    ax.legend()


# for ag, ax in zip(age_groups, ax_flat):
#     severity = np.array([RUNS[acc]['result'][:, ag, :].sum() for acc in acceptance_rate_delta])
#     normalisation = severity[np.argwhere(acceptance_rate_delta == 0)].flatten()

#     severity_no_id = np.array([RUNS[acc]['result_no_id'][:, ag, :].sum() for acc in acceptance_rate_delta])
#     normalisation_no_id = severity_no_id[np.argwhere(acceptance_rate_delta == 0)].flatten()

#     ax.plot(100*acceptance_rate_delta, 100*(severity/normalisation - 1), "-o", label='inf. dyn.')
#     ax.plot(100*acceptance_rate_delta, 100*(severity_no_id/normalisation_no_id - 1), "-o", label='no inf. dyn.')
#     ax.grid()
#     ax.set_title(age_group_names[ag])
#     ax.legend()
fig.supylabel("Number of infections")
plt.tight_layout()

plt.show()
# -

# ## Severity

hosp_total_observed = df.loc[df['Age_group'] == 'total', 'hosp_total'].sum()
hosp_total_predicted = (D_a.sum() * result.sum(axis=(1, 2))/P_t).sum()
print(f"Total hospitalisations - predicted: {hosp_total_predicted}, observed: {hosp_total_observed}")


# + code_folding=[]
def plot():
    fig, axes = plt.subplots(1, 2, figsize=(2*PLOT_WIDTH, 1*PLOT_HEIGHT), sharey=False, sharex=True)

    plt.subplots_adjust(hspace=0.15, wspace=0.3)
    plt.suptitle('Hospitalisations overall')

    df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axes[0].plot(df_tmp['Sunday_date'], df_tmp['hosp_total'], label="observed")
    axes[0].plot(week_dates, D_a.sum() * result.sum(axis=(1, 2))/P_t, "--", label="scenario")
    axes[0].set_title("Weekly")
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].set_ylabel("weekly hospitalisations")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(df_tmp['Sunday_date'], df_tmp['hosp_total'].cumsum(), label="observed")
    axes[1].plot(week_dates, (D_a.sum() * result.sum(axis=(1,2))/P_t).cumsum(), "--", label="scenario")
    axes[1].set_title("Cumulative")
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].set_ylabel("cumulative hospitalisations")
    axes[1].legend()
    axes[1].grid()
    
    plt.savefig(
        FIG_DIR / (RUN_NAME.replace(" ", "_") +'_hospitalisations_overall.png'),
        dpi=200, 
        bbox_inches='tight',
    )

    plt.show()

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(2, 2, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)

    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by vaccination status')

    df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axes[0,0].plot(df_tmp['Sunday_date'], df_tmp['hosp_unvaccinated'], label="observed")
    axes[0,0].plot(week_dates, D_a.sum() * result.sum(axis=1)[:, 0]/P_t, "--", label="scenario")
    axes[0,0].set_title("Unvaccinated")
    axes[0,0].tick_params(axis='x', labelrotation=45)
    axes[0,0].legend()
    axes[0,0].grid()

    axes[0,1].plot(df_tmp['Sunday_date'], df_tmp['hosp_after_1st_dose'], label="observed")
    axes[0,1].plot(week_dates, D_a.sum() * result.sum(axis=1)[:, 1]/P_t, "--", label="scenario")
    axes[0,1].set_title("1st dose")
    axes[0,1].tick_params(axis='x', labelrotation=45)
    axes[0,1].legend()
    axes[0,1].grid()

    axes[1,0].plot(df_tmp['Sunday_date'], df_tmp['hosp_after_2nd_dose'], label="observed")
    axes[1,0].plot(week_dates, D_a.sum() * result.sum(axis=1)[:, 2]/P_t, "--", label="scenario")
    axes[1,0].set_title("2nd dose")
    axes[1,0].tick_params(axis='x', labelrotation=45)
    axes[1,0].legend()
    axes[1,0].grid()

    axes[1,1].plot(df_tmp['Sunday_date'], df_tmp['hosp_after_3rd_dose'], label="observed")
    axes[1,1].plot(week_dates, D_a.sum() * result.sum(axis=1)[:, 3]/P_t, "--", label="scenario")
    axes[1,1].set_title("3rd dose")
    axes[1,1].tick_params(axis='x', labelrotation=45)
    axes[1,1].legend()
    axes[1,1].grid()
    
    plt.savefig(
        FIG_DIR / (RUN_NAME.replace(" ", "_") +'_hospitalisations_by_vac_status.png'),
        dpi=200, 
        bbox_inches='tight',
    )

    plt.show()

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    ax_flat = axes.flatten()
    ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by age group (all vac. statuses)')
    
    for i, (ax, ag) in enumerate(zip(ax_flat, ages)):
        df_tmp = df[df['Age_group'] == ag]
        ax.plot(df_tmp['Sunday_date'], df_tmp['hosp_total'], label="observed")
        ax.plot(week_dates, D_a.sum() * result.sum(axis=2)[:, i]/P_t, "--", label="scenario")
        ax.set_title(f"{ag}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.grid()
        

    df_tmp = df[df['Age_group'] == '60-69']
    
    plt.savefig(
        FIG_DIR / (RUN_NAME.replace(" ", "_") +'_hospitalisations_by_age_group.png'),
        dpi=200, 
        bbox_inches='tight',
    )
    plt.show()

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    ax_flat = axes.flatten()
    ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by age group (unvaccinated)')
    
    for i, (ax, ag) in enumerate(zip(ax_flat, ages)):
        df_tmp = df[df['Age_group'] == ag]
        ax.plot(df_tmp['Sunday_date'], df_tmp['hosp_unvaccinated'], label="observed")
        ax.plot(week_dates, D_a.sum() * result[:, i, 0]/P_t, "--", label="scenario")
        ax.set_title(f"{ag}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.grid()
        

    df_tmp = df[df['Age_group'] == '60-69']

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    ax_flat = axes.flatten()
    ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by age group (1st dose)')
    
    for i, (ax, ag) in enumerate(zip(ax_flat, ages)):
        df_tmp = df[df['Age_group'] == ag]
        ax.plot(df_tmp['Sunday_date'], df_tmp['hosp_after_1st_dose'], label="observed")
        ax.plot(week_dates, D_a.sum() * result[:, i, 1]/P_t, "--", label="scenario")
        ax.set_title(f"{ag}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.grid()
        

    df_tmp = df[df['Age_group'] == '60-69']

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    ax_flat = axes.flatten()
    ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by age group (2nd dose)')
    
    for i, (ax, ag) in enumerate(zip(ax_flat, ages)):
        df_tmp = df[df['Age_group'] == ag]
        ax.plot(df_tmp['Sunday_date'], df_tmp['hosp_after_2nd_dose'], label="observed")
        ax.plot(week_dates, D_a.sum() * result[:, i, 2]/P_t, "--", label="scenario")
        ax.set_title(f"{ag}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.grid()
        

    df_tmp = df[df['Age_group'] == '60-69']

plot()


# + code_folding=[]
def plot():
    fig, axes = plt.subplots(3, 3, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    ax_flat = axes.flatten()
    ages = sorted(list(set(df['Age_group'].unique()) - {'total'}))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Number of weekly hopspitalisations by age group (3rd dose)')
    
    for i, (ax, ag) in enumerate(zip(ax_flat, ages)):
        df_tmp = df[df['Age_group'] == ag]
        ax.plot(df_tmp['Sunday_date'], df_tmp['hosp_after_3rd_dose'], label="observed")
        ax.plot(week_dates, D_a.sum() * result[:, i, 3]/P_t, "--", label="scenario")
        ax.set_title(f"{ag}")
        ax.tick_params(axis='x', labelrotation=45)
        ax.legend()
        ax.grid()
        

    df_tmp = df[df['Age_group'] == '60-69']

plot()


# -

# ## Vaccination Policy

# + code_folding=[]
def plot():
    fig, axes = plt.subplots(1, 3, figsize=(3*PLOT_WIDTH, PLOT_HEIGHT), sharey=True)

    df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    axes[0].plot(df_tmp['Sunday_date'], df_tmp['1st_dose'], color=colors[0], label="observed")
    axes[0].plot(week_dates, U_2.sum(axis=(0, 2))[:-1], "--", color=colors[1], label="scenario")
    axes[0].legend()
    axes[0].tick_params(axis='x', labelrotation=45)
    axes[0].set_title("1st doses")
    axes[0].set_ylabel("doses")

    axes[1].plot(df_tmp['Sunday_date'], df_tmp['2nd_dose'], color=colors[0], label="observed")
    axes[1].plot(week_dates, U_2.sum(axis=(0, 1))[:-1], "--", color=colors[1], label="scenario")
    axes[1].legend()
    axes[1].tick_params(axis='x', labelrotation=45)
    axes[1].set_title("2nd doses")

    U_2_repeat = np.repeat(
        U_2[:, :-1, :-1, np.newaxis], U_2.shape[1]-1, axis=3
    )  # copy over dimension t3
    u_3_repeat = np.repeat(
        u_3[:, np.newaxis, :-1, :-1], u_3.shape[1]-1, axis=1
    )  # copy over dimension t1
    third_doses_policy = (U_2_repeat * u_3_repeat).sum(axis=(0, 1, 2))

    axes[2].plot(df_tmp['Sunday_date'], df_tmp['3rd_dose'], color=colors[0], label="observed")
    axes[2].plot(week_dates, third_doses_policy, "--", color=colors[1], label="scenario")
    axes[2].legend()
    axes[2].tick_params(axis='x', labelrotation=45)
    axes[2].set_title("3rd doses")

    plt.show()

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(2, 2, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    plt.suptitle('Vaccination status over time')

    df_tmp = df[df['Age_group'] == 'total']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    axes[0,0].plot(
        df_tmp['Sunday_date'],
        1 - df_tmp['1st_dose_cum_rel'],
        color=colors[0],
        label="observed",
    )
    axes[0,0].plot(
        week_dates, 
        1 - U_2.sum(axis=(0, 2))[:-1].cumsum()/D_a.sum(),
        "--",
        color=colors[1],
        label="scenario",
    )
    axes[0,0].legend()
    axes[0,0].tick_params(axis='x', labelrotation=45)
    axes[0,0].set_title("unvaccinated")
    axes[0,0].set_ylabel("doses")
    axes[0,0].grid()

    axes[0,1].plot(
        df_tmp['Sunday_date'],
        df_tmp['1st_dose_cum_rel'] - df_tmp['2nd_dose_cum_rel'],
        color=colors[0],
        label="observed",
    )
    axes[0,1].plot(
        week_dates, 
        (U_2.sum(axis=(0, 2))[:-1] - U_2.sum(axis=(0, 1))[:-1]).cumsum()/D_a.sum(),
        "--",
        color=colors[1],
        label="scenario",
    )
    axes[0,1].legend()
    axes[0,1].tick_params(axis='x', labelrotation=45)
    axes[0,1].set_title("1st dose")
    axes[0,1].set_ylabel("doses")
    axes[0,1].grid()
    
    U_2_repeat = np.repeat(
        U_2[:, :-1, :-1, np.newaxis], U_2.shape[1]-1, axis=3
    )  # copy over dimension t3
    u_3_repeat = np.repeat(
        u_3[:, np.newaxis, :-1, :-1], u_3.shape[1]-1, axis=1
    )  # copy over dimension t1
    third_doses_policy = (U_2_repeat * u_3_repeat).sum(axis=(0, 1, 2))

    axes[1,0].plot(
        df_tmp['Sunday_date'],
        df_tmp['2nd_dose_cum_rel'] - df_tmp['3rd_dose_cum_rel'],
        color=colors[0],
        label="observed",
    )
    axes[1,0].plot(
        week_dates, 
        (U_2.sum(axis=(0, 1))[:-1] - third_doses_policy).cumsum()/D_a.sum(),
        "--",
        color=colors[1],
        label="scenario",
    )
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', labelrotation=45)
    axes[1,0].set_title("2nd dose")
    axes[1,0].grid()

    axes[1,1].plot(
        df_tmp['Sunday_date'],
        df_tmp['3rd_dose_cum_rel'],
        color=colors[0],
        label="observed",
    )
    axes[1,1].plot(
        week_dates, 
        (third_doses_policy).cumsum()/D_a.sum(),
        "--",
        color=colors[1],
        label="scenario",
    )
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', labelrotation=45)
    axes[1,1].set_title("3rd dose")
    axes[1,1].grid()

    plt.show()

plot()
# -

print(f"Share of never vaccinated: {U_2[:, -1, -1].sum() / D_a.sum()}")

# + code_folding=[0]
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


# + code_folding=[]
def plot():
    fig, axes = plt.subplots(2, 2, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Vaccination status by age (scenario)')


    for a, name in zip(age_groups, age_group_names):
        axes[0,0].plot(week_dates, u_product[:, a, 0], label=f"{name}")
    axes[0,0].tick_params(axis='x', labelrotation=45)
    axes[0,0].set_title("V = 0")
    axes[0,0].set_ylabel("population share")
    axes[0,0].grid()
    
    for a, name in zip(age_groups, age_group_names):
        axes[0,1].plot(week_dates, u_product[:, a, 1], label=f"{name}")
    axes[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0,1].tick_params(axis='x', labelrotation=45)
    axes[0,1].set_title("V = 1")
    axes[0,1].grid()
    
    for a, name in zip(age_groups, age_group_names):
        axes[1,0].plot(week_dates, u_product[:, a, 2], label=f"{name}")
    axes[1,0].tick_params(axis='x', labelrotation=45)
    axes[1,0].set_title("V = 2")
    axes[1,0].set_ylabel("population share")
    axes[1,0].grid()
    
    for a, name in zip(age_groups, age_group_names):
        axes[1,1].plot(week_dates, u_product[:, a, 3], label=f"{name}")
    axes[1,1].tick_params(axis='x', labelrotation=45)
    axes[1,1].set_title("V = 3")
    axes[1,1].grid()
    
    plt.savefig(
        FIG_DIR / (RUN_NAME.replace(" ", "_") +'_vaccination_status_by_age.png'),
        dpi=200, 
        bbox_inches='tight',
    )

    plt.show()

plot()


# + code_folding=[0]
def plot():
    fig, axes = plt.subplots(2, 2, figsize=(2*PLOT_WIDTH, 2*PLOT_HEIGHT), sharey=True, sharex=True)
    plt.subplots_adjust(hspace=0.15, wspace=0.1)
    plt.suptitle('Vaccination status by age (observed)')


    for ag in age_group_names:
        df_tmp = df[df['Age_group'] == ag]
        axes[0,0].plot(
            df_tmp['Sunday_date'],
            df_tmp['unvaccinated_cum_rel'],
            label=f"{ag}",
        )
    axes[0,0].tick_params(axis='x', labelrotation=45)
    axes[0,0].set_title("V = 0")
    axes[0,0].set_ylabel("population share")
    axes[0,0].grid()
    
    for ag in age_group_names:
        df_tmp = df[df['Age_group'] == ag]
        axes[0,1].plot(
            df_tmp['Sunday_date'],
            df_tmp['1st_dose_cum_rel'] - df_tmp['2nd_dose_cum_rel'],
            label=f"{ag}",
        )
    axes[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0,1].tick_params(axis='x', labelrotation=45)
    axes[0,1].set_title("V = 1")
    axes[0,1].grid()
    
    for ag in age_group_names:
        df_tmp = df[df['Age_group'] == ag]
        axes[1,0].plot(
            df_tmp['Sunday_date'],
            df_tmp['2nd_dose_cum_rel'] - df_tmp['3rd_dose_cum_rel'],
            label=f"{ag}",
        )
    axes[1,0].tick_params(axis='x', labelrotation=45)
    axes[1,0].set_title("V = 2")
    axes[1,0].set_ylabel("population share")
    axes[1,0].grid()
    
    for ag in age_group_names:
        df_tmp = df[df['Age_group'] == ag]
        axes[1,1].plot(
            df_tmp['Sunday_date'],
            df_tmp['3rd_dose_cum_rel'],
            label=f"{ag}",
        )
    axes[1,1].tick_params(axis='x', labelrotation=45)
    axes[1,1].set_title("V = 3")
    axes[1,1].grid()
    
    plt.savefig(
        FIG_DIR / (RUN_NAME.replace(" ", "_") +'_vaccination_status_by_age.png'),
        dpi=200, 
        bbox_inches='tight',
    )

    plt.show()

plot()


# + code_folding=[0]
def plot():
    plt.figure()
    for a in age_groups:
        plt.plot(week_dates, (waning_time_distr[a, :, :] * weeks).sum(axis=1), label=age_group_names[a])

    plt.legend()
    plt.ylabel("mean waning time")
    plt.title("Waning time")
    plt.grid()
    plt.gca().tick_params(axis='x', labelrotation=45)
    plt.savefig(
            FIG_DIR / (RUN_NAME.replace(" ", "_") +'_mean_waning_time.png'),
            dpi=200, 
            bbox_inches='tight',
        )
    plt.show()

plot()
# -


