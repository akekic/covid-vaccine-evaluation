# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: causal-covid-analysis
#     language: python
#     name: causal-covid-analysis
# ---

# +
import numpy as np
import pandas as pd

PLOT_HEIGHT = 4.8
PLOT_WIDTH = 6.4


# + code_folding=[]
def load_run(run_dir):
    run = {}

    run["result"] = np.load(run_dir / "result.npy")

    try:
        run["result_samples"] = np.load(run_dir / "result_samples.npy")
    except FileNotFoundError:
        print(f"Result samples not found for {run_dir / 'result_samples.npy'}")
        run["result_samples"] = None

    try:
        run["result_no_id"] = np.load(run_dir / "result_no_id.npy")
    except FileNotFoundError:
        print(
            f"Results without infection dynamics not found for {run_dir / 'result_no_id_test.npy'}"
        )
        run["result_no_id"] = None

    run["week_dates"] = np.load(run_dir / "parameters" / "week_dates.npy")
    run["P_t"] = np.load(run_dir / "parameters" / "P_t.npy")
    run["D_a"] = np.load(run_dir / "parameters" / "D_a.npy")
    run["P_a"] = np.load(run_dir / "parameters" / "P_a.npy")
    run["age_groups"] = np.load(run_dir / "parameters" / "age_groups.npy")
    run["age_group_names"] = np.load(run_dir / "parameters" / "age_group_names.npy")
    run["weeks"] = np.load(run_dir / "parameters" / "weeks.npy")
    run["weeks_extended"] = np.load(run_dir / "parameters" / "weeks_extended.npy")
    run["vaccination_statuses"] = np.load(
        run_dir / "parameters" / "vaccination_statuses.npy"
    )

    run["g"] = np.load(run_dir / "severity_factorisation" / "g.npy")
    run["f_0"] = np.load(run_dir / "severity_factorisation" / "f_0.npy")
    run["f_1"] = np.load(run_dir / "severity_factorisation" / "f_1.npy")

    run["observed_severity_df"] = pd.read_csv(
        run_dir / "factorisation_data" / "observed_severity_data.csv",
        parse_dates=["Sunday_date"],
    )
    run["observed_infection_df"] = pd.read_csv(
        run_dir / "factorisation_data" / "observed_infection_data.csv",
        parse_dates=["Sunday_date"],
    )
    run["observed_vaccination_df"] = pd.read_csv(
        run_dir / "factorisation_data" / "vaccination_data.csv",
        parse_dates=["Sunday_date"],
    )

    h_params_path = run_dir / "severity_factorisation" / "h_params.npy"
    try:
        run["h_params"] = np.load(h_params_path)
    except FileNotFoundError:
        print(f"h_params data not found under {h_params_path}")
        run["h_params"] = None

    vaccine_efficacy_params_path = (
        run_dir / "severity_factorisation" / "vaccine_efficacy_params.npy"
    )
    try:
        run["vaccine_efficacy_params"] = np.load(vaccine_efficacy_params_path)
    except FileNotFoundError:
        print(
            f"vaccine_efficacy_params data not found under {vaccine_efficacy_params_path}"
        )
        run["vaccine_efficacy_params"] = None

    infection_dynamics_path = (
        run_dir / "severity_factorisation" / "infection_dynamics.csv"
    )
    try:
        run["infection_dynamics_df"] = pd.read_csv(infection_dynamics_path)
    except FileNotFoundError:
        print(f"Infection dynamics data not found under {infection_dynamics_path}")
        run["infection_dynamics_df"] = None

    infection_dynamics_samples_path = (
        run_dir / "severity_factorisation" / "infection_dynamics_samples.npy"
    )
    try:
        run["infection_dynamics_samples"] = np.load(infection_dynamics_samples_path)
    except FileNotFoundError:
        print(
            f"Infection dynamics samples data not found under {infection_dynamics_samples_path}"
        )
        run["infection_dynamics_samples"] = None

    median_weekly_base_R_t_path = (
        run_dir / "severity_factorisation" / "median_weekly_base_R_t.npy"
    )
    try:
        run["median_weekly_base_R_t"] = np.load(median_weekly_base_R_t_path)
    except FileNotFoundError:
        print(f"Weekly base R_t data not found under {median_weekly_base_R_t_path}")
        run["median_weekly_base_R_t"] = None
    
    weekly_base_R_t_samples_path = (
        run_dir / "severity_factorisation" / "weekly_base_R_t_samples.npy"
    )
    try:
        run["weekly_base_R_t_samples"] = np.load(weekly_base_R_t_samples_path)
    except FileNotFoundError:
        print(f"Weekly base R_t samples data not found under {weekly_base_R_t_samples_path}")
        run["weekly_base_R_t_samples"] = None

    median_weekly_eff_R_t_path = (
        run_dir / "severity_factorisation" / "median_weekly_eff_R_t.npy"
    )
    try:
        run["median_weekly_eff_R_t"] = np.load(median_weekly_eff_R_t_path)
    except FileNotFoundError:
        print(f"Weekly effective R_t data not found under {median_weekly_eff_R_t_path}")
        run["median_weekly_eff_R_t"] = None

    run["U_2"] = np.load(run_dir / "vaccination_policy" / "U_2.npy")
    run["u_3"] = np.load(run_dir / "vaccination_policy" / "u_3.npy")

    return run


# +
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


def compute_weekly_first_doses_per_age(U_2):
    weekly_first_doses = U_2.sum(axis=2)[:, :-1]
    return weekly_first_doses


def compute_weekly_second_doses_per_age(U_2):
    weekly_second_doses = U_2.sum(axis=1)[:, :-1]
    return weekly_second_doses


def compute_weekly_third_doses_per_age(U_2, u_3):
    U_2_repeat = np.repeat(
        U_2[:, :-1, :-1, np.newaxis], U_2.shape[1] - 1, axis=3
    )  # copy over dimension t3
    u_3_repeat = np.repeat(
        u_3[:, np.newaxis, :-1, :-1], u_3.shape[1] - 1, axis=1
    )  # copy over dimension t1
    weekly_third_doses = (U_2_repeat * u_3_repeat).sum(axis=(1, 2))
    return weekly_third_doses
    

def compute_weekly_doses_per_age(U_2, u_3):
    weekly_first_doses = compute_weekly_first_doses_per_age(U_2)
    weekly_second_doses = compute_weekly_second_doses_per_age(U_2)
    weekly_third_doses = compute_weekly_third_doses_per_age(U_2, u_3)
    
    return weekly_first_doses + weekly_second_doses + weekly_third_doses


# +
def infection_incidence(
    infection_dynamics_sample,
    population,
    week_dates,
    split_date_to=None,
    split_date_from=None,
    end_date=None,
):
    if split_date_to is not None:
        split_date_index = np.argwhere(week_dates == split_date_to).flatten()[0]
        infection_dynamics_sample_slice = infection_dynamics_sample[
            :split_date_index, ...
        ]
    elif split_date_from is not None:
        assert end_date is not None
        split_date_index = np.argwhere(week_dates == split_date_from).flatten()[0]
        end_date_index = np.argwhere(week_dates == end_date).flatten()[0]
        infection_dynamics_sample_slice = infection_dynamics_sample[
            split_date_index:end_date_index, ...
        ]
    else:
        infection_dynamics_sample_slice = infection_dynamics_sample
    infection_incidence = infection_dynamics_sample_slice.sum() * 1e5 / population
    return infection_incidence


def infection_incidence_observed(
    df_input,
    population,
    split_date_to=None,
    split_date_from=None,
    end_date=None,
):
    df = df_input.copy()
    df = df[df["Age_group"] == "total"]
    if split_date_to is not None:
        df = df[df["Sunday_date"] < split_date_to]
    elif split_date_from is not None:
        assert end_date is not None
        df = df[(df["Sunday_date"] >= split_date_from) & (df["Sunday_date"] < end_date)]
    infection_incidence = (
        df["positive_unvaccinated"]
        + df["positive_after_1st_dose"]
        + df["positive_after_2nd_dose"]
        + df["positive_after_3rd_dose"]
    ).sum() * 1e5 / population

    return infection_incidence


def severe_case_incidence(
    result,
    n_weeks,
    week_dates,
    split_date_to=None,
    split_date_from=None,
    end_date=None,
):
    if split_date_to is not None:
        split_date_index = np.argwhere(week_dates == split_date_to).flatten()[0]
        result_slice = result[:split_date_index, ...]
    elif split_date_from is not None:
        assert end_date is not None
        split_date_index = np.argwhere(week_dates == split_date_from).flatten()[0]
        end_date_index = np.argwhere(week_dates == end_date).flatten()[0]
        result_slice = result[split_date_index:end_date_index, ...]
    else:
        result_slice = result
    severe_case_incidence = n_weeks * result_slice.sum() * 1e5
    return severe_case_incidence

def severe_case_incidence_observed(
    df_input,
    population,
    split_date_to=None,
    split_date_from=None,
    end_date=None,
):
    df = df_input.copy()
    df = df[df["Age_group"] == "total"]
    if split_date_to is not None:
        df = df[df["Sunday_date"] < split_date_to]
    elif split_date_from is not None:
        assert end_date is not None
        df = df[(df["Sunday_date"] >= split_date_from) & (df["Sunday_date"] < end_date)]
    severe_case_incidence = (
        df["hosp_unvaccinated"]
        + df["hosp_after_1st_dose"]
        + df["hosp_after_2nd_dose"]
        + df["hosp_after_3rd_dose"]
    ).sum() * 1e5 / population

    return severe_case_incidence

def severe_case_incidence_observed_trajectory(
    df_input,
    population,
    age_group=None,
    split_date_to=None,
    split_date_from=None,
    end_date=None,
):
    df = df_input.copy()
    if age_group is not None:
        df = df[df["Age_group"] == age_group]
    else:
        df = df[df["Age_group"] == "total"]
    
    if split_date_to is not None:
        df = df[df["Sunday_date"] < split_date_to]
    elif split_date_from is not None:
        assert end_date is not None
        df = df[(df["Sunday_date"] >= split_date_from) & (df["Sunday_date"] < end_date)]
    severe_case_incidence = (
        df["hosp_unvaccinated"]
        + df["hosp_after_1st_dose"]
        + df["hosp_after_2nd_dose"]
        + df["hosp_after_3rd_dose"]
    ).values * 1e5 / population

    return severe_case_incidence
# -


