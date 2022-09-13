import logging
import datetime
import os
import sys
from pprint import pformat

import configargparse

from pathlib import Path

import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from causal_model import TargetFunction, Parameters, SeverityFactorisation
from utils import save_run
from vaccination_policy import ObservedVaccinationPolicyGenerator

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5s]  %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def compute_dose_increase(params: Parameters, delta_rate: float):
    extra_doses_needed = (params.D_a * delta_rate * 3).sum()
    total_doses = params.observed_vaccinations.sum()
    return extra_doses_needed / total_doses


COMBINED_AGE_GROUPS = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5, 6, 7, 8],
]


def compute_absolute_ppt_acceptance_change(
    combined_group, delta_rate, params: Parameters
):
    number_of_converted_people = params.D_a.sum() * delta_rate
    number_of_unvaccinated_people = (
        1 - params.vaccine_acceptance_rate[combined_group, 0]
    ) * params.D_a[combined_group]
    if number_of_unvaccinated_people.sum() < number_of_converted_people:
        return None

    number_of_converted_people_per_age_group = (
        number_of_converted_people / number_of_unvaccinated_people.sum()
    ) * number_of_unvaccinated_people

    ppt_change_per_age_group = (
        number_of_converted_people_per_age_group / params.D_a[combined_group]
    )

    ppt_acceptance_change = np.zeros_like(params.age_groups, dtype=float)
    ppt_acceptance_change[combined_group] = ppt_change_per_age_group
    return 100 * ppt_acceptance_change


def run_simulation(params, args, ppt_acceptance_change, run_dir):
    factorisation_data_dir = Path(args.input_dir)
    vaccination_policy_generator = ObservedVaccinationPolicyGenerator(
        params, ppt_acceptance_change=ppt_acceptance_change
    )
    vaccination_policy = vaccination_policy_generator.generate()
    severity_factorisation = SeverityFactorisation(
        factorisation_data_dir,
        load_correction_factor=args.load_correction_factor,
        vaccination_policy=vaccination_policy,
        observed_vaccination_policy=ObservedVaccinationPolicyGenerator(
            params
        ).generate(),
        C_mat_param=args.C_mat_param,
        V1_eff=args.V1_eff,
        V2_eff=args.V2_eff,
        V3_eff=args.V3_eff,
        draws=args.draws,
        influx=args.influx,
    )
    target_function = TargetFunction(
        params=params,
        severity_factorisation=severity_factorisation,
        waning_states=args.waning_states,
    )
    result_global, results_global_samples = target_function(
        vaccination_policy=vaccination_policy,
        n_samples=args.samples,
        n_workers=args.workers,
    )
    result_global_no_id, _ = target_function(
        vaccination_policy=vaccination_policy, ignore_infection_dynamics=True
    )

    # save output
    if args.save_output:
        save_run(
            params=params,
            vaccination_policy=vaccination_policy,
            severity_factorisation=severity_factorisation,
            target_function=target_function,
            result=result_global,
            result_no_id=result_global_no_id,
            result_samples=results_global_samples,
            run_dir=run_dir,
            factorisation_data_dir=factorisation_data_dir,
        )
    return result_global, result_global_no_id


if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description="Compute severity target function for given vaccine allocation policy."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        is_config_file=True,
        help="path to config",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="input data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="run",
        help="output directory",
    )
    parser.add_argument(
        "-s",
        "--save-output",
        action="store_true",
        help="save output in output directory",
    )
    parser.add_argument(
        "--waning-states",
        nargs="*",
        default=[1, 2, 3],
        type=list,
        help="vaccination states with waning immunity",
    )
    parser.add_argument(
        "--load-correction-factor",
        action="store_true",
        help="load infection dynamics correction factor",
    )
    parser.add_argument(
        "--acceptance-rate-delta",
        nargs="+",
        default=[-1, 0, 1],
        type=float,
        help="Changes in acceptance rate to simulate.",
    )
    parser.add_argument(
        "--start-week",
        type=datetime.date.fromisoformat,
        default=None,
        help="Sunday date of start week - format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end-week",
        type=datetime.date.fromisoformat,
        default=None,
        help="Sunday date of end week - format YYYY-MM-DD (inclusive)",
    )
    parser.add_argument(
        "--C-mat-param",
        type=str,
        default=None,
        help="Contact matrix mixing parameter",
    )
    parser.add_argument(
        "--V1-eff",
        type=int,
        default=None,
        help="Vaccine efficacy of first dose used for infection dynamics",
    )
    parser.add_argument(
        "--V2-eff",
        type=int,
        default=None,
        help="Vaccine efficacy of second dose used for infection dynamics",
    )
    parser.add_argument(
        "--V3-eff",
        type=int,
        default=None,
        help="Vaccine efficacy of third dose used for infection dynamics",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=500,
        help="Number of draws used in training of infection dynamics model",
    )
    parser.add_argument(
        "--influx",
        type=float,
        default=0.5,
        help="Influx infections used in infection dynamics",
    )
    parser.add_argument(
        "--skip-global-mode",
        action="store_true",
    )
    parser.add_argument(
        "--skip-relative-mode",
        action="store_true",
    )
    parser.add_argument(
        "--skip-absolute-mode",
        action="store_true",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of samples of target function to compute",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of workers to use for multiprocessing",
    )

    args = parser.parse_args()

    exp_dir = args.output_dir / (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f") + "_" + "acc_exp"
    )
    if args.save_output:
        exp_dir.mkdir(parents=True, exist_ok=False)
        file_handler = logging.FileHandler(exp_dir / "logs.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logger.info(pformat(args))
    logger.info(parser.format_values())

    factorisation_data_dir = Path(args.input_dir)
    delta_rate_values = np.array(args.acceptance_rate_delta) / 100
    results_global = []

    for delta_rate in delta_rate_values:
        params = Parameters(
            path=factorisation_data_dir,
            start_week=args.start_week,
            end_week=args.end_week,
        )

        if not args.skip_global_mode:
            logger.info(
                f"Simulation of {delta_rate*100} ppt change in global vaccine acceptance"
            )
            ppt_acceptance_change = 100 * delta_rate
            result_global, result_global_no_id = run_simulation(
                params=params,
                args=args,
                ppt_acceptance_change=ppt_acceptance_change,
                run_dir=exp_dir / "global" / f"delta_rate_{delta_rate}",
            )

            logger.info(f"Result = {result_global.sum()}")
            logger.info(f"Result no inf. dyn. = {result_global_no_id.sum()}")
            results_global.append(result_global.sum())

        if not args.skip_relative_mode:
            results_age_rel = []
            for a in params.age_groups:
                logger.info(
                    f"Simulation of {delta_rate * 100} ppt change in vaccine acceptance "
                    f"for age group {params.age_group_names[a]}"
                )
                base_vector = np.zeros_like(params.age_groups)
                base_vector[a] = 1
                ppt_acceptance_change = 100 * delta_rate * base_vector
                run_dir = (
                    exp_dir
                    / "age_rel"
                    / params.age_group_names[a]
                    / f"delta_rate_{delta_rate}"
                )
                result_age_rel, result_age_rel_no_id = run_simulation(
                    params=params,
                    args=args,
                    ppt_acceptance_change=ppt_acceptance_change,
                    run_dir=run_dir,
                )
                logger.info(f"Result = {result_age_rel.sum()}")
                logger.info(f"Result no inf. dyn. = {result_age_rel_no_id.sum()}")
                results_age_rel.append(result_age_rel.sum())

        if not args.skip_absolute_mode:
            results_age_abs = []
            combined_age_names = [
                params.age_group_names[0],
                params.age_group_names[1],
                params.age_group_names[2],
                params.age_group_names[3],
                params.age_group_names[4],
                "60+",
            ]
            for combined_group, combined_group_name in zip(
                COMBINED_AGE_GROUPS, combined_age_names
            ):
                logger.info(
                    f"Simulation of {delta_rate * 100} ppt-equivalent change in vaccine acceptance "
                    f"for age groups {[params.age_group_names[a] for a in combined_group]}"
                )
                ppt_acceptance_change = compute_absolute_ppt_acceptance_change(
                    combined_group, delta_rate, params
                )
                if ppt_acceptance_change is None:
                    logger.warning(
                        "Acceptance rate change larger than unvaccinated population. "
                        "This scenario is skipped."
                    )
                    continue

                run_dir = (
                    exp_dir
                    / "age_abs"
                    / combined_group_name
                    / f"delta_rate_{delta_rate}"
                )
                result_age_abs, result_age_abs_no_id = run_simulation(
                    params=params,
                    args=args,
                    ppt_acceptance_change=ppt_acceptance_change,
                    run_dir=run_dir,
                )
                logger.info(f"Result = {result_age_abs.sum()}")
                logger.info(f"Result no inf. dyn. = {result_age_abs_no_id.sum()}")
                results_age_abs.append(result_age_abs.sum())

    logger.info(f"Delta acceptance rate values: {delta_rate_values}")
    if not args.skip_global_mode:
        logger.info(f"Results global: {results_global}")
