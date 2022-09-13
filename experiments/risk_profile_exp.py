import logging
import datetime
import os
import shutil
import sys

from itertools import product
from pprint import pformat
from pathlib import Path

import numpy as np
import pandas as pd

from parser_risk_profile_exp import create_parser

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from causal_model import TargetFunction, Parameters, SeverityFactorisation
from utils import save_run
from vaccination_policy import (
    ObservedVaccinationPolicyGenerator,
    UniformVaccinationPolicyGenerator,
    RankedVaccinationPolicyGenerator,
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5s]  %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def save_risk_factors(df_g, path):
    df_g.to_csv(path / "risk_factors.csv")


def generate_risk_profiles(type: str, scaling_factor: float):
    factorisation_path = Path("data/factorisation-with-fit")
    tmp_output_path = Path(
        "tmp/risk-profile-experiment"
    ) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

    df_g = pd.read_csv(factorisation_path / "risk_factors.csv").set_index("Age_group")

    ratio1 = (df_g["g1"] / df_g["g0"]).mean()
    ratio2 = (df_g["g2"] / df_g["g0"]).mean()
    ratio3 = (df_g["g3"] / df_g["g0"]).mean()

    if type == "flat":
        df_g_profile = flat_risk_factors(df_g, ratio1, ratio2, ratio3, scaling_factor)
    elif type == "spanish":
        df_g_profile = spanish_risk_factors(
            df_g, ratio1, ratio2, ratio3, scaling_factor
        )
    elif type == "covid":
        df_g_profile = None
    else:
        raise ValueError()

    shutil.copytree(factorisation_path, tmp_output_path / type)
    if type != "covid":
        os.remove(tmp_output_path / type / "risk_factors.csv")
        save_risk_factors(df_g_profile, path=tmp_output_path / type)

    return tmp_output_path / type


def spanish_risk_factors(df_g, ratio1, ratio2, ratio3, scaling_factor):
    # fmt: off
    excess_mortality_rate = np.array([
        110,
        87, 67, 52, 38, 26, 21, 19, 18, 17, 17,
        17, 18, 19, 21, 26, 37, 42, 48, 56, 60,
        70, 75, 77, 79, 79, 80, 79, 78, 74, 70,
        68, 66, 64, 60, 55, 50, 43, 40, 35, 32,  # age 40
        30, 29, 27, 25, 22, 20, 19, 17, 15, 15,
        14, 14, 13, 12, 10, 9, 7, 7, 8, 10,  # age 60
        11, 11, 12, 12, 14, 15, 17, 18, 18, 19,
        17, 15, 12, 8, 6, 3, 2, 3, 6, 11,  # age 80
    ])
    # fmt: on

    df_g_spanish = df_g.copy()
    df_g_spanish["g0"]["0-19"] = scaling_factor * excess_mortality_rate[:20].mean()
    df_g_spanish["g0"]["20-29"] = scaling_factor * excess_mortality_rate[20:30].mean()
    df_g_spanish["g0"]["30-39"] = scaling_factor * excess_mortality_rate[30:40].mean()
    df_g_spanish["g0"]["40-49"] = scaling_factor * excess_mortality_rate[40:50].mean()
    df_g_spanish["g0"]["50-59"] = scaling_factor * excess_mortality_rate[50:60].mean()
    df_g_spanish["g0"]["60-69"] = scaling_factor * excess_mortality_rate[60:70].mean()
    df_g_spanish["g0"]["70-79"] = scaling_factor * excess_mortality_rate[70:].mean()
    df_g_spanish["g0"]["80-89"] = scaling_factor * excess_mortality_rate[70:].mean()
    df_g_spanish["g0"]["90+"] = scaling_factor * excess_mortality_rate[70:].mean()
    df_g_spanish["g1"] = scaling_factor * ratio1 * df_g_spanish["g0"]
    df_g_spanish["g2"] = scaling_factor * ratio2 * df_g_spanish["g0"]
    df_g_spanish["g3"] = scaling_factor * ratio3 * df_g_spanish["g0"]
    return df_g_spanish


def flat_risk_factors(df_g, ratio1, ratio2, ratio3, scaling_factor):
    df_g_flat = df_g.copy()
    df_g_flat["g0"] = scaling_factor * 1.0
    df_g_flat["g1"] = scaling_factor * ratio1 * df_g_flat["g0"]
    df_g_flat["g2"] = scaling_factor * ratio2 * df_g_flat["g0"]
    df_g_flat["g3"] = scaling_factor * ratio3 * df_g_flat["g0"]
    return df_g_flat


def normalisation_run(input_dir, args):
    params = Parameters(
        path=input_dir,
        start_week=args.start_week,
        end_week=args.end_week,
    )
    policy = UniformVaccinationPolicyGenerator(params).generate()
    severity_factorisation = SeverityFactorisation(
        input_dir,
        load_correction_factor=args.load_correction_factor,
        vaccination_policy=policy,
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
    result, _ = target_function(
        vaccination_policy=policy,
        n_samples=0,
        n_workers=0,
    )
    return result.sum()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    exp_dir = args.output_dir / (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
        + "_"
        + "risk_profile_exp"
    )
    if args.save_output:
        exp_dir.mkdir(parents=True, exist_ok=False)
        file_handler = logging.FileHandler(exp_dir / "logs.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logger.info(pformat(args))
    logger.info(parser.format_values())

    # generate risk profiles
    profile_types = ["covid", "spanish", "flat"]
    scaling_factors = {"covid": 1.0, "spanish": 1.0, "flat": 1.0}
    input_dir_dict = {
        type: generate_risk_profiles(type, scaling_factor=scaling_factors[type])
        for type in profile_types
    }
    normalisation_reached = False
    normalisation_reached_per_type = {
        type: False for type in set(profile_types) - {"covid"}
    }

    # reference run
    logger.info("Normalisation reference run")
    normalisation_result = normalisation_run(input_dir_dict["covid"], args)
    logger.info(f"Result = {normalisation_result}")
    logger.info("************************************************************")

    iter_ctr = 1
    while not normalisation_reached:
        logger.info(f"Normalisation round no. {iter_ctr}")
        iter_ctr += 1
        input_dir_dict = {
            type: generate_risk_profiles(type, scaling_factor=scaling_factors[type])
            for type in profile_types
        }
        for type in set(profile_types) - {"covid"}:
            if normalisation_reached_per_type[type]:
                continue

            logger.info(f"Run simulation for risk profile '{type}'")

            result = normalisation_run(input_dir_dict[type], args)
            logger.info(f"Result = {result}")

            scaling_factors[type] *= normalisation_result / result
            relative_normalisation_error = (
                abs(result - normalisation_result) / normalisation_result
            )
            logger.info(
                f"Relative normalisation error = {relative_normalisation_error}"
            )

            if relative_normalisation_error < 1e-4:
                normalisation_reached_per_type[type] = True
                logger.info(f"Normalisation reached for risk profile '{type}'")
            else:
                logger.info(f"Normalisation not yet reached for risk profile '{type}'")
            logger.info("************************************************************")
        if all(normalisation_reached_per_type.values()):
            normalisation_reached = True
            logger.info(f"Final scaling factors: {scaling_factors}")

    logger.info(input_dir_dict)

    results = {}

    for input, policy in product(input_dir_dict.values(), args.vaccination_policy):
        logger.info(f"Simulation of input {input.name} and vaccination policy {policy}")

        factorisation_data_dir = Path(input)
        params = Parameters(
            path=factorisation_data_dir,
            start_week=args.start_week,
            end_week=args.end_week,
        )

        vaccination_policy_generator_map = {
            "observed": ObservedVaccinationPolicyGenerator(params),
            "uniform": UniformVaccinationPolicyGenerator(
                params,
                use_vaccine_acceptance=(
                    True
                    if args.vaccine_acceptance_rate in ["observed_relaxed", "observed"]
                    else False
                ),
            ),
            "elderly_first": RankedVaccinationPolicyGenerator(
                params,
                ranking="elderly_first",
                vaccine_acceptance_rate=args.vaccine_acceptance_rate,
                constraints_per_dose=True,
            ),
            "young_first": RankedVaccinationPolicyGenerator(
                params,
                ranking="young_first",
                vaccine_acceptance_rate=args.vaccine_acceptance_rate,
                constraints_per_dose=True,
            ),
            "risk_ranked": RankedVaccinationPolicyGenerator(
                params,
                ranking="risk_ranked",
                vaccine_acceptance_rate=args.vaccine_acceptance_rate,
                constraints_per_dose=True,
                severity_factorisation=SeverityFactorisation(
                    factorisation_data_dir, load_correction_factor=False
                ),
            ),
            "risk_ranked_reversed": RankedVaccinationPolicyGenerator(
                params,
                ranking="risk_ranked_reversed",
                vaccine_acceptance_rate=args.vaccine_acceptance_rate,
                constraints_per_dose=True,
                severity_factorisation=SeverityFactorisation(
                    factorisation_data_dir, load_correction_factor=False
                ),
            ),
        }
        if input == "flat" and policy in ["risk_ranked", "risk_ranked_reversed"]:
            logger.info(
                f"Using policy 'uniform' instead of {policy} for risk profile 'flat'"
            )
            vaccination_policy_generator = vaccination_policy_generator_map["uniform"]
        else:
            vaccination_policy_generator = vaccination_policy_generator_map[policy]
        vaccination_policy = vaccination_policy_generator.generate()
        vaccination_policy.validate()  # TODO: should this be commented out?

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

        result, result_samples = target_function(
            vaccination_policy=vaccination_policy,
            n_samples=args.samples,
            n_workers=args.workers,
        )

        logger.info(f"Result = {result.sum()}")
        results[(factorisation_data_dir.name, policy)] = result.sum()

        # save output
        if args.save_output:
            dir = Path(args.output_dir)
            run_dir = exp_dir / factorisation_data_dir.name / policy
            save_run(
                params=params,
                vaccination_policy=vaccination_policy,
                severity_factorisation=severity_factorisation,
                target_function=target_function,
                result=result,
                result_samples=result_samples,
                run_dir=run_dir,
                factorisation_data_dir=factorisation_data_dir,
            )
        logger.info("************************************************************")
    logger.info(f"Results: {pformat(results)}")
