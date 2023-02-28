import logging
import datetime
import os
import sys
from pprint import pformat

import configargparse

from pathlib import Path

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


def maybe_str_or_float(arg):
    try:
        return float(arg)  # try convert to int
    except ValueError:
        pass
    if arg == "observed" or arg == "observed_relaxed":
        return arg
    raise configargparse.ArgumentTypeError(
        "argument must be an int or 'observed' or 'observed_relaxed'"
    )


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
        nargs="*",
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
        "--vaccination-policy",
        nargs="*",
        type=str,
        default=["observed"],
        help="vaccination policy",
    )
    parser.add_argument(
        "--waning-states",
        nargs="*",
        default=[1, 2, 3],
        type=int,
        help="vaccination states with waning immunity",
    )
    parser.add_argument(
        "--vaccine-acceptance-rate",
        default="observed_relaxed",
        type=maybe_str_or_float,
        help="vaccine acceptance rate to assume for vaccination policies",
    )
    parser.add_argument(
        "--generate-correction-factors",
        action="store_true",
        help="generate infection dynamics correction factor (otherwise assume correction_factor=1 for age groups and time steps)",
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
        default=80,
        help="Contact matrix mixing parameter",
    )
    parser.add_argument(
        "--V1-eff",
        type=int,
        default=70,
        help="Vaccine efficacy of first dose used for infection dynamics",
    )
    parser.add_argument(
        "--V2-eff",
        type=int,
        default=90,
        help="Vaccine efficacy of second dose used for infection dynamics",
    )
    parser.add_argument(
        "--V3-eff",
        type=int,
        default=95,
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
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f") + "_" + "waning_exp"
    )
    if args.save_output:
        exp_dir.mkdir(parents=True, exist_ok=False)
        file_handler = logging.FileHandler(exp_dir / "logs.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logger.info(pformat(args))
    logger.info(parser.format_values())

    factorisation_data_dir = args.input_dir[0]
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
                factorisation_data_dir, generate_correction_factors=False
            ),
        ),
        "risk_ranked_reversed": RankedVaccinationPolicyGenerator(
            params,
            ranking="risk_ranked_reversed",
            vaccine_acceptance_rate=args.vaccine_acceptance_rate,
            constraints_per_dose=True,
            severity_factorisation=SeverityFactorisation(
                factorisation_data_dir, generate_correction_factors=False
            ),
        ),
    }

    results = {}

    # waning_curves = generate_waning_curves()
    for i, input in enumerate(args.input_dir):
        waning_path = input / "vaccine_efficacy_waning_data.csv"
        for j, policy in enumerate(args.vaccination_policy):
            logger.info(
                f"Simulation of vaccination policy "
                f"{policy} ({i*len(args.vaccination_policy)+ j + 1}/{len(args.vaccination_policy)*len(args.input_dir)})"
            )

            vaccination_policy_generator = vaccination_policy_generator_map[policy]
            vaccination_policy = vaccination_policy_generator.generate()

            severity_factorisation = SeverityFactorisation(
                input,
                generate_correction_factors=args.generate_correction_factors,
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
                waning_path=waning_path,
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
            results[policy] = result.sum()

            # save output
            if args.save_output:
                dir = Path(args.output_dir)
                run_dir = exp_dir / input.name / policy
                save_run(
                    params=params,
                    vaccination_policy=vaccination_policy,
                    severity_factorisation=severity_factorisation,
                    target_function=target_function,
                    result=result,
                    result_samples=result_samples,
                    run_dir=run_dir,
                    factorisation_data_dir=input,
                )
            logger.info("************************************************************")

    logger.info(f"Results: {pformat(results)}")
