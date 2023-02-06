import logging
import datetime
import os
from pprint import pformat

import configargparse
import warnings

from pathlib import Path

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
        "--vaccination-policy",
        type=str,
        default="observed",
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
        type=int,
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

    run_dir = args.output_dir / (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "_"
        + args.vaccination_policy
    )
    if args.save_output:
        run_dir.mkdir(parents=True, exist_ok=False)
        file_handler = logging.FileHandler(run_dir / "logs.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logger.info(pformat(args))
    logger.info(parser.format_values())

    logger.info(f"{args.workers}/{os.cpu_count()} CPUs used for multiprocessing")

    factorisation_data_dir = Path(args.input_dir)
    params = Parameters(
        path=factorisation_data_dir,
        start_week=args.start_week,
        end_week=args.end_week,
    )

    vaccination_policy_generator_map = {
        "observed": ObservedVaccinationPolicyGenerator(params),
        "uniform": UniformVaccinationPolicyGenerator(params),
        "elderly_first": RankedVaccinationPolicyGenerator(
            params,
            ranking="elderly_first",
            constraints_per_dose=True,
            vaccine_acceptance_rate="observed_relaxed",
        ),
        "young_first": RankedVaccinationPolicyGenerator(
            params,
            ranking="young_first",
            constraints_per_dose=True,
            vaccine_acceptance_rate="observed_relaxed",
        ),
    }

    vaccination_policy_generator = vaccination_policy_generator_map[
        args.vaccination_policy
    ]
    vaccination_policy = vaccination_policy_generator.generate()

    severity_factorisation = SeverityFactorisation(
        factorisation_data_dir=factorisation_data_dir,
        generate_correction_factors=args.generate_correction_factors,
        vaccination_policy=vaccination_policy,
        observed_vaccination_policy=vaccination_policy_generator_map[
            "observed"
        ].generate(),
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

    # save output
    if args.save_output:
        dir = Path(args.output_dir)
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
