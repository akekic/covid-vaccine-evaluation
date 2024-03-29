import datetime
from pathlib import Path

import configargparse


def maybe_str_or_float(arg):
    try:
        return float(arg)  # try convert to float
    except ValueError:
        pass
    if arg == "observed" or arg == "observed_relaxed":
        return arg
    raise configargparse.ArgumentTypeError(
        "argument must be an int or 'observed' or 'observed_relaxed'"
    )


def create_parser():
    global parser
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
        default=0.9,
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
    return parser
