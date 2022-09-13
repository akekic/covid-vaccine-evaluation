import logging
import os
import shutil
from typing import Optional

import numpy as np
import numpy.typing as npt

from pathlib import Path

from causal_model import TargetFunction, Parameters, SeverityFactorisation
from vaccination_policy import VaccinationPolicy

logger = logging.getLogger(__name__)


def save_run(
    params: Parameters,
    vaccination_policy: VaccinationPolicy,
    severity_factorisation: SeverityFactorisation,
    target_function: TargetFunction,
    result: npt.NDArray,
    run_dir: Path,
    factorisation_data_dir: Path,
    result_samples: Optional[npt.NDArray] = None,
    result_no_id: Optional[npt.NDArray] = None,
):
    params.save(run_dir)
    vaccination_policy.save(run_dir)
    severity_factorisation.save(run_dir)
    target_function.save(run_dir)
    np.save(run_dir / "result", result)
    if result_samples is not None:
        np.save(run_dir / "result_samples", result_samples)
    if result_no_id is not None:
        np.save(run_dir / "result_no_id", result_no_id)

    # copy factorisation data
    src_files = os.listdir(factorisation_data_dir)
    dest_dir = run_dir / "factorisation_data"
    dest_dir.mkdir(parents=True, exist_ok=False)
    for file_name in src_files:
        full_file_name = factorisation_data_dir / file_name
        if os.path.isfile(full_file_name):
            dest_file_name = dest_dir / file_name
            shutil.copy(full_file_name, dest_file_name)
    logger.info(f"Run saved in {run_dir}")
