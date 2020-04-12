# -*- coding: utf-8 -*-
from datetime import datetime

import torch

from pytorch_lightning.logging import TestTubeLogger


def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(
        save_dir="experiments/",
        version=job_id if job_id else dt_string,
        name="lightning_logs",
    )
