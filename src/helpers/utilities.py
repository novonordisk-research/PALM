import logging

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import matthews_corrcoef


logger = logging.getLogger(__name__)


def nJobs(mem_per_job: float) -> int:  # noqa: N802 function names should be lowercase
    """
    Determine the number of jobs that can be run in parallel, given the requirement per job and available resources

    Args:
        mem_per_job (int): The memory required per job in GB
    """
    total_mem = psutil.virtual_memory().total / 2 ** (10 * 3)
    total_cores = psutil.cpu_count()
    n_jobs = max(1, min(total_cores, int(np.floor(total_mem / mem_per_job))))
    logger.info(f"Total memory: {total_mem}GB, N jobs: {n_jobs}, GB/job = {total_mem / n_jobs}")
    logger.info(f"Current available memory: {psutil.virtual_memory().available / 2 ** (10 * 3)}")
    return n_jobs


class CudaOutOfMemoryError(Exception):
    def __init__(self, message, seq_lengths=None):
        super().__init__(message)
        self.seq_lengths = seq_lengths


def find_optimal_cutoff(y: np.array, y_pred_prob: np.array, func: callable = matthews_corrcoef):
    """
    Brute force scan to find the best probability cutoff

    Args:
        y: the labels (0,1)
        y_pred_prob: the predicted probability of 1
        func: the metric that is to be computed

    Returns:
        float: the optimal cutoff
    """
    if func not in {matthews_corrcoef}:
        raise ValueError("Currently only MCC is supported")

    # scan over cutoffs with a step size of 0.01
    cutoffs = np.linspace(0, 1, 101)
    y_pred = y_pred_prob > np.expand_dims(cutoffs, axis=-1)
    metric_df = pd.DataFrame({"cutoffs": cutoffs, "predictions": [x for x in y_pred]})

    # calculate the metric for each cutoff and select the best
    metric_df["metric"] = metric_df.apply(lambda x: func(y, x.predictions), axis=1)
    optimal_cutoff = float(metric_df.iloc[metric_df.metric.idxmax()].cutoffs)
    metric_val = float(metric_df.iloc[metric_df.metric.idxmax()].metric)
    return optimal_cutoff, metric_val
