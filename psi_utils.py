import numpy as np
import pandas as pd


def calculate_psi(reference, current, bins=10):
    """
    Calculate PSI for a single feature.
    reference: pandas Series of the baseline (training) values
    current: pandas Series of the new (live) values
    bins: number of buckets to divide values into
    """
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_percents = ref_counts / len(reference)
    cur_percents = cur_counts / len(current)

    ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
    cur_percents = np.where(cur_percents == 0, 0.0001, cur_percents)

    psi_value = np.sum((cur_percents - ref_percents) * np.log(cur_percents / ref_percents))
    return psi_value