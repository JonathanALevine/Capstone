import numpy as np


def transform_labels(values):
    return -1/np.log10(np.abs(values))