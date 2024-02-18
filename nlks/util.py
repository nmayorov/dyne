"""Utility functions."""
import numpy as np


def rms(data):
    return np.mean(np.square(data), axis=0) ** 0.5
