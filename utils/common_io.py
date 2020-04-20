# coding: utf-8

"""
The utils module provides commonly used functions.
"""

import numpy as np


def read_lst_file(fname):
    """
    Returns a list where each element refers to each line of given file.
    """

    assert isinstance(fname, str)

    with open(fname, "r") as fd:
        return [line.strip() for line in fd]


def read_data_file(fname):
    """
    Return a dict for a <key> <values vector> file.
    """

    assert isinstance(fname, str)

    with open(fname, "r") as fd:
        data = {}
        for line in fd:
            line = line.strip()
            arr = line.split(" ")
            label = arr[0]
            data[label] = [float(x) for x in arr[1:]]
            assert np.all(np.isfinite(data[label])), "not finite value"

            return data
