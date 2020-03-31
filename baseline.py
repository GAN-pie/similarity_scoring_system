#!/usr/bin/env python3
# coding: utf-8

import os
import sys
from os import path
import time
from concurrent import futures

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score
from scipy.stats import ttest_ind

from config import *
from utils import *


def compute_baseline(features):
    x, y = features
    # return np.sqrt(np.sum(np.square(x - y)))
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


if __name__ == "__main__":

    options = Config()

    english_features = np.load(options.val_en_array)
    french_features = np.load(options.val_fr_array)
    trials = np.load(options.val_trials_array)

    batch_size = math.ceil(len(trials) / 20)

    pairs_generator = ( (english_features[trial[0]], french_features[trial[1]]) for trial in trials )

    with futures.ThreadPoolExecutor() as executor:
        distances = np.array(list(executor.map(compute_baseline, pairs_generator)))


    mask_target = trials[:,2] == 0
    mask_nontar = trials[:,2] == 1


    target_distances = distances[mask_target]
    nontar_distances = distances[mask_nontar]


    print("Nontarget distances : max {0:.2f} - min {1:.2f} - mean {2:.2f} - std {3:.2f} - 25q {4:.2f} - 50q {5:.2f} - 75q {6:.2f}".format(np.max(nontar_distances), np.min(nontar_distances), np.mean(nontar_distances), np.std(nontar_distances), np.percentile(nontar_distances, 25), np.percentile(nontar_distances, 50), np.percentile(nontar_distances, 75)))


    print("Target distances : max {0:.2f} - min {1:.2f} - mean {2:.2f} - std {3:.2f} - 25q {4:.2f} - 50q {5:.2f} - 75q {6:.2f}".format(np.max(target_distances), np.min(target_distances), np.mean(target_distances), np.std(target_distances), np.percentile(target_distances, 25), np.percentile(target_distances, 50), np.percentile(target_distances, 75)))



    fpr, tpr, threshold = roc_curve(trials[:,2], distances)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = interp1d(fpr, threshold)(eer)
    decisions = pd.Series(distances).apply(lambda x: 1 if x > eer_threshold else 0)
    acc = accuracy_score(trials[:,2], decisions)
    print("EER={0:.2f} - threshold={1:.4f} - accuracy={2:.2f}".format(eer, float(eer_threshold), acc), file=sys.stderr)
