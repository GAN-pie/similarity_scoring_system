#!/usr/bin/env python3
# coding: utf-8

import sys
import time
import os
from os import path
import json
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score, auc
from scipy.stats import ttest_ind


"""
This module gives scores agglomerative functions
"""



def concatenate_scores(arrays_list):
    arrays = []
    for array_path in arrays_list:
        arrays += [np.load(array_path)]

    return np.concatenate(arrays, axis=0)


def concatenate_targets(targets_list):
    targets = []
    for targets_path in targets_list:
        targets += [np.load(targets_path)]

    return np.concatenate(targets, axis=0)


def test_performances(predictions, targets):

    fpr, tpr, threshold = roc_curve(targets, predictions)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = {0:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("roc_auc.pdf")

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = interp1d(fpr, threshold)(eer)
    decisions = pd.Series(predictions).apply(lambda x: 1 if x > eer_threshold else 0)
    acc = accuracy_score(targets, decisions)
    print("EER={0:.2f} - threshold={1:.4f} - accuracy={2:.2f}".format(eer, float(eer_threshold), acc), file=sys.stderr)

    mask_tar = np.where(targets == 1)
    mask_nontar = np.where(targets == 0)

    tar_norm_preds = predictions[mask_tar]
    nontar_norm_preds = predictions[mask_nontar]

    t_score, p_value = ttest_ind(tar_norm_preds, nontar_norm_preds, equal_var=False)
    print("t-score={0:.4f} - p-value={1:.4f}".format(t_score, p_value), file=sys.stderr)

    fig = plt.figure()
    seaborn.boxplot(data=pd.DataFrame({"target": tar_norm_preds, "nontarget": nontar_norm_preds}))
    plt.title("Similarity scores")
    plt.savefig("scores_distribution.pdf")

    results = {}
    results["EER"] = float(eer)
    results["EER-threshold"] = float(eer_threshold)
    results["accuracy"] = float(acc)
    results["tscore"] = float(t_score)
    results["pvalue"] = float(p_value)

    with open("evaluation_metrics.json", "w") as fd:
        json.dump(results, fd)


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, metavar='predictions', nargs="+")
    parser.add_argument("--targets", required=True, metavar='outcomes', nargs="+")
    options = parser.parse_args()

    predictions = concatenate_scores(options.scores)
    targets = concatenate_targets(options.targets)

    test_performances(predictions, targets)

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start), file=sys.stderr)
