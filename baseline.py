#!/usr/bin/env python3
# coding: utf-8

import json
import os
from os import path
import time
import sys
from concurrent import futures

import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score, auc
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn

from config import *


def compute_baseline(features):
    x, y = features
    return np.sqrt(np.sum(np.square(x - y)))
    # return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


if __name__ == "__main__":

    options = Config()

    english_features = np.load(options.test_en_array)
    french_features = np.load(options.test_fr_array)
    trials = np.load(options.test_trials_array)

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

    distances = distances / distances.max()

    fpr, tpr, threshold = roc_curve(trials[:,2], distances)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = {0:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(path.join(options.result_files_path, "baseline_roc_auc.pdf"))

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = interp1d(fpr, threshold)(eer)
    decisions = pd.Series(distances).apply(lambda x: 1 if x > eer_threshold else 0)
    acc = accuracy_score(trials[:,2], decisions)
    print("EER={0:.2f} - threshold={1:.4f} - accuracy={2:.2f}".format(eer, float(eer_threshold), acc), file=sys.stderr)

    tar_norm_preds = distances[mask_target]
    nontar_norm_preds = distances[mask_nontar]

    t_score, p_value = ttest_ind(tar_norm_preds, nontar_norm_preds, equal_var=False)
    print("t-score={0:.4f} - p-value={1:.4f}".format(t_score, p_value), file=sys.stderr)

    # print("{0:.2f} {1:.2f} {2:.2f}".format(acc, t_score, p_value), file=sys.stdout)

    # fig = plt.figure()
    # seaborn.kdeplot(tar_norm_preds, shade=True, color="g")
    # seaborn.kdeplot(nontar_norm_preds, shade=True, color="r")
    # plt.xlabel("scores")
    # plt.title("Distribution of target/nontarget scores")
    # plt.savefig(path.join(mdl_dir, path.basename(data_dir)+"_scores_distribution.pdf"))
    fig = plt.figure()
    seaborn.boxplot(data=pd.DataFrame({"target": tar_norm_preds, "nontarget": nontar_norm_preds}))
    plt.title("Similarity scores")
    plt.savefig(path.join(options.result_files_path, "baseline_scores_distribution.pdf"))

    results = {}
    results["EER"] = float(eer)
    results["EER-threshold"] = float(eer_threshold)
    results["accuracy"] = float(acc)
    results["tscore"] = float(t_score)
    results["pvalue"] = float(p_value)

    with open(path.join(options.result_files_path, "baseline_evaluation_metrics.json"), "w") as fd:
        json.dump(results, fd)




    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = interp1d(fpr, threshold)(eer)
    decisions = pd.Series(distances).apply(lambda x: 1 if x > eer_threshold else 0)
    acc = accuracy_score(trials[:,2], decisions)
    print("EER={0:.2f} - threshold={1:.4f} - accuracy={2:.2f}".format(eer, float(eer_threshold), acc), file=sys.stderr)
