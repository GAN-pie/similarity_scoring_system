#!/usr/bin/env python3
# coding: utf-8
import time
import os
from os import path
import json
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score, auc
from scipy.stats import ttest_ind



from keras.models import load_model, Model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from nnet import *
from data import *
from config import *


if __name__ == "__main__":

    options = Config()
    options.parse_command_line()

    start = time.time()

    if options.use_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu_id
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        set_session(sess)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    options.mode = "test"


    if options.loss == "lecun":
        model = load_model(options.test_model_path, custom_objects={"loss": contrastive_loss(options)})
    else:
        model = load_model(options.test_model_path)

    # model.load_weights(options.load_model_weights_path)

    test_data = Dataset(options)

    [X1, X2], Y = test_data(testing=False)

    np.save(path.join(options.result_files_path, "targets.npy"), Y)

    distance_layer_model = Model(inputs=model.input, outputs=model.get_layer("metric").output)

    predictions = distance_layer_model.predict([X1, X2], batch_size=options.test_batch_size).flatten()

    if options.loss == "lecun":
        normalized_preds = predictions / predictions.max()
    else:
        normalized_preds = predictions

    np.save(path.join(options.result_files_path, "distances.npy"), normalized_preds)

    # compute accuracy and EER on the fly
    fpr, tpr, threshold = roc_curve(Y, normalized_preds)

    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = {0:.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig(path.join(options.result_files_path, "roc_auc.pdf"))

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    eer_threshold = interp1d(fpr, threshold)(eer)
    decisions = pd.Series(normalized_preds).apply(lambda x: 1 if x > eer_threshold else 0)
    acc = accuracy_score(Y, decisions)
    print("EER={0:.2f} - threshold={1:.4f} - accuracy={2:.2f}".format(eer, float(eer_threshold), acc), file=sys.stderr)


    # compute tscore
    mask_tar = np.where(Y == 0)
    mask_nontar = np.where(Y == 1)

    tar_norm_preds = normalized_preds[mask_tar]
    nontar_norm_preds = normalized_preds[mask_nontar]

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
    plt.savefig(path.join(options.result_files_path, "scores_distribution.pdf"))

    results = {}
    results["EER"] = float(eer)
    results["EER-threshold"] = float(eer_threshold)
    results["accuracy"] = float(acc)
    results["tscore"] = float(t_score)
    results["pvalue"] = float(p_value)

    with open(path.join(options.result_files_path, "evaluation_metrics.json"), "w") as fd:
        json.dump(results, fd)

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start), file=sys.stderr)
