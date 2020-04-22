# coding: utf-8

from os import path

import numpy as np
import pandas as pd

from keras import backend as K
from keras.layers import Input, Dense, Activation, Lambda
from keras.optimizers import Adadelta, SGD
from keras.models import Model
from keras.callbacks import Callback

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, accuracy_score, auc
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import seaborn

from .feedforward import make_feedforward


def eucd(x):
    (x1, x2) = x
    return K.sqrt(K.sum(K.square(x1 - x2), axis=1, keepdims=True))


def mand(x):
    (x1, x2) = x
    return K.sum(K.abs(x1 - x2), axis=1, keepdims=True)


def contrastive_loss(options):
    """
    Compute the contrastive loss Y.Lecun.
    When examples are similar Y=0.
    """

    def loss(y, d):
        similar_term = (1.0 - y) * K.square(d)
        dissimilar_term = y * K.square(K.maximum((options.margin - d), 0))
        return K.mean(similar_term + K.maximum(dissimilar_term, K.epsilon()))

    return loss


def make_siamese(options):
    input_shape = options.input_shape
    margin = options.margin

    if options.optimizer == "adadelta":
        optimizer = Adadelta(lr=options.lr, rho=options.lr_decay)
    elif options.optimizer == "sgd":
        optimizer = SGD(lr=options.lr, momentum=options.momentum)

    left_in_layer = Input(shape=input_shape, name="left_in")
    right_in_layer = Input(shape=input_shape, name="right_in")

    feedforward = make_feedforward(options)
    feedforward.summary()

    left_embedding = feedforward(left_in_layer)
    right_embedding = feedforward(right_in_layer)

    if options.metric == "manhattan":
        metric = Lambda(mand, name="metric")([left_embedding, right_embedding])
    else:
        metric = Lambda(eucd, name="metric")([left_embedding, right_embedding])

    if options.loss == "lecun":
        loss = contrastive_loss(options)
        siamese = Model(inputs=[left_in_layer, right_in_layer], outputs=metric, name="siamese")
        siamese.compile(optimizer, loss)
    elif options.loss == "binary_crossentropy":
        loss = "binary_crossentropy"
        metric = Dense(
            1,
            activation="sigmoid",
            kernel_initializer="glorot_normal",
            bias_initializer="zero", name="similarity")(metric)
        siamese = Model(inputs=[left_in_layer, right_in_layer], outputs=metric, name="siamese")
        siamese.compile(optimizer, loss, metrics=["accuracy"])

    siamese.summary()
    return siamese


class Accuracy(Callback):
    def __init__(self, data, options):
        self.acc = []
        self._data = data
        self.options = options

    def on_epoch_end(self, epoch, logs={}):
        predictions = self.model.predict(self._data(testing=True)).flatten()

        if self.options.loss == "lecun":
            normalized_preds = predictions / predictions.max()
        else:
            normalized_preds = predictions

        true_labels = self._data.trials[:, 2].flatten()

        fpr, tpr, threshold = roc_curve(true_labels, normalized_preds)

        if epoch % self.options.print_freq == 0:
            roc_auc = auc(fpr, tpr)

            fig = plt.figure()
            plt.plot(fpr, tpr, label="ROC curve (area = {0:.2f})".format(roc_auc))
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Receiver Operating Characteristic")
            plt.legend(loc="lower right")
            plt.savefig(path.join(self.options.result_files_path, "validation_roc_auc_"+str(epoch)+".pdf"))


        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer_threshold = interp1d(fpr, threshold)(eer)

        decisions = pd.Series(normalized_preds).apply(lambda x: 1 if x > eer_threshold else 0)
        acc = accuracy_score(true_labels, decisions)
        self.acc += [acc]
        print("\tvalidation accuracy = {0:.4f} threshold = {1:.4f}".format(acc, eer_threshold))
