#!/usr/bin/env python3
# coding: utf-8

import os
from os import path
import time
import pickle

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

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


    # prepare data

    # train_data = DatasetGenerator(options)
    # test_data = DatasetGenerator(options)
    (X, Y) = Dataset(options)()
    test_data = Dataset(options, validation=True)


    ckpt = ModelCheckpoint(path.join(options.checkpoints_path, "model_checkpoint.h5"), monitor="val_loss", save_best_only=True)
    callb = [ckpt]

    decay = ReduceLROnPlateau(monitor='val_loss', factor=options.lr_decay, patience=4)
    callb += [decay]

    # if options.loss == "lecun":
    acc_clb = Accuracy(test_data, options)
    callb += [acc_clb]

    if options.early_stopping:
        early = EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=8)
        callb += [early]

    # tb = TensorBoard(log_dir=path.join(options.log_files_path, "graph"), histogram_freq=1, write_graph=True)
    # callb += [tb]

    siamese = make_siamese(options)

    siamese.save_weights(path.join(options.checkpoints_path, "init_weights.h5"))

    # train model
    hist = siamese.fit(
        x=X,
        y=Y,
        epochs=options.max_epoch,
        callbacks=callb,
        validation_data=test_data()
    )

    # hist = siamese.fit_generator(
    #     train_data,
    #     steps_per_epoch=len(train_data),
    #     epochs=options.max_epoch,
    #     callbacks=callb,
    #     validation_data=test_data,
    #     validation_steps=len(test_data),
    #     max_queue_size=50,
    #     workers=options.num_workers,
    #     use_multiprocessing=True,
    #     shuffle=False)

    with open(path.join(options.log_files_path, "train_history.pkl"), "wb") as fd:
        fd.write(pickle.dumps(hist.history))

    end = time.time()
    print("program ended in {0:.2f} seconds".format(end - start))
