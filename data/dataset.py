#!/usr/bin/env python3
# coding: utf-8

import pickle

import numpy as np
from keras.utils import Sequence



class Dataset:
    """The Dataset class provides the full trial pairs set ready to be processed by neural net.
    Warning: can be very memory consumptive with large trials file."""

    def __init__(self, options, validation=False):

        self._input_dim = options.input_size
        self.mode = options.mode

        if self.mode == "train":
            self._shuffle = True
            self.batch_size = options.train_batch_size
            if validation:
                self._shuffle = False
                self.batch_size = options.test_batch_size
                self.trials = np.load(options.val_trials_array)
                self.left_inputs = np.load(options.val_en_array)
                self.right_inputs = np.load(options.val_fr_array)
            else:
                self.trials = np.load(options.train_trials_array)
                self.left_inputs = np.load(options.train_en_array)
                self.right_inputs = np.load(options.train_fr_array)
        else:
            self._shuffle = False
            self.batch_size = options.test_batch_size
            self.trials = np.load(options.test_trials_array)
            self.left_inputs = np.load(options.test_en_array)
            self.right_inputs = np.load(options.test_fr_array)


        self._num_pairs = len(self.trials)

    def __len__(self):
        return self._num_pairs

    @property
    def num_pairs(self):
        return self._num_pairs

    def __call__(self, testing=False):
        X1 = np.zeros((self._num_pairs, self._input_dim))
        X2 = np.zeros((self._num_pairs, self._input_dim))
        Y = np.zeros((self._num_pairs, ), dtype=int)

        if self._shuffle:
            np.random.shuffle(self.trials)

        for i, (x, y, t) in enumerate(self.trials):
            X1[i] = self.left_inputs[x]
            X2[i] = self.right_inputs[y]
            Y[i] = t

        if testing:
            return [X1, X2]
        else:
            return ([X1, X2], Y)


class DatasetGenerator(Sequence):
    """
    The DatasetGenerator class provides a batched data generator for trial pairs
    ready to be processed by neural net. It's an memory saving alternative to the Dataset class to use with large trials file.
    """

    def __init__(self, options, validation=False):
        """
        Generator initialize with trials file plus left and right features path.
        """

        self.input_size = options.input_size
        self.mode = options.mode

        if self.mode == "train":
            self._shuffle = True
            self.batch_size = options.train_batch_size
            if validation:
                self.trials = np.load(options.val_trials_array)
                self.left_inputs = np.load(options.val_en_array)
                self.right_inputs = np.load(options.val_fr_array)
            else:
                self._shuffle = False
                self.trials = np.load(options.train_trials_array)
                self.left_inputs = np.load(options.train_en_array)
                self.right_inputs = np.load(options.train_fr_array)
        else:
            self._shuffle = False
            self.batch_size = options.test_batch_size
            self.trials = np.load(options.test_trials_array)
            self.left_inputs = np.load(options.test_en_array)
            self.right_inputs = np.load(options.test_fr_array)

        self.indexes = np.arange(len(self.trials))
        self._on_epoch_end()

        self._epoch_completed = 0
        self._remaining = 0

    @property
    def num_pairs(self):
        return len(self.trials)

    def __len__(self):
        """Returns the number of iterations to proceed all batches."""
        return int(np.ceil(len(self.trials) / self.batch_size))

    def _on_epoch_end(self):
        if self.shuffle is True:
            self.shuffle_indexes()

    def shuffle_indexes(self):
        """Shuffles the data."""
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """The main function returning a single batch of the all data
        using such kind of call:
        (x_left, x_right, y) = data_generator[i]
        It returns the i-th batch of data.
        """
        start = index * self.batch_size
        if start + self.batch_size > len(self.trials):
            self._epoch_completed += 1
            self._remaining = len(self.trials) - start

            end = start + self._remaining
        else:
            end = start + self.batch_size
        indexes_tmp = self.indexes[start:end]
        return self.__data_generation(indexes_tmp)

    def __data_generation(self, indexes):
        """Fill the batch with data indexes respective trials."""
        if not self._remaining:
            x1_batch = np.empty((self.batch_size, self.input_size))
            x2_batch = np.empty((self.batch_size, self.input_size))
            y_batch = np.empty((self.batch_size), dtype=int)
        else:
            x1_batch = np.empty((self._remaining, self.input_size))
            x2_batch = np.empty((self._remaining, self.input_size))
            y_batch = np.empty((self._remaining), dtype=int)
            self._remaining = 0
        for k, i in enumerate(indexes):
            (x, y, t) = self.trials[i]
            x1_batch[k, ] = self.left_inputs[x]
            x2_batch[k, ] = self.right_inputs[y]
            y_batch[k] = t

        if self.mode == "train":
            return ([x1_batch, x2_batch], y_batch)
        else:
            return [x1_batch, x2_batch]
