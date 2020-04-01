#!/usr/bin/env python3
# coding: utf-8

import sys
from os import path
import argparse
import random

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from utils import read_lst_file


class DualCorpus:
    """The DualCorpus class preserves the equivalence between english and
    french list. It allows to extract test entries given quantity constraints
    and allows to split the entries list into a train and dev folds."""

    def __init__(self, english, french, separator, fields=None):
        self._separator = separator
        self._fields = fields

        self._en_lst = read_lst_file(english)
        self._fr_lst = read_lst_file(french)
        self._mirrored = False

        self._true_labels = self.get_labels()
        self._apply_mirror()

    def display(self):
        if not self._mirrored:
            self._apply_mirror()

        for i in range(len(self._en_lst)):
            print("{} {}".format(self._en_lst[i], self._fr_lst[i]))

    def get_labels(self):
        return list(map(lambda x: x.split(self._separator)[1], self._en_lst))

    def get_unique_labels(self):
        labels = []
        for l in self._true_labels:
            if l not in labels:
                labels += [l]
        return labels

    def filter_quantity(self, q):
        tmp = {}
        for i in range(len(self._en_lst)):
            if tmp.get(self._true_labels[i]):
                tmp[self._true_labels[i]] += [self._en_lst[i]]
            else:
                tmp[self._true_labels[i]] = [self._en_lst[i]]

        new_en_lst = []
        for l in tmp.keys():
            if len(tmp[l]) >= q:
                new_en_lst += random.sample(tmp[l], q)

        random.shuffle(new_en_lst)
        self._en_lst = new_en_lst

        self._mirrored = False
        self._apply_mirror()

    def _apply_mirror(self):
        if self._mirrored is True:
            return

        fr_lst_id_map = {}
        for f in self._fr_lst:
            i = f.split(self._separator)[-1]
            fr_lst_id_map[i] = f

        new_fr_lst = []
        for e in self._en_lst:
            i = e.split(self._separator)[-1]
            new_fr_lst += [fr_lst_id_map[i]]

        self._fr_lst = new_fr_lst

        self._true_labels = self.get_labels()
        self._mirrored = True

    def train_test_separation(self, test_labels=[]):
        assert isinstance(test_labels, list)
        if not self._mirrored:
            self._apply_mirror()

        tmp = {}
        test_en = []
        test_fr = []
        for i in range(len(self._en_lst)):
            lb = self._true_labels[i]
            if lb in test_labels:
                test_en += [self._en_lst[i]]
                test_fr += [self._fr_lst[i]]
            else:
                if tmp.get(lb):
                    tmp[lb] += [self._en_lst[i]]
                else:
                    tmp[lb] = [self._en_lst[i]]

        new_en_lst = []
        for entries in tmp.values():
            new_en_lst += entries

        random.shuffle(new_en_lst)
        self._en_lst = new_en_lst

        self._mirrored = False
        self._apply_mirror()

        return test_en, test_fr

    def split_train_dev(self):
        if not self._mirrored:
            self._apply_mirror()

        le = LabelEncoder()
        encoded_labels = le.fit_transform(self._true_labels)

        indexes = np.arange(len(self._en_lst))

        # kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)
        kfold = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        for train_id, test_id in kfold.split(indexes, encoded_labels):

            train_indexes = indexes[train_id].tolist()
            test_indexes = indexes[test_id].tolist()

            train_en_lst = [self._en_lst[i] for i in train_indexes]
            train_fr_lst = [self._fr_lst[i] for i in train_indexes]

            test_en_lst = [self._en_lst[i] for i in test_indexes]
            test_fr_lst = [self._fr_lst[i] for i in test_indexes]

            yield train_en_lst, train_fr_lst, test_en_lst, test_fr_lst


def write_lst(fname, lst):
    with open(fname, "w") as fd:
        for x in lst:
            print(x, file=fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("english")
    parser.add_argument("french")
    parser.add_argument("--minimum", default=160, type=int)
    parser.add_argument("--field-separator", default=",", choices=[",", "."])
    parser.add_argument("--validation", default=False, action="store_true")
    parser.add_argument("--num-test-labels", "-n", default=4, type=int)
    parser.add_argument("--num-splits", default=4, type=int)
    parser.add_argument("--no-test", default=False, action="store_true")
    parser.add_argument("--output-dir", default="./")

    options = vars(parser.parse_args())

    sep = options["field_separator"]
    minimum = options["minimum"]
    validation = options["validation"]
    out_dir = options["output_dir"]
    n = options["num_test_labels"]
    n_splits = options["num_splits"]

    en_lst_path = options["english"]
    fr_lst_path = options["french"]

    dual_corpus = DualCorpus(en_lst_path, fr_lst_path, sep)
    dual_corpus.filter_quantity(minimum)
    labels = dual_corpus.get_unique_labels()

    if options["no_test"]:
        for train_en, train_fr, test_en, test_fr in dual_corpus.split_train_dev():

            write_lst(path.join(out_dir, "train_en.lst"), train_en)
            write_lst(path.join(out_dir, "train_fr.lst"), train_fr)

            write_lst(path.join(out_dir, "val_en.lst"), test_en)
            write_lst(path.join(out_dir, "val_fr.lst"), test_fr)
        sys.exit(0)

    if len(labels) % n != 0:
        print("ERROR: number of test labels mismatch {}/{} (the former must divides the latter).".format(n, len(labels)))
        sys.exit(1)

    total_n_splits = len(labels) / n

    if not n_splits <= total_n_splits:
        print("ERROR: can not set n_splits larger than possible splits ({}, {}).".format(
            n_splits, total_n_splits))
        sys.exit(1)

    random.shuffle(labels)
    cases_generator = (labels[i:i+n] for i in range(0, len(labels), n))

    del dual_corpus

    for i in range(1, n_splits+1):

        case = next(cases_generator)
        print("#{}: {}".format(i, " ".join(case)))

        # reset dual corpus object
        dual_corpus = DualCorpus(en_lst_path, fr_lst_path, sep)
        dual_corpus.filter_quantity(minimum)
        test_en, test_fr = dual_corpus.train_test_separation(test_labels=case)

        write_lst(path.join(out_dir, "test_en_"+str(i)+".lst"), test_en)
        write_lst(path.join(out_dir, "test_fr_"+str(i)+".lst"), test_fr)

        if validation:
            for train_en, train_fr, test_en, test_fr in dual_corpus.split_train_dev():

                write_lst(path.join(out_dir, "train_en_"+str(i)+".lst"), train_en)
                write_lst(path.join(out_dir, "train_fr_"+str(i)+".lst"), train_fr)

                write_lst(path.join(out_dir, "val_en_"+str(i)+".lst"), test_en)
                write_lst(path.join(out_dir, "val_fr_"+str(i)+".lst"), test_fr)
        else:
            raise NotImplementedError("no validation")
