#!/usr/bin/env python3
# coding: utf-8

import argparse
import csv
import itertools
import os
from os import path
import math
import concurrent
from concurrent import futures

import numpy as np

from utils import read_lst_file, read_data_file


def read_meta_file(fname, corpus="masseffect"):
    """Returns dict containing information for characters."""
    assert isinstance(fname, str)
    with open(fname, "r") as csv_f:
        infos = {}
        spamreader = csv.reader(csv_f, delimiter=",", quotechar="\"")

        if corpus == "masseffect":
            for row in spamreader:
                infos[row[0]] = {
                    "qte": row[1],
                    "gender": row[2],
                    "race": row[3],
                    "ctg": row[4],
                    "actor": row[5]}
        elif corpus == "skyrim":
            for row in spamreader:
                infos[row[0]] = {
                    "qte": row[1],
                    "gender": row[2]}
        return infos


def balance(trials, shuffle=True):
    """Returns a balanced array with same number of target and non-target trials."""
    assert isinstance(trials, list)
    trials = np.stack(trials, axis=0)
    sim_trials = trials[np.where(trials[:, 2] == 0)]
    dis_trials = trials[np.where(trials[:, 2] == 1)]

    num_sim, num_dis = len(sim_trials), len(dis_trials)
    if num_dis > num_sim:
        ids_dis_trials = np.arange(num_dis)
        rand_ids_dis_trials = np.random.choice(ids_dis_trials, size=num_sim, replace=False)
        dis_trials = dis_trials[rand_ids_dis_trials]
        trials = np.concatenate([dis_trials, sim_trials])
    elif num_dis < num_sim:
        ids_sim_trials = np.arange(num_sim)
        rand_ids_sim_trials = np.random.choice(ids_sim_trials, size=num_dis, replace=False)
        sim_trials = sim_trials[rand_ids_sim_trials]
        trials = np.concatenate([sim_trials, dis_trials])

    if shuffle:
        np.random.shuffle(trials)
    return trials

# def init_trials(english_l, french_l, info_corpus=None, balancing=True, shuffle=True, sep=","):
#     """Creates an array of trials given two list of entries. Extra information
#     can be passed to function with info_corpus argument as a dict. The way entry label are
#     processed is defined by the sep argument. The function returns and balanced and/or
#     shuffled array of trials."""
#     assert isinstance(english_l, list) and isinstance(french_l, list)
#     assert isinstance(sep, str)
#     assert isinstance(info_corpus, dict) or info_corpus is None

#     english_charnames = [ instance.split(sep)[1] for instance in english_l ]
#     french_charnames = [ instance.split(sep)[1] for instance in french_l ]

#     if info_corpus:
#         en_gender_l = list(map(lambda x : info_corpus[x]['gender'], english_charnames))
#         fr_gender_l = list(map(lambda x : info_corpus[x]['gender'], french_charnames))

#     pair_producer = itertools.product(range(len(english_l)), range(len(french_l)))
#     trials = []
#     for (i, j) in pair_producer:

#         if info_corpus and en_gender_l[i] != fr_gender_l[j]:
#             continue

#         t = 1 if english_charnames[i] == french_charnames[j] else 0
#         trials += [ (i, j, t) ]

#     if balancing:
#         return balance(trials, shuffle)
#     else:
#         trials = np.stack(trials, axis=0)
#         if shuffle:
#             np.random.shuffle(trials)
#         return trials


class TrialsMaker:
    """Creating trials array using multiprocessing."""

    def __init__(self, en_lst, fr_lst, meta=None, balancing=True, shuffle=True, sep=","):
        self.en_lst = en_lst
        self.fr_lst = fr_lst
        self.meta = meta
        self.balancing = balancing
        self.shuffle = shuffle
        self.sep = sep

        self.en_chars = [x.split(sep)[1] for x in self.en_lst]    # extract character label
        self.fr_chars = [x.split(sep)[1] for x in self.fr_lst]

        if self.meta:
            def get_gender_fn(x): return self.meta[x]["gender"]
            self.en_gdr = list(map(get_gender_fn, self.en_chars))
            self.fr_gdr = list(map(get_gender_fn, self.fr_chars))

    def _job(self, pair):
        (i, j) = pair
        # apply gender constraint if any
        if self.meta and self.en_gdr[i] != self.fr_gdr[j]:
            return None

        t = 0 if self.en_chars[i] == self.fr_chars[j] else 1
        return (i, j, t)

    def make(self):
        pairs = itertools.product(range(len(self.en_lst)), range(len(self.fr_lst)))

        chuncks = math.floor((len(self.en_lst) * len(self.fr_lst)) / os.cpu_count())

        trials = []
        with futures.ProcessPoolExecutor(os.cpu_count()) as executor:
            for trial in executor.map(self._job, pairs, chunksize=chuncks):
                if trial:
                    trials += [trial]

        if self.balancing:
            return balance(trials, self.shuffle)
        else:
            trials = np.stack(trials, axis=0)
            if self.shuffle:
                np.random.shuffle(trials)
            return trials


if __name__ == "__main__":
    options_parser = argparse.ArgumentParser("Create trials file from english and french voice utterances.")
    options_parser.add_argument("english", help="A .lst file containing english utterances.")
    options_parser.add_argument("french", help="A .lst file containing french utterances.")
    options_parser.add_argument("features", help="A .txt file containing features vector for all uttenrances.")

    options_parser.add_argument("output-directory", help="Defines directory where trials files will be stored.")
    options_parser.add_argument("--meta-data", help="A .csv file containing meta information on voices.")
    options_parser.add_argument("--separator", default=",", help="Defines the field separator.")
    options_parser.add_argument("--balance", default=False, action="store_true", help="Define whether or not trials must be balanced.")
    options_parser.add_argument("--verbose", "-v", default=False, action="store_true", help="Display trials.")

    options = vars(options_parser.parse_args())
    output_dir = options["output-directory"]
    sep = options["separator"]
    balancing = options["balance"]
    verbose = options["verbose"]

    english_lst = read_lst_file(options["english"])
    french_lst = read_lst_file(options["french"])
    features = read_data_file(options["features"])

    meta = None
    if options["meta_data"]:
        meta = read_meta_file(options["meta_data"], corpus="masseffect" if sep == "," else "skyrim")

    # trials = init_trials(english_lst, french_lst, meta, balancing, True, sep=sep)
    maker = TrialsMaker(english_lst, french_lst, meta=meta, balancing=balancing, shuffle=True, sep=sep)
    trials = maker.make()

    if verbose:
        for trial in trials:
            print("{} {} {}".format(english_lst[trial[0]], french_lst[trial[1]], "nontarget" if trial[2] == 1 else "target"))

    trials_path = path.join(output_dir, "trials.npy")
    np.save(trials_path, trials)

    english_features = np.array([features[x] for x in english_lst])
    french_features = np.array([features[x] for x in french_lst])

    english_feat_path = path.join(output_dir, "english_feats.npy")
    np.save(english_feat_path, english_features)

    french_feat_path = path.join(output_dir, "french_feats.npy")
    np.save(french_feat_path, french_features)
