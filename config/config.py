#!/usr/bin/env python3
# coding: utf-8


"""
Configuration module
"""

import argparse


class Config:
    backend = "feedforward"
    loss = "binary_crossentropy"
    input_size = 400
    hidden_size = 128
    metric = "euclidean"
    margin = 32.0

    train_trials_array = "data/array/train_4/trials.npy"
    train_en_array = "data/array/train_4/english_feats.npy"
    train_fr_array = "data/array/train_4/french_feats.npy"
    val_trials_array = "data/array/val_4/trials.npy"
    val_en_array = "data/array/val_4/english_feats.npy"
    val_fr_array = "data/array/val_4/french_feats.npy"

    test_trials_array = "data/array/test_4/trials.npy"
    test_en_array = "data/array/test_4/english_feats.npy"
    test_fr_array = "data/array/test_4/french_feats.npy"

    checkpoints_path = "checkpoints/4/"
    load_model_weights_path = "checkpoints/4/init_weights.h5"
    test_model_path = "checkpoints/4/model_checkpoint.h5"

    log_files_path = "logs/4"
    result_files_path = "results/4"

    mode = "train"

    train_batch_size = 24
    test_batch_size = 128

    input_shape = (400,)

    optimizer = "adadelta"
    lr = 1.0
    lr_step = 10
    lr_decay = 0.95
    momentum = 0.8
    weight_decay = None

    use_gpu = True
    gpu_id = "0, 1"

    num_workers = 4
    print_freq = 5

    max_epoch = 50
    early_stopping = True

    plot_freq = 5

    def parse_command_line(self):
        command_parser = argparse.ArgumentParser()

        command_parser.add_argument("train-trials-array")
        command_parser.add_argument("train-en-array")
        command_parser.add_argument("train-fr-array")

        command_parser.add_argument("val-trials-array")
        command_parser.add_argument("val-en-array")
        command_parser.add_argument("val-fr-array")

        command_parser.add_argument("test-trials-array")
        command_parser.add_argument("test-en-array")
        command_parser.add_argument("test-fr-array")

        command_parser.add_argument("checkpoints-path")
        command_parser.add_argument("load_model_weights_path")
        command_parser.add_argument("test_model_path")
        command_parser.add_argument("log_files_path")
        command_parser.add_argument("result_files_path")

        command_parser.add_argument("--backend", default="feedforward")

        command_parser.add_argument("--loss", default="binary_crossentropy", choices=["binary_crossentropy", "lecun"])

        command_parser.add_argument("--metric", default="euclidean", choices=["euclidean", "manhattan"])

        command_parser.add_argument("--optimizer", default="adadelta", choices=["adadelta", "sgd"])

        command_parser.add_argument("--lr", default=1.0, type=float)

        command_parser.add_argument("--lr-step", default=10, type=int)

        command_parser.add_argument("--lr-decay", default=0.95, type=float)

        command_parser.add_argument("--weight-decay", default=None, type=float)


        use_gpu = command_parser.add_mutually_exclusive_group()

        use_gpu.add_argument("--use-gpu", action="store_const", dest="use_gpu", const=True, default=True)

        use_gpu.add_argument("--no-gpu", action="store_const", dest="use_gpu", const=False)


        command_parser.add_argument("--gpu-id", default="0, 1")

        command_parser.add_argument("--num-workers", default=4, type=int)

        command_parser.add_argument("--print-frep", default=5, type=int)

        command_parser.add_argument("--max-epoch", default=50, type=int)


        early_stopping = command_parser.add_mutually_exclusive_group()

        early_stopping.add_argument("--early-stopping", action="store_const", dest="early_stopping", const=True, default=True)

        early_stopping.add_argument("--no-early-stopping", action="store_const", dest="early_stopping", const=False)


        command_parser.add_argument("--input-size", default=400, type=int)

        command_parser.add_argument("--input-shape", dest="input_shape", default="(400,)", action=FormatInputShapeAction, required=True)

        command_parser.add_argument("--hidden-size", default="128", type=int)

        command_parser.add_argument("--margin", default=32.0, type=float)

        command_parser.add_argument("--mode", default="train", choices=["train", "test"])

        command_parser.add_argument("--train-batch-size", default=24, type=int)

        command_parser.add_argument("--test-batch-size", default=128, type=int)

        self.__dict__ = vars(command_parser.parse_args())


class FormatInputShapeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, eval(values))


if __name__ == "__main__":
    config = Config()

    config.parse_command_line()

    print(config.__dict__)
