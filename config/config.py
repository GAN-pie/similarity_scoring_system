# coding: utf-8


"""
Configuration module
"""

class Config:
    backend = "feedforward"
    loss = "binary_crossentropy"
    input_size = 400
    hidden_size = 128
    metric = "euclidean"
    margin = 32.0

    train_trials_array = "data/array/train_1/trials.npy"
    train_en_array = "data/array/train_1/english_feats.npy"
    train_fr_array = "data/array/train_1/french_feats.npy"
    val_trials_array = "data/array/val_1/trials.npy"
    val_en_array = "data/array/val_1/english_feats.npy"
    val_fr_array = "data/array/val_1/french_feats.npy"

    test_trials_array = "data/array/test_1/trials.npy"
    test_en_array = "data/array/test_1/english_feats.npy"
    test_fr_array = "data/array/test_1/french_feats.npy"

    checkpoints_path = "checkpoints/1/"
    load_model_weights_path = "checkpoints/1/init_weights.h5"
    test_model_path = "checkpoints/1/model_checkpoint.h5"

    log_files_path = "logs/1"
    result_files_path = "results/1"

    mode = "train"

    train_batch_size = 24
    test_batch_size = 1024

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
