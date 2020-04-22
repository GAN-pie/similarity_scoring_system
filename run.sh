#!/bin/bash


create_dir() {
    if [ -d "$1" ]; then
        if [ -d "$1.backup" ]; then
            rm -rf "$1.backup"
        fi
        mv "$1" "$1.backup"
    fi
    mkdir -p "$1"
}


set -e

# DEFINING DATA PATH

DATA_DIR="$(pwd)/data"

if [ ! -d "$DATA_DIR" ]; then echo "ERROR: no such file or directory: $DATA_DIR"; fi || exit 1;

# CHECK MANDATORY FILES EXISTENCE

required="$DATA_DIR/english.lst $DATA_DIR/french.lst $DATA_DIR/features.txt $DATA_DIR/meta.csv"
for f in $required; do
    if [ ! -f "$f" ];
    then
        echo "ERROR: no such file: $f"
        exit 1;
    fi
done

################################
# PREPARING MASSEFFECT3 CORPUS #
################################

# CREATING FEATURES AND IDENTIFIERS LIST SUBDIRECTORIES

LST_DIR="$DATA_DIR/lst"
ARRAY_DIR="$DATA_DIR/array"

if [ ! -d "$LST_DIR" ]; then
    mkdir "$LST_DIR"
fi

if [ ! -d "$ARRAY_DIR" ]; then
    mkdir "$ARRAY_DIR"
fi

# CREATING K-FOLD VALIDATION TRAIN VAL TEST IDENTIFIERS LIST

# python bin/split-corpus.py "$DATA_DIR/english.lst" "$DATA_DIR/french.lst" \
#     --output-dir $LST_DIR \
#     --validation --minimum 88 \
#     --field-separator "," \
#     --num-test-labels 4 \
#     --num-splits 4


# CREATING FEATURES VECTORS FOR EACH FOLDS

for k in `seq 1 4`;
do
    # TRAIN

    if [ -d "$ARRAY_DIR/train_$k" ]; then
        rm -rf "$ARRAY_DIR/train_$k"
    fi
    mkdir -p "$ARRAY_DIR/train_$k"

    python bin/make-trials.py "$LST_DIR/train_en_$k.lst" "$LST_DIR/train_fr_$k.lst" \
        "$DATA_DIR/features.txt" "$ARRAY_DIR/train_$k" \
        --meta-data "$DATA_DIR/meta.csv" --separator "," \
        --balance --verbose > "$LST_DIR/train_trials_${k}.lst" &

    # VALIDATION

    if [ -d "$ARRAY_DIR/val_$k" ]; then
        rm -rf "$ARRAY_DIR/val_$k"
    fi
    mkdir -p "$ARRAY_DIR/val_$k"

    python bin/make-trials.py "$LST_DIR/val_en_$k.lst" "$LST_DIR/val_fr_$k.lst" \
        "$DATA_DIR/features.txt" "$ARRAY_DIR/val_$k" \
        --meta-data "$DATA_DIR/meta.csv" --separator "," \
        --balance --verbose > "$LST_DIR/val_trials_${k}.lst" &

    # TEST

    if [ -d "$ARRAY_DIR/test_$k" ]; then
        rm -rf "$ARRAY_DIR/test_$k"
    fi
    mkdir -p "$ARRAY_DIR/test_$k"

    python bin/make-trials.py "$LST_DIR/test_en_$k.lst" "$LST_DIR/test_fr_$k.lst" \
        "$DATA_DIR/features.txt" "$ARRAY_DIR/test_$k" \
        --meta-data "$DATA_DIR/meta.csv" --separator "," \
        --balance --verbose > "$LST_DIR/test_trials_${k}.lst" &
done

wait

# TRAINING AND SCORING


for k in `seq 1 4`;
do
    # CREATING MODEL/LOG/RESULT DIRECTORIES

    MDL_DIR="checkpoints/$k"
    LOG_DIR="logs/$k"
    RES_DIR="results/$k"

    create_dir $MDL_DIR
    create_dir $LOG_DIR
    create_dir $RES_DIR


    # TRAINING

    # export CUDA_VISIBLE_DEVICES=1
    python train.py \
        "${ARRAY_DIR}/train_${k}/trials.npy" "${ARRAY_DIR}/train_${k}/english_feats.npy" "${ARRAY_DIR}/train_${k}/french_feats.npy" \
        "${ARRAY_DIR}/val_${k}/trials.npy" "${ARRAY_DIR}/val_${k}/english_feats.npy" "${ARRAY_DIR}/val_${k}/french_feats.npy" \
        "${ARRAY_DIR}/test_${k}/trials.npy" "${ARRAY_DIR}/test_${k}/english_feats.npy" "${ARRAY_DIR}/test_${k}/french_feats.npy" \
        "${MDL_DIR}" "${MDL_DIR}/init_weights.h5" "${MDL_DIR}/model_checkpoint.h5" \
        "${LOG_DIR}" "${RES_DIR}" \
        --use-gpu --gpu-id "1" \
        --input-shape "(400,)" --input-size 400 \
        --train-batch-size 24 --max-epoch 100 --early-stopping \
        --loss "lecun" --margin 30.0 --metric "euclidean"


    # SCORING

    # export CUDA_VISIBLE_DEVICES=""
    python test.py \
        "${ARRAY_DIR}/train_${k}/trials.npy" "${ARRAY_DIR}/train_${k}/english_feats.npy" "${ARRAY_DIR}/train_${k}/french_feats.npy" \
        "${ARRAY_DIR}/val_${k}/trials.npy" "${ARRAY_DIR}/val_${k}/english_feats.npy" "${ARRAY_DIR}/val_${k}/french_feats.npy" \
        "${ARRAY_DIR}/test_${k}/trials.npy" "${ARRAY_DIR}/test_${k}/english_feats.npy" "${ARRAY_DIR}/test_${k}/french_feats.npy" \
        "${MDL_DIR}" "${MDL_DIR}/init_weights.h5" "${MDL_DIR}/model_checkpoint.h5" \
        "${LOG_DIR}" "${RES_DIR}" \
        --no-gpu \
        --input-shape "(400,)" --input-size 400 \
        --train-batch-size 24 --max-epoch 100 --early-stopping \
        --loss "lecun" --margin 30.0 --metric "euclidean"
done

echo "Done!"
