#!/bin/sh
poetry run python3 finetune.py -train_file data/train_data_40k.csv \
-val_file data/val_data.csv \
-test_file data/test_data.csv \
-predict_file data/test_data.csv \
-fast_dev_run true


