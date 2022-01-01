#!/bin/sh
poetry run python3 finetune.py -train_file data/ourcalls_train_data_dir.csv \
-val_file data/ourcalls_val_data_dir.csv \
-test_file data/ourcalls_test_data_dir.csv \
-predict_file data/ourcalls_test_data_dir.csv


