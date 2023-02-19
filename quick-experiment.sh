#!/bin/bash
#
# quick access to ./experiments/EXAMPLE_gpt.yml most important settings

NAME_EXPERIMENT="dud"
NAME_DATASET="dud-dataset"
PATH_TRAINING="/tmp/train/metadata.csv"
NAME_VALIDATION="dud-validation"
PATH_VALIDATION="/tmp/val/metadata.csv"

# NAME_EXPERIMENT="your_experiment_name"
# NAME_DATASET="training_dataset_name"
# PATH_TRAINING="path_to_training_dataset"
# NAME_VALIDATION="validation_dataset_name"
# PATH_VALIDATION="path_to_validation_dataset"



# YOU SHOULD DIRECTLY PLAY WITH THIS FILE
QUICK="./experiments/EXAMPLE_gpt.yml"


cp "./experiments/EXAMPLE_gpt_raw.yml" "./experiments/EXAMPLE_gpt.yml"

sed -i "s|CHANGEME_your_experiment_name|$NAME_EXPERIMENT|" $QUICK
sed -i "s|CHANGEME_training_dataset_name|$NAME_DATASET|" $QUICK
sed -i "s|CHANGEME_path_to_training_dataset|$PATH_TRAINING|" $QUICK
sed -i "s|CHANGEME_validation_dataset_name|$NAME_VALIDATION|" $QUICK
sed -i "s|CHANGEME_path_to_validation_dataset|$PATH_VALIDATION|" $QUICK
