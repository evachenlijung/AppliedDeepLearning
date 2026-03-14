#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: bash ./run.sh /path/to/model_folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json"
    exit 1
fi

PATH_TO_MODEL_CHECKPOINT_FOLDER="$1"
PATH_TO_ADAPTER_CHECKPOINT="$2"
PATH_TO_INPUT_FILE="$3"
PATH_TO_OUTPUT_FILE="$4"

python inference.py \
    --path_to_model_checkpoint_folder "$PATH_TO_MODEL_CHECKPOINT_FOLDER" \
    --path_to_adapter_checkpoint_folder "$PATH_TO_ADAPTER_CHECKPOINT" \
    --test_data_path "$PATH_TO_INPUT_FILE" \
    --prediction_output_path "$PATH_TO_OUTPUT_FILE"