#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: bash run.sh <context.json> <test.json> <prediction.csv>"
    exit 1
fi

CONTEXT_JSON="$1"
TEST_JSON="$2"
PREDICTION_CSV="$3"


accelerate launch code/gen_choice.py \
--model_name_or_path package/model_mc \
--tokenizer_name package/model_mc \
--context_file "$CONTEXT_JSON" \
--test_file "$TEST_JSON" \
--output_dir output/multichoice \
--max_seq_length 512 


accelerate launch code/qa_span.py \
--model_name_or_path package/model_qa \
--tokenizer_name package/model_qa \
--context_file package/data/context.json \
--test_file output/multichoice/test_prediction.json \
--output_dir "$PREDICTION_CSV" \
--max_seq_length 512 \
--do_predict