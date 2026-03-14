# Chinese Extractive Question Answering (QA)
- Task: Given 1 question and 4 paragraphs, find the answer span within the paragraphs.
- Breakdown: Multiple Choice + Span Selection



## Project Summary
| Item | Description |
| :--- | :--- |
| Task Type | Extractive QA |
| Input | 1 question + 4 paragraphs |
| Output | Answer span within paragraphs |
| Language | Traditional Chinese |
| Sub-tasks | Multiple Choice + Span Selection |
| Tokenizer & Pre-trained LM | hfl/chinese-lert-base ([LERT](https://arxiv.org/pdf/2211.05344)) |
| Evaluation Metric | Exact Match |



## Environment Setup
```
conda create -n adl_hw1_test python=3.10 -y
conda activate adl_hw1_test
pip install -r requirements.txt
```



## Quickstart
Run the following command to download the model and data, and generate prediction:  

```bash
bash ./download.sh
bash ./run.sh ./package/data/context.json ./package/data/test.json ./output/span_selection
```


## Output
After execution, the prediction will be saved as:
`./output/span_selection/prediction.csv`


## File structure:

    root/
    │
    └── package/
        |
        ├── model_mc/
        │
        ├── model_qa/
        │
        ├── data/
        |   |
        |   ├── context.json
        |   ├── test.json
        |   ├── train.json
        |   └── valid.json
        |
        └── code/

        

## Plot curves
To plot training loss and Exact Match metric value curve on validation set of span selection model:

```
accelerate launch code/qa_span_curve.py \
--model_name_or_path package/model_qa \
--tokenizer_name package/model_qa \
--context_file package/data/context.json \
--train_file package/data/train.json \
--validation_file package/data/valid.json \
--test_file package/data/test.json \
--max_seq_length 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--num_train_epochs 5 \
--learning_rate 3e-5 \
--output_dir output/span_selection_curve \
--with_tracking
```

And the figures will be under `output/span_selection_curve`

## Train from scratch on multiple-choice task
```
accelerate launch code/gen_choice_from_scratch.py \
--config_name code/mini_roberta_config.json \
--tokenizer_name roberta-base \
--context_file package/data/context.json \
--train_file package/data/train.json \
--validation_file package/data/valid.json \
--test_file package/data/test.json \
--max_seq_length 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-4 \
--num_train_epochs 20 \
--output_dir output/multichoice_from_scratch
```

The results would be under `output/multichoice_from_scratch`