# Retrieval-Augmented Generation (RAG)
- Fine-tune retriever & reranker
- Design prompt to optimize the generation performance



## Environment Setup
```
conda create -n adl_hw3 python=3.12 -y
conda activate adl_hw3
pip install -r requirements.txt
```

## Convert train.txt to train.json
```
python data2json.py --filepath ./data/train.txt
```
and the train.json will be saved at `./data/train.json`

## Finetune retriever
```
python retriever.py --retriever_dir retriever
```
and the finetuned retriever will be saved at `retriever_mnr_1`

## Save embeddings
```
python save_embeddings.py --retriever_model_path retriever_mnr_1 --build_db --batch_size 512
```

## Finetune reranker
```
python reranker.py --retriever_dir retriever_mnr_1 --reranker_dir reranker_hard_neg_1 
```
and the finetuned reranker will be saved at `reranker_hard_neg_1`

## Inference
```
python inference_batch.py --retriever_model_path retriever_mnr_1 --reranker_model_path reranker_hard_neg_1 --test_data_path ./data/test_open.txt
```

## Donwload models and inference
```
bash ./download.sh
python save_embeddings.py --retriever_model_path ./models/retriever --build_db
python inference_batch.py --retriever_model_path ./models/retriever --reranker_model_path ./models/reranker --test_data_path ./data/test_open.txt
```