# LLM Jailbreak
- Design a rewriting agent that rewrites toxic prompt to pass guqrd model and chat LLM to get useful response.

| Model | Name |
| :--- | :--- |
| Guard Model | [Qwen/Qwen3Guard-Gen-0.6B](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B) |
| Chat Model | [unsloth/Llama-3.2-3B-Instruct](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct) |
| Usefulness Judge | [Qwen3-1.7B-Usefulness-Judge](https://huggingface.co/theblackcat102/Qwen3-1.7B-Usefulness-Judge) |

## Download challenge data and setup environment.
```
git clone https://github.com/yenshan0530/2025-ADL-Final-Challenge-Release.git
conda create -n ADL-final python=3.12 -y
conda activate ADL-final
cd 2025-ADL-Final-Challenge-Release
pip install -r requirements.txt
```



## Model Preloading
Put `run.sh`, which is under `code/` folder, to `src/` and execute it under `src/` to load the models first.
```
cd src
bash run.sh
cd ..
```

## Rewriting
Just replace the original `src/algorithms.py` with our `algorithms.py` which is under `code/` folder, and run the commands below under the root directory (`2025-ADL-Final-Challenge-Release`) to get rewritten prompts for public dataset and private dataset, respectively.
```
python run_inference.py --dataset theblackcat102/ADL_Final_25W_part1_with_cost
python run_inference.py --dataset theblackcat102/ADL_Final_25W_part2_with_cost
``` 

## Results
We provide our results under `results/` folder.