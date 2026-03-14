# Instruction Tuning (Traditional Chinese) with QLoRA
文言文、白話文互譯。</br>
Translation between Classical Chinese and Modern Chinese.

- Example 1
    - Instruction:
        - 翻譯成文言文：雅裏惱怒地說：從前在福山田獵時，你誣陷獵官，現在又說這種話。答案：
    - Output:
        - 雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。

- Example 2
    - Instruction:
        - 議雖不從，天下咸重其言。翻譯成白話文：
    - Output:
        - 他的建議雖然不被采納，但天下都很敬重他的話。



## Environment Setup
```
conda create -n adl_hw2_test python=3.10 -y
conda activate adl_hw2_test
pip install -r requirements.txt
```


## Download & Predict
```bash
bash download.sh
bash run.sh ./final_model adapter_checkpoint ./private_test.json ./output/r14922112_output.json
```
After execution, the prediction will be saved at:
`./output/r14922112_output.json`



## Fine-tune & Plot curves / Inference
```
# fine-tune QWen3-4B with QLoRA and plot learning curve
python main.py  # output under ./output/

# zero-shot inference
python main_no_qlora_0_shot.py  # output under ./output_no_qlora_0_shot/

# few-shot inference
python main_no_qlora_few_shot.py    # output under ./output_no_qlora_few_shot/

# fine-tune Llama-3.1-Taiwan-8B with QLoRA and plot learning curve
python main_llama_taiwan_8b.py  # output under ./output_llama_taiwan/
```