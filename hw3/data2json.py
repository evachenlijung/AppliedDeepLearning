import json
import os
import argparse
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--filepath", type=str, required=True)
args = argparser.parse_args()

filepath = args.filepath

data_list = []
filepath = os.path.normpath(filepath)
dir_name = os.path.dirname(filepath)
file_name = os.path.splitext(os.path.basename(filepath))[0]

try:
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if line.strip():
                try:
                    json_object = json.loads(line)
                    data_list.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"警告：跳過無法解析的行：{line.strip()} (錯誤: {e})")

    if data_list:
        keys = data_list[0].keys()
    else:
        print("警告：輸入檔案中未找到任何有效的 JSON 物件。")
        
    output_filepath = os.path.join(dir_name, f"{file_name}.json")
    with open(output_filepath, "w", encoding='utf8') as outfile:
        json.dump(data_list, outfile, ensure_ascii=False, indent=2)

    print(f"\n成功將 {len(data_list)} 個物件格式化並寫入至 {output_filepath}")

except FileNotFoundError:
    print(f"錯誤：找不到輸入檔案 {filepath}。請確認路徑是否正確。")
except Exception as e:
    print(f"發生了一個意外錯誤: {e}")