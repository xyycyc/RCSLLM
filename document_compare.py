import os
import json
import filecmp


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in file]
        else:
            return json.load(file)


def compare_files(dir1, dir2):
    files1 = {f for f in os.listdir(dir1) if f.endswith(('.json', '.jsonl'))}
    files2 = {f for f in os.listdir(dir2) if f.endswith(('.json', '.jsonl'))}

    all_files = files1.union(files2)
    differences = []

    for file in all_files:
        path1 = os.path.join(dir1, file)
        path2 = os.path.join(dir2, file)

        if file not in files1:
            differences.append(f"{file} 在第一个文件夹中不存在")
        elif file not in files2:
            differences.append(f"{file} 在第二个文件夹中不存在")
        else:
            data1 = load_json(path1)
            data2 = load_json(path2)

            if data1 != data2:
                differences.append(f"{file} 文件不一致")

    return differences


# 指定两个目录路径
dir1 = r'LLaMA-Factory/saves/llama3-8b/lora_tw16/seed_control/seed=2024_label/base'
dir2 = r"D:\Desktop\resnet_FNN\finetune\LLaMA-Factory\saves\qwen-7b\lora\seed_control\seed=2024\checkpoint-500"
dir1 = r"D:\Desktop\resnet_FNN\finetune\LLaMA-Factory\saves\qwen-7b\lora\seed_control\seed=2024\checkpoint-640"

# 比较两个目录中的文件
differences = compare_files(dir1, dir2)
if not differences:
    print("两个文件夹中的所有JSON和JSONL文件完全一致。")
else:
    print("存在不一致的文件:")
    for diff in differences:
        print(diff)
