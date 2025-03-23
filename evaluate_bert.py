from sklearn.metrics import accuracy_score, f1_score, classification_report
import json
from content_label_merge import ContentLabelMerge as clm
import re


def read_jsonl(file_path):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    epoch = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line.strip())
            weighted_avg_match = re.search(r"weighted avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", entry['report'])
            accuracy.append(float(re.search(r"accuracy\s+([\d.]+)",entry['report']).group(1)))
            precision.append(float(weighted_avg_match.group(1)))
            recall.append(float(weighted_avg_match.group(2)))
            f1.append(float(weighted_avg_match.group(3)))
            epoch.append(entry['epoch'])
    return accuracy, precision, recall, f1, epoch


dataprocess = clm()
model_name = r'LLAMA'
model_name = r'qwen'
evaluate = {}
index1 = [2024,2025,2026,2027,2028]
index2 = [['resnet','news'],['resnet','suggestion'],['resnet','resnet'],['news','suggestion'],['news','resnet'],['news','news'],['suggestion','news'],['suggestion','resnet'],['suggestion','suggestion']]
for seed in index1:
    for mode in index2:
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        evaluate_path = fr'dataset/exp_cache\seed={seed}\result\{mode[0]}_{mode[1]}_{model_name}_evaluate.json'
        print('seed and checkpoint:', seed, mode)
        accuracy, precision, recall, f1, epoch = read_jsonl(evaluate_path)
        for epoch in epoch:
            index = epoch - 1
            if seed not in evaluate:
                evaluate[seed] = {}
            if f'{mode[0]}_{mode[1]}' not in evaluate[seed]:
                evaluate[seed][f'{mode[0]}_{mode[1]}'] = {}
            if epoch not in evaluate[seed][f'{mode[0]}_{mode[1]}']:
                evaluate[seed][f'{mode[0]}_{mode[1]}'][epoch] = {}
            evaluate[seed][f'{mode[0]}_{mode[1]}'][epoch] = {
                "accuracy": accuracy[index],
                "precision": precision[index],
                'recall': recall[index],
                'f1': f1[index],
            }
            accuracy_list.append(float(accuracy[index]))
            f1_list.append(float(f1[index]))
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        average_f1 = sum(f1_list) / len(f1_list)
        print("Average accuracy:", average_accuracy)
        print("Average f1:", average_f1)
        print("Max accuracy:", max(accuracy_list))
        print("Max f1:", max(f1_list))
        print("Min accuracy:", min(accuracy_list))
        print("Min f1:", min(f1_list))
        print("\n")
