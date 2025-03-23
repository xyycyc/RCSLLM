import json
import re
from content_label_merge import ContentLabelMerge as clm
import numpy as np

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
            accuracy.append(float(re.search(r"accuracy\s+([\d.]+)", entry['report']).group(1)))
            precision.append(float(weighted_avg_match.group(1)))
            recall.append(float(weighted_avg_match.group(2)))
            f1.append(float(weighted_avg_match.group(3)))
            epoch.append(entry['epoch'])
            # print(weighted_avg_match)
            # print(re.search(r"accuracy\s+([\d.]+)", entry['report']))
    return accuracy, precision, recall, f1, epoch

def find_best_metrics(data):
    result = {}
    for first_level_key, first_level_value in data.items():
        best_metrics = {
            'accuracy': {'avg': float('-inf'), 'std': None, 'subdirectory': None},
            'precision': {'avg': float('-inf'), 'std': None, 'subdirectory': None},
            'recall': {'avg': float('-inf'), 'std': None, 'subdirectory': None},
            'f1': {'avg': float('-inf'), 'std': None, 'subdirectory': None}
        }

        for second_level_key, second_level_value in first_level_value.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                avg_metric = second_level_value[metric]['avg_min_max'][0]
                std_metric = second_level_value[metric]['std']

                if avg_metric > best_metrics[metric]['avg']:
                    best_metrics[metric] = {
                        'avg': avg_metric,
                        'std': std_metric,
                        'subdirectory': second_level_key
                    }

        result[first_level_key] = best_metrics
    return result

def get_max_i_average(values, i):
    # 取最大的 i 个数的平均值
    max_i_values = sorted(values, reverse=True)[:i]
    return np.mean(max_i_values)


with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)

# file_path = fr"result/{hyper_parameters['dataset_name']}_results.json"
# evaluate_method = 'max'
evaluate_method = 'average'
if evaluate_method == 'max':
    # file_path = fr"result/{hyper_parameters['dataset_name']}_results_max.json"
    file_path = fr"result/{hyper_parameters['dataset_name']}_results_max.json"
else:
    file_path = fr"result/{hyper_parameters['dataset_name']}_results.json"
dataprocesser = clm()
compare_path = fr"dataset/exp_cache/{hyper_parameters['dataset_name']}_llm.json"
llm_path = [fr"result\lora_seed_times_test_llama.json",
            fr"result\lora_seed_times_test_qwen.json"]
# llm_path = [fr"dataset\exp_cache\lora_seed_times_test_llama.json",
#             fr"dataset\exp_cache\lora_seed_times_test_qwen.json"]
index1 = [2024, 2025, 2026, 2027, 2028]
index2 = [['resnet', 'news'], ['resnet', 'suggestion'], ['resnet', 'resnet'], ['news', 'suggestion'],
          ['news', 'resnet'], ['news', 'news'], ['suggestion', 'news'], ['suggestion', 'resnet'],
          ['suggestion', 'suggestion']]
i = 3
error_threshold = 0.3
error_threshold_llm = 0.5
name = ['LLAMA', 'QWEN', 'LLAMA_lora', 'QWEN_lora']
data_component = {
    f'{name[0]}': {},
    f'{name[1]}': {},
    f'{name[2]}': {},
    f'{name[3]}': {}
}
for model_name in name:
    if model_name == 'LLAMA' and evaluate_method == r'average':
        print('LLAMA')
        for mode in index2:
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            error_list = []
            for seed in index1:
                slm_llm_path = fr'dataset/exp_cache\seed={seed}\result\{mode[0]}_{mode[1]}_{model_name}_evaluate.json'
                accuracy, precision, recall, f1, epoch = read_jsonl(slm_llm_path)
                end_accuracy = np.mean(accuracy[-i:])
                end_precision = np.mean(precision[-i:])
                end_recall = np.mean(recall[-i:])
                end_f1 = np.mean(f1[-i:])

                if end_accuracy > error_threshold:
                    accuracy_list.append(end_accuracy)
                    precision_list.append(end_precision)
                    recall_list.append(end_recall)
                    f1_list.append(end_f1)
                else:
                    error_list.append([end_accuracy, seed])

            average_accuracy = np.mean(accuracy_list)
            std_accuracy = np.std(accuracy_list)
            min_accuracy = np.min(accuracy_list)
            max_accuracy = np.max(accuracy_list)

            average_precision = np.mean(precision_list)
            std_precision = np.std(precision_list)
            min_precision = np.min(precision_list)
            max_precision = np.max(precision_list)

            average_recall = np.mean(recall_list)
            std_recall = np.std(recall_list)
            min_recall = np.min(recall_list)
            max_recall = np.max(recall_list)

            average_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list)
            min_f1 = np.min(f1_list)
            max_f1 = np.max(f1_list)

            if f'{mode[0]}_{mode[1]}' not in data_component[model_name]:
                data_component[model_name][f'{mode[0]}_{mode[1]}'] = {
                    'wrong list': error_list,
                    'accuracy': {
                        'right list': accuracy_list,
                        'avg_min_max': [round(average_accuracy, 4), round(min_accuracy, 4), round(max_accuracy, 4)],
                        'updif and lowfid': [round(max_accuracy - average_accuracy, 4),
                                             round(average_accuracy - min_accuracy, 4)],
                        'std': round(std_accuracy, 4)
                    },
                    'precision': {
                        'right list': precision_list,
                        'avg_min_max': [round(average_precision, 4), round(min_precision, 4), round(max_precision, 4)],
                        'updif and lowfid': [round(max_precision - average_precision, 4),
                                             round(average_precision - min_precision, 4)],
                        'std': round(std_precision, 4)
                    },
                    'recall': {
                        'right list': recall_list,
                        'avg_min_max': [round(average_recall, 4), round(min_recall, 4), round(max_recall, 4)],
                        'updif and lowfid': [round(max_recall - average_recall, 4),
                                             round(average_recall - min_recall, 4)],
                        'std': round(std_recall, 4)
                    },
                    'f1': {
                        'right list': f1_list,
                        'avg_min_max': [round(average_f1, 4), round(min_f1, 4), round(max_f1, 4)],
                        'updif and lowfid': [round(max_f1 - average_f1, 4), round(average_f1 - min_f1, 4)],
                        'std': round(std_f1, 4)
                    }
                }

    if model_name == 'QWEN' and evaluate_method == r'average':
        print('QWEN')
        for mode in index2:
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            error_list = []
            for seed in index1:
                slm_llm_path = fr'dataset/exp_cache\seed={seed}\result\{mode[0]}_{mode[1]}_{model_name}_evaluate.json'
                accuracy, precision, recall, f1, epoch = read_jsonl(slm_llm_path)
                end_accuracy = np.mean(accuracy[-i:])
                end_precision = np.mean(precision[-i:])
                end_recall = np.mean(recall[-i:])
                end_f1 = np.mean(f1[-i:])

                if end_accuracy > error_threshold:
                    accuracy_list.append(end_accuracy)
                    precision_list.append(end_precision)
                    recall_list.append(end_recall)
                    f1_list.append(end_f1)
                else:
                    error_list.append([end_accuracy, seed])

            average_accuracy = np.mean(accuracy_list)
            std_accuracy = np.std(accuracy_list)
            min_accuracy = np.min(accuracy_list)
            max_accuracy = np.max(accuracy_list)

            average_precision = np.mean(precision_list)
            std_precision = np.std(precision_list)
            min_precision = np.min(precision_list)
            max_precision = np.max(precision_list)

            average_recall = np.mean(recall_list)
            std_recall = np.std(recall_list)
            min_recall = np.min(recall_list)
            max_recall = np.max(recall_list)

            average_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list)
            min_f1 = np.min(f1_list)
            max_f1 = np.max(f1_list)

            if f'{mode[0]}_{mode[1]}' not in data_component[model_name]:
                data_component[model_name][f'{mode[0]}_{mode[1]}'] = {
                    'wrong list': error_list,
                    'accuracy': {
                        'right list': accuracy_list,
                        'avg_min_max': [round(average_accuracy, 4), round(min_accuracy, 4), round(max_accuracy, 4)],
                        'updif and lowfid': [round(max_accuracy - average_accuracy, 4),
                                             round(average_accuracy - min_accuracy, 4)],
                        'std': round(std_accuracy, 4)
                    },
                    'precision': {
                        'right list': precision_list,
                        'avg_min_max': [round(average_precision, 4), round(min_precision, 4), round(max_precision, 4)],
                        'updif and lowfid': [round(max_precision - average_precision, 4),
                                             round(average_precision - min_precision, 4)],
                        'std': round(std_precision, 4)
                    },
                    'recall': {
                        'right list': recall_list,
                        'avg_min_max': [round(average_recall, 4), round(min_recall, 4), round(max_recall, 4)],
                        'updif and lowfid': [round(max_recall - average_recall, 4),
                                             round(average_recall - min_recall, 4)],
                        'std': round(std_recall, 4)
                    },
                    'f1': {
                        'right list': f1_list,
                        'avg_min_max': [round(average_f1, 4), round(min_f1, 4), round(max_f1, 4)],
                        'updif and lowfid': [round(max_f1 - average_f1, 4), round(average_f1 - min_f1, 4)],
                        'std': round(std_f1, 4)
                    }
                }

    if model_name == 'LLAMA' and evaluate_method == 'max':
        print('LLAMA')
        for mode in index2:
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            error_list = []
            for seed in index1:
                slm_llm_path = fr'dataset/exp_cache\seed={seed}\result\{mode[0]}_{mode[1]}_{model_name}_evaluate.json'
                accuracy, precision, recall, f1, epoch = read_jsonl(slm_llm_path)

                # 修改后的逻辑：取最大的 i 个数的平均值
                end_accuracy = get_max_i_average(accuracy, i)
                end_precision = get_max_i_average(precision, i)
                end_recall = get_max_i_average(recall, i)
                end_f1 = get_max_i_average(f1, i)

                if end_accuracy > error_threshold:
                    accuracy_list.append(end_accuracy)
                    precision_list.append(end_precision)
                    recall_list.append(end_recall)
                    f1_list.append(end_f1)
                else:
                    error_list.append([end_accuracy, seed])

            average_accuracy = np.mean(accuracy_list)
            std_accuracy = np.std(accuracy_list)
            min_accuracy = np.min(accuracy_list)
            max_accuracy = np.max(accuracy_list)

            average_precision = np.mean(precision_list)
            std_precision = np.std(precision_list)
            min_precision = np.min(precision_list)
            max_precision = np.max(precision_list)

            average_recall = np.mean(recall_list)
            std_recall = np.std(recall_list)
            min_recall = np.min(recall_list)
            max_recall = np.max(recall_list)

            average_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list)
            min_f1 = np.min(f1_list)
            max_f1 = np.max(f1_list)

            if f'{mode[0]}_{mode[1]}' not in data_component[model_name]:
                data_component[model_name][f'{mode[0]}_{mode[1]}'] = {
                    'wrong list': error_list,
                    'accuracy': {
                        'right list': accuracy_list,
                        'avg_min_max': [round(average_accuracy, 4), round(min_accuracy, 4), round(max_accuracy, 4)],
                        'updif and lowfid': [round(max_accuracy - average_accuracy, 4),
                                             round(average_accuracy - min_accuracy, 4)],
                        'std': round(std_accuracy, 4)
                    },
                    'precision': {
                        'right list': precision_list,
                        'avg_min_max': [round(average_precision, 4), round(min_precision, 4), round(max_precision, 4)],
                        'updif and lowfid': [round(max_precision - average_precision, 4),
                                             round(average_precision - min_precision, 4)],
                        'std': round(std_precision, 4)
                    },
                    'recall': {
                        'right list': recall_list,
                        'avg_min_max': [round(average_recall, 4), round(min_recall, 4), round(max_recall, 4)],
                        'updif and lowfid': [round(max_recall - average_recall, 4),
                                             round(average_recall - min_recall, 4)],
                        'std': round(std_recall, 4)
                    },
                    'f1': {
                        'right list': f1_list,
                        'avg_min_max': [round(average_f1, 4), round(min_f1, 4), round(max_f1, 4)],
                        'updif and lowfid': [round(max_f1 - average_f1, 4), round(average_f1 - min_f1, 4)],
                        'std': round(std_f1, 4)
                    }
                }

    if model_name == 'QWEN' and evaluate_method == 'max':
        print('QWEN')
        for mode in index2:
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            error_list = []
            for seed in index1:
                slm_llm_path = fr'dataset/exp_cache\seed={seed}\result\{mode[0]}_{mode[1]}_{model_name}_evaluate.json'
                accuracy, precision, recall, f1, epoch = read_jsonl(slm_llm_path)

                # 修改后的逻辑：取最大的 i 个数的平均值
                end_accuracy = get_max_i_average(accuracy, i)
                end_precision = get_max_i_average(precision, i)
                end_recall = get_max_i_average(recall, i)
                end_f1 = get_max_i_average(f1, i)

                if end_accuracy > error_threshold:
                    accuracy_list.append(end_accuracy)
                    precision_list.append(end_precision)
                    recall_list.append(end_recall)
                    f1_list.append(end_f1)
                else:
                    error_list.append([end_accuracy, seed])

            average_accuracy = np.mean(accuracy_list)
            std_accuracy = np.std(accuracy_list)
            min_accuracy = np.min(accuracy_list)
            max_accuracy = np.max(accuracy_list)

            average_precision = np.mean(precision_list)
            std_precision = np.std(precision_list)
            min_precision = np.min(precision_list)
            max_precision = np.max(precision_list)

            average_recall = np.mean(recall_list)
            std_recall = np.std(recall_list)
            min_recall = np.min(recall_list)
            max_recall = np.max(recall_list)

            average_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list)
            min_f1 = np.min(f1_list)
            max_f1 = np.max(f1_list)

            if f'{mode[0]}_{mode[1]}' not in data_component[model_name]:
                data_component[model_name][f'{mode[0]}_{mode[1]}'] = {
                    'wrong list': error_list,
                    'accuracy': {
                        'right list': accuracy_list,
                        'avg_min_max': [round(average_accuracy, 4), round(min_accuracy, 4), round(max_accuracy, 4)],
                        'updif and lowfid': [round(max_accuracy - average_accuracy, 4),
                                             round(average_accuracy - min_accuracy, 4)],
                        'std': round(std_accuracy, 4)
                    },
                    'precision': {
                        'right list': precision_list,
                        'avg_min_max': [round(average_precision, 4), round(min_precision, 4), round(max_precision, 4)],
                        'updif and lowfid': [round(max_precision - average_precision, 4),
                                             round(average_precision - min_precision, 4)],
                        'std': round(std_precision, 4)
                    },
                    'recall': {
                        'right list': recall_list,
                        'avg_min_max': [round(average_recall, 4), round(min_recall, 4), round(max_recall, 4)],
                        'updif and lowfid': [round(max_recall - average_recall, 4),
                                             round(average_recall - min_recall, 4)],
                        'std': round(std_recall, 4)
                    },
                    'f1': {
                        'right list': f1_list,
                        'avg_min_max': [round(average_f1, 4), round(min_f1, 4), round(max_f1, 4)],
                        'updif and lowfid': [round(max_f1 - average_f1, 4), round(average_f1 - min_f1, 4)],
                        'std': round(std_f1, 4)
                    }
                }


    if model_name == 'LLAMA_lora':
        print('LLAMA_lora')
        with open(llm_path[0], 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        accuracy_list_ck = []
        precision_list_ck = []
        recall_list_ck = []
        f1_list_ck = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        error_list = []
        for seed in index1:
            for _, item in llm_data.items():
                for key in item:
                    if llm_data[f'{seed}'][key]['accuracy'] > error_threshold_llm:
                        accuracy_list_ck.append(llm_data[f'{seed}'][key]['accuracy'])
                        precision_list_ck.append(llm_data[f'{seed}'][key].get('precision', 0))
                        recall_list_ck.append(llm_data[f'{seed}'][key].get('recall', 0))
                        f1_list_ck.append(llm_data[f'{seed}'][key].get('f1', 0))
            end_accuracy = np.mean(accuracy_list_ck)
            end_precision = np.mean(precision_list_ck)
            end_recall = np.mean(recall_list_ck)
            end_f1 = np.mean(f1_list_ck)
            if end_accuracy > error_threshold:
                accuracy_list.append(end_accuracy)
                precision_list.append(end_precision)
                recall_list.append(end_recall)
                f1_list.append(end_f1)
            else:
                error_list.append([end_accuracy, seed])
        average_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        min_accuracy = np.min(accuracy_list)
        max_accuracy = np.max(accuracy_list)

        average_precision = np.mean(precision_list)
        std_precision = np.std(precision_list)
        min_precision = np.min(precision_list)
        max_precision = np.max(precision_list)

        average_recall = np.mean(recall_list)
        std_recall = np.std(recall_list)
        min_recall = np.min(recall_list)
        max_recall = np.max(recall_list)

        average_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        min_f1 = np.min(f1_list)
        max_f1 = np.max(f1_list)

        if f'{model_name}_prediction' not in data_component[model_name]:
            data_component[model_name][f'{model_name}_prediction'] = {
                'wrong list': error_list,
                'accuracy': {
                    'right list': accuracy_list,
                    'avg_min_max': [round(average_accuracy, 4), round(min_accuracy, 4), round(max_accuracy, 4)],
                    'updif and lowfid': [round(max_accuracy - average_accuracy, 4),
                                         round(average_accuracy - min_accuracy, 4)],
                    'std': round(std_accuracy, 4)
                },
                'precision': {
                    'right list': precision_list,
                    'avg_min_max': [round(average_precision, 4), round(min_precision, 4), round(max_precision, 4)],
                    'updif and lowfid': [round(max_precision - average_precision, 4),
                                         round(average_precision - min_precision, 4)],
                    'std': round(std_precision, 4)
                },
                'recall': {
                    'right list': recall_list,
                    'avg_min_max': [round(average_recall, 4), round(min_recall, 4), round(max_recall, 4)],
                    'updif and lowfid': [round(max_recall - average_recall, 4),
                                         round(average_recall - min_recall, 4)],
                    'std': round(std_recall, 4)
                },
                'f1': {
                    'right list': f1_list,
                    'avg_min_max': [round(average_f1, 4), round(min_f1, 4), round(max_f1, 4)],
                    'updif and lowfid': [round(max_f1 - average_f1, 4), round(average_f1 - min_f1, 4)],
                    'std': round(std_f1, 4)
                }
            }

    if model_name == 'QWEN_lora':
        print('QWEN_lora')
        with open(llm_path[1], 'r', encoding='utf-8') as f:
            llm_data = json.load(f)
        accuracy_list_ck = []
        precision_list_ck = []
        recall_list_ck = []
        f1_list_ck = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        error_list = []
        for seed in index1:
            for _, item in llm_data.items():
                for key in item:
                    if llm_data[f'{seed}'][key]['accuracy'] > error_threshold_llm:
                        accuracy_list_ck.append(llm_data[f'{seed}'][key]['accuracy'])
                        precision_list_ck.append(llm_data[f'{seed}'][key].get('precision', 0))
                        recall_list_ck.append(llm_data[f'{seed}'][key].get('recall', 0))
                        f1_list_ck.append(llm_data[f'{seed}'][key].get('f1', 0))
            end_accuracy = np.mean(accuracy_list_ck)
            end_precision = np.mean(precision_list_ck)
            end_recall = np.mean(recall_list_ck)
            end_f1 = np.mean(f1_list_ck)
            if end_accuracy > error_threshold:
                accuracy_list.append(end_accuracy)
                precision_list.append(end_precision)
                recall_list.append(end_recall)
                f1_list.append(end_f1)
            else:
                error_list.append([end_accuracy, seed])
        average_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        min_accuracy = np.min(accuracy_list)
        max_accuracy = np.max(accuracy_list)

        average_precision = np.mean(precision_list)
        std_precision = np.std(precision_list)
        min_precision = np.min(precision_list)
        max_precision = np.max(precision_list)

        average_recall = np.mean(recall_list)
        std_recall = np.std(recall_list)
        min_recall = np.min(recall_list)
        max_recall = np.max(recall_list)

        average_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        min_f1 = np.min(f1_list)
        max_f1 = np.max(f1_list)

        if f'{model_name}_prediction' not in data_component[model_name]:
            data_component[model_name][f'{model_name}_prediction'] = {
                'wrong list': error_list,
                'accuracy': {
                    'right list': accuracy_list,
                    'avg_min_max': [round(average_accuracy, 4), round(min_accuracy, 4), round(max_accuracy, 4)],
                    'updif and lowfid': [round(max_accuracy - average_accuracy, 4),
                                         round(average_accuracy - min_accuracy, 4)],
                    'std': round(std_accuracy, 4)
                },
                'precision': {
                    'right list': precision_list,
                    'avg_min_max': [round(average_precision, 4), round(min_precision, 4), round(max_precision, 4)],
                    'updif and lowfid': [round(max_precision - average_precision, 4),
                                         round(average_precision - min_precision, 4)],
                    'std': round(std_precision, 4)
                },
                'recall': {
                    'right list': recall_list,
                    'avg_min_max': [round(average_recall, 4), round(min_recall, 4), round(max_recall, 4)],
                    'updif and lowfid': [round(max_recall - average_recall, 4),
                                         round(average_recall - min_recall, 4)],
                    'std': round(std_recall, 4)
                },
                'f1': {
                    'right list': f1_list,
                    'avg_min_max': [round(average_f1, 4), round(min_f1, 4), round(max_f1, 4)],
                    'updif and lowfid': [round(max_f1 - average_f1, 4), round(average_f1 - min_f1, 4)],
                    'std': round(std_f1, 4)
                }
            }

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data_component, file, ensure_ascii=False, indent=4)

# analysis and print result
best_metrics_results = find_best_metrics(data_component)
for key, value in best_metrics_results.items():
    print(f"一级目录: {key}")
    for metric, metric_data in value.items():
        print(f"  指标: {metric}")
        print(f"  最佳二级目录: {metric_data['subdirectory']}")
        print(f"  平均值: {round(metric_data['avg'], 4)}")
        print(f"  标准差: {round(metric_data['std'], 4) if metric_data['std'] is not None else None}")
        print(f"  上差和下差: {[round(val, 4) for val in data_component[key][metric_data['subdirectory']][metric]['updif and lowfid']]}")
