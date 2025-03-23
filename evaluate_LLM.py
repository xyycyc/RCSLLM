from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
import re
def calculate_metrics(y_true, y_pred, filename):
    """Calculate and print metrics."""
    match_seed = re.search(r"seed=(\d+)", filename)
    match_time = re.search(r"checkpoint-(\d+)", filename)

    if match_time:
        time_number = match_time.group(1)
    else:
        time_number = 0
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Metrics for {match_seed.group(1)},{time_number}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

def give_score(filename, valid_labels):
    y_true = []
    y_pred = []
    errors = []  # Store invalid data records
    invalid_predictions_count = 0  # Count invalid predictions
    error_lp = 0  # Count errors due to JSON parsing or type issues

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                true_label = int(data['label'])  # Assuming labels are already integers
                predicted_label = int(data['predict'])  # Assuming predictions are also integers

                # Check if labels and predictions are valid
                if true_label not in valid_labels:
                    errors.append({'error_type': 'invalid_label', 'data': data})
                elif predicted_label not in valid_labels:
                    invalid_predictions_count += 1
                else:
                    y_true.append(true_label)
                    y_pred.append(predicted_label)

            except (ValueError, KeyError, json.JSONDecodeError):
                error_lp += 1
                errors.append({'error_type': 'parsing_error', 'data': line})

    # Output errors and invalid predictions
    # if errors:
        # print("Errors occurred during processing:")
        # for error in errors:
            # print(error)

    # print(f"Number of invalid predictions: {invalid_predictions_count}")
    # print(f"Number of parsing errors: {error_lp}")

    # Calculate metrics if valid predictions exist
    if y_true and y_pred:
        return calculate_metrics(y_true, y_pred, filename), [errors, invalid_predictions_count]
    else:
        print("No valid data to compute metrics.")
        return None, None, None, None









valid_labels = list(range(1,3))
mode = r'llama'
# mode = r'qwen'
checkpint_index_qwen = [0, 500, 740]  # tw16
checkpint_index_llama = [0, 500, 740]
# checkpint_index_qwen = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3630]  # isot
# checkpint_index_llama = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3630]
# checkpint_index_qwen = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3630]  # isot
# checkpint_index_llama = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3630]
# [0,500,640]
if mode == r'llama':
    index1 = [2024,2025,2026,2027,2028]
    # index2 = [500,1000,1500,2000,2500,3000,3500,3810]
    index2 =[0,500,640]
    evaluate = {}
    # YAML 文件路径
    for seed in index1:
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        for times in index2:
            if times == 0:#0就是base
                output_dir = fr"LLaMA-Factory/saves/llama3-8b/lora/seed_control/seed={seed}_label/base/generated_predictions.jsonl"
            else:
                output_dir = fr"LLaMA-Factory/saves/llama3-8b/lora/seed_control/seed={seed}_label/checkpoint-{times}/generated_predictions.jsonl"
            print('seed and checkpoint:',seed,times)
            [accuracy, precision, recall, f1], [error, invalid_predictions_count] =give_score(output_dir, valid_labels)
            if seed not in evaluate:
                evaluate[seed] = {}
            if times not in evaluate[seed]:
                evaluate[seed][times] = {}
            evaluate[seed][times] = {
                "accuracy": accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'error': len(error),
                'invalid_predictions_count': invalid_predictions_count
            }
            accuracy_list.append(float(accuracy))
            f1_list.append(float(f1))
            precision_list.append(float(precision))
            recall_list.append(float(recall))
            # print(accuracy, f1)
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        average_f1 = sum(f1_list) / len(f1_list)
        average_precision = sum(precision_list) / len(precision_list)
        average_recall = sum(recall_list) / len(recall_list)
        print("Average accuracy:", average_accuracy)
        print("Average f1:", average_f1)
        print("Average precision:", average_precision)
        print("Average recall:", average_recall)
        print("Max accuracy:", max(accuracy_list))
        print("Max f1:", max(f1_list))
        print("Max precision:", max(precision_list))
        print("Max recall:", max(recall_list))
        print('\n')

    with open(r'result\lora_seed_times_test_llama.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(evaluate, indent=4, ensure_ascii=False))

if mode == r'qwen':
    index1 = [2024, 2025, 2026, 2027, 2028]
    # index2 = [500, 1000, 1270]
    index2 = [0,500,640]
    evaluate = {}
    # YAML 文件路径
    for seed in index1:
        accuracy_list = []
        f1_list = []
        precision_list = []
        recall_list = []
        for times in index2:
            if times == 0:#0就是base
                output_dir = fr"LLaMA-Factory/saves/qwen-7b/lora/seed_control/seed={seed}_label/base/generated_predictions.jsonl"
            else:
                output_dir = fr"LLaMA-Factory/saves/qwen-7b/lora/seed_control/seed={seed}_label/checkpoint-{times}/generated_predictions.jsonl"
            print('seed and checkpoint:', seed, times)
            [accuracy, precision, recall, f1], [error, invalid_predictions_count] = give_score(output_dir, valid_labels)
            if seed not in evaluate:
                evaluate[seed] = {}
            if times not in evaluate[seed]:
                evaluate[seed][times] = {}
            evaluate[seed][times] = {
                "accuracy": accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'error': len(error),
                'invalid_predictions_count': invalid_predictions_count
            }
            accuracy_list.append(float(accuracy))
            f1_list.append(float(f1))
            precision_list.append(float(precision))
            recall_list.append(float(recall))
            print(accuracy, f1)
        average_accuracy = sum(accuracy_list) / len(accuracy_list)
        average_f1 = sum(f1_list) / len(f1_list)
        average_precision = sum(precision_list) / len(precision_list)
        average_recall = sum(recall_list) / len(recall_list)
        print("Average accuracy:", average_accuracy)
        print("Average f1:", average_f1)
        print("Average precision:", average_precision)
        print("Average recall:", average_recall)
        print("Max accuracy:", max(accuracy_list))
        print("Max f1:", max(f1_list))
        print("Max precision:", max(precision_list))
        print("Max recall:", max(recall_list))
        print('\n')
    with open(r'result\lora_seed_times_test_qwen.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(evaluate, indent=4, ensure_ascii=False))