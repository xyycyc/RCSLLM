import yaml
import json
import os
import re
def replace_path_segment(path, old_segment, new_segment):
    # 替换路径中的指定部分
    new_path = path.replace(old_segment, new_segment)
    return new_path
def list_files(directory):
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 拼接文件的完整路径
            file_path = os.path.join(root, filename)
            write_yaml(file_path)
            # print(file_path)


def extract_seed_info(path):
    match = re.search(r'seed=(\d+)', path)
    if match:
        seed_value = match.group(1)  # 获取匹配到的数字部分
        return seed_value
    else:
        return False


def write_yaml(original_file):
    with open(original_file, 'r') as file:
        data = yaml.safe_load(file)
    seed = extract_seed_info(original_file)
    full_path = replace_path_segment(original_file, r"yaml_example", myyaml_name)
    # print(full_path,seed)
    if not seed:
        if original_file ==r'LLaMA-Factory\yaml_example\seed_test\base_content\llama3_base_predict_test.yaml':
            data['model_name_or_path'] = hyper_parameters['llama_model_path']
            data['eval_dataset'] = f'{dataset_name}_slm_llm_testset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['output_dir'] = f'saves/llama3-8b/base/content_{dataset_name}_testset'
        elif original_file ==r'LLaMA-Factory\yaml_example\seed_test\base_content\llama3_base_predict_train.yaml':
            data['model_name_or_path'] = hyper_parameters['llama_model_path']
            data['eval_dataset'] = f'{dataset_name}_slm_llm_trainset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['output_dir'] = f'saves/llama3-8b/base/content_{dataset_name}_trainset'
        elif original_file ==r'LLaMA-Factory\yaml_example\seed_test\base_content\qwen2_base_predict_test.yaml':
            data['model_name_or_path'] = hyper_parameters['qwen_model_path']
            data['eval_dataset'] = f'{dataset_name}_slm_llm_testset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['output_dir'] = f'saves/qwen-7b/base/content_{dataset_name}_testset'
        elif original_file ==r'LLaMA-Factory\yaml_example\seed_test\base_content\qwen2_base_predict_train.yaml':
            data['model_name_or_path'] = hyper_parameters['qwen_model_path']
            data['eval_dataset'] = f'{dataset_name}_slm_llm_trainset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['output_dir'] = f'saves/qwen-7b/base/content_{dataset_name}_trainset'
    else:
        if os.path.basename(original_file) == r'llama3_lora_predict_label.yaml':
            data['model_name_or_path'] = hyper_parameters['llama_model_path']
            data['adapter_name_or_path'] = fr"saves/llama3-8b/lora/seed_control/seed={seed}"
            data['eval_dataset'] = fr'{dataset_name}_llm_testset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['output_dir'] = fr'saves/llama3-8b/lora/seed_control/seed={seed}_label'

        elif os.path.basename(original_file) == r'llama3_lora_sft_train.yaml':
            data['model_name_or_path'] = hyper_parameters['llama_model_path']
            data['output_dir'] = fr"saves/llama3-8b/lora/seed_control/seed={seed}"
            data['dataset'] = fr'identity,{dataset_name}_llm_trainset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['num_train_epochs'] = hyper_parameters['epoch']

        elif os.path.basename(original_file) == r'qwen2_lora_predict_label.yaml':
            data['model_name_or_path'] = hyper_parameters['qwen_model_path']
            data['adapter_name_or_path'] = fr"saves/qwen-7b/lora/seed_control/seed={seed}"
            data['eval_dataset'] = fr'{dataset_name}_llm_testset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['output_dir'] = fr'saves/qwen-7b/lora/seed_control/seed={seed}_label'

        elif os.path.basename(original_file) == r'qwen2_lora_sft_train.yaml':
            data['model_name_or_path'] = hyper_parameters['qwen_model_path']
            data['output_dir'] = fr"saves/qwen-7b/lora/seed_control/seed={seed}"
            data['dataset'] = fr'identity,{dataset_name}_llm_trainset'
            data['max_samples'] = hyper_parameters['max_samples']
            data['num_train_epochs'] = hyper_parameters['epoch']

    # print(os.path.dirname(full_path))
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))
    try:
        with open(full_path, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)
    except Exception as e:
        print("Failed to write file:", e)




if __name__ == "__main__":
    with open('config.json', 'r') as f:
        hyper_parameters = json.load(f)
    yaml_example =hyper_parameters['yaml_example']
    myyaml_name =hyper_parameters['yaml_name']
    dataset_name = hyper_parameters['dataset_name']
    list_files(yaml_example)
    print('yaml write down')
