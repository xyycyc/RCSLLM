import json
import random
import os
with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)
def split_dataset(file_path,dataset_path, register_path,train_ratio=0.7):
    # 读取JSON文件中的数据
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 按照比例计算训练集和测试集的大小
    train_size = int(len(data) * train_ratio)

    # 切分数据集
    trainset = data[:train_size]
    testset = data[train_size:]
    path_train = file_path[:-5] + '_trainset.json'
    path_test = file_path[:-5] + '_testset.json'
    data_train = dataset_path + r'\\' +os.path.basename(path_train)
    data_test = dataset_path + r'\\' +os.path.basename(path_test)
    # 保存训练集和测试集到新文件
    with open(path_train, 'w', encoding='utf-8') as train_file:
        json.dump(trainset, train_file, ensure_ascii=False, indent=4)

    with open(path_test, 'w', encoding='utf-8') as test_file:
        json.dump(testset, test_file, ensure_ascii=False, indent=4)

    with open(data_train, 'w', encoding='utf-8') as train:
        json.dump(trainset, train, ensure_ascii=False, indent=4)

    with open(data_test, 'w', encoding='utf-8') as test:
        json.dump(testset, test, ensure_ascii=False, indent=4)

    with open(register_path, 'r') as file:
        register_list = json.load(file)
    new_register_train = {"file_name": os.path.basename(dataset_path)+ r'/' +os.path.basename(path_train)}
    new_register_test = {"file_name": os.path.basename(dataset_path)+ r'/' +os.path.basename(path_test)}
    register_list[os.path.basename(path_train)[:-5]] = new_register_train
    register_list[os.path.basename(path_test)[:-5]] = new_register_test

    with open(hyper_parameters['data_info'], 'w') as file:
        json.dump(register_list, file, indent=4)

    print(f"数据集已分割完成：训练集包含 {len(trainset)} 条，测试集包含 {len(testset)} 条。")
    # 使用示例
split_dataset(hyper_parameters['outpath_data_format'], hyper_parameters['dataset_register'],hyper_parameters['data_info'])
split_dataset(hyper_parameters['outpath_slm_llm_data_format'], hyper_parameters['dataset_register'],hyper_parameters['data_info'])
try:
    with open(hyper_parameters['data_info'], 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    data = {}


