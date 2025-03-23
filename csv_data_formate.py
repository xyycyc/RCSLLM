import pandas as pd
import json
import random
import os

# 读取 CSV 文件并合并转换为 JSON，每十条取一条
def read_csv_and_convert_to_json(csv_files, json_file, seed=42):
    # 初始化空列表来保存所有组件
    components = []

    # 遍历每个 CSV 文件并读取内容
    for csv_file in csv_files:
        # 使用 pandas 读取 csv 文件
        df = pd.read_csv(csv_file)

        # 设置 output 值，根据文件名判断
        output_value = "true" if "True.csv" in csv_file else "false"

        # 遍历 DataFrame 的每十行并创建 JSON 组件
        for index, row in df.iloc[::10].iterrows():  # 这里的 `::10` 实现每十条取一条
            component = {
                "id": len(components),  # 使用当前组件列表的长度作为唯一 ID
                "input": row.iloc[0],  # 第一列作为 input
                "output": output_value  # 根据文件设置 output
            }
            components.append(component)

    # 打乱组件列表，受随机种子控制
    random.shuffle(components)

    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(json_file)):
        os.makedirs(os.path.dirname(json_file))

    # 将组件列表写入 JSON 文件
    with open(json_file, 'w') as f:
        json.dump(components, f, indent=4)


# 示例使用
with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)

dataset_name = 'isot'
csv_files = ['dataset/isot/Fake.csv', 'dataset/ISOT/True.csv']  # 请将此路径替换为实际的 CSV 文件路径
json_file = fr'dataset/exp_cache/formatted_{dataset_name}.json'  # 输出的 JSON 文件路径

# 设置随机种子
random.seed(hyper_parameters['seed'])

# 调用函数转换文件
read_csv_and_convert_to_json(csv_files, json_file)
print(f"Successfully converted {csv_files} to {json_file}.")
