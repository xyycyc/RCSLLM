import matplotlib
matplotlib.use('TkAgg')  # 更改为 TkAgg 后端
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 准确率数据
name = 'Twitter15'
accuracy_data = {
    'BiGRU': 78.64,
    'GRU': 77.70,
    'BiLSTM': 77.74,
    'LSTM': 79.09,
    'BERT': 82.06,
    'QWEN': 81.47,
    'LLAMA': 79.45,
    'BERT+QWEN': 84.71,
    'BERT+LLAMA': 84.07
}


# name = 'Twitter16'
# accuracy_data = {
#     'BiGRU': 77.15,
#     'GRU': 77.40,
#     'BiLSTM': 77.68,
#     'LSTM': 77.95,
#     'BERT': 78.00,
#     'QWEN': 73.11,
#     'LLAMA': 80.07,
#     'BERT+QWEN': 82.47,
#     'BERT+LLAMA': 80.57
# }





# 提取模型名称
models = list(accuracy_data.keys())

# 计算模型之间的准确率差异
# 创建一个空的矩阵用于存储结果
improvement_matrix = np.zeros((len(models), len(models)))

# 填充矩阵，计算每两个模型之间的准确率提升
for i, base_model in enumerate(models):
    for j, target_model in enumerate(models):
        improvement_matrix[i, j] = accuracy_data[target_model] - accuracy_data[base_model]

# 创建 DataFrame
df_improvement = pd.DataFrame(improvement_matrix, index=models, columns=models)

# 转置 DataFrame
df_improvement_transposed = df_improvement.T

# 设置绘图风格
plt.figure(figsize=(12, 10))

# 绘制转置后的热力图
sns.heatmap(df_improvement_transposed, annot=True, cmap='coolwarm', fmt='.2f', center=0, linewidths=0.5)


# 设置标题和标签
plt.title(f'Model Performance Comparison Heatmap of {name}')
plt.xlabel('Target Model')
plt.ylabel('Base Model')

# 显示热力图
plt.show()
