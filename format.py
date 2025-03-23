import json

with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)
# 初始化字典来存储 tweets 和 labels
tweets = {}
labels = {}
content_path =hyper_parameters['content_path_format']
label_path = hyper_parameters['label_path_format']
output_path = hyper_parameters['output_path_format']
# 读取 tweets 数据

with open(content_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        tweet_id = parts[0]
        tweet_text = parts[1]
        # 移除文本中的单词 "URL"
        # tweet_text = re.sub(r'\bURL\b', '', tweet_text)
        # 将处理后的文本存储到字典
        tweets[tweet_id] = tweet_text

# 读取 labels 数据
with open(label_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 分割每行数据为标签和编号
        parts = line.strip().split(':')
        label = parts[0]
        tweet_id = parts[1]
        # 将标签存储到字典
        labels[tweet_id] = label

# 创建一个新的列表来存储整合后的数据
formatted_data = []
for tweet_id in tweets:
    if tweet_id in labels:
        # 为每个 tweet 创建一个字典，包含 input 和 output
        formatted_data.append({'id': tweet_id,'input': tweets[tweet_id], 'output': labels[tweet_id]})

# 将整合后的数据保存为 JSON 文件
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print("Data formatted and saved as JSON.")
