
import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import random
import os
from content_label_merge import ContentLabelMerge as clm


def save_results_to_json(epoch, accuracy, report, file_path, seed, mode, model_name):
    data = {
        "model": model_name,
        "seed": seed,
        "mode": mode,
        "epoch": epoch,
        "accuracy": accuracy,
        "report": report
    }
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, 'a') as f:
        json.dump(data, f)
        f.write('\n')  # 为每个记录添加换行，以便于阅读和解析
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA
    torch.backends.cudnn.deterministic = True  # 保证CUDA每次返回同样的结果
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('Seed set to {}'.format(seed))


class NewsDataset(Dataset):
    def __init__(self, filename, tokenizer, max_len):
        with open(filename, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 这里我们只使用 'input' 字段，可以根据需要调整使用 'instruction' 字段
        text = self.data[index]['instruction']  + self.data[index]['input']
        label = int(self.data[index]['output']) - 1  # 将输出转换为0-3的索引

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate_4class(model, device, test_loader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            pred_labels.extend(predictions.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=['True', 'False', 'Unverified', 'Non-rumor'], digits=4)
    return accuracy, report


def evaluate_2class(model, device, test_loader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            pred_labels.extend(predictions.cpu().numpy())
            true_labels.extend(batch['labels'].cpu().numpy())
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, target_names=['False', 'True'], digits=4)
    return accuracy, report


def train_and_evaluate_model(seed, mode, model_name, train_path, test_path, outpath, epoch=40):
    # 初始化分词器
    # tokenizer = BertTokenizer.from_pretrained(r"D:\Desktop\sun_rise\LSTMBERT\pretrain_model\bert-base-uncased\tokenizer")
    # model = BertForSequenceClassification.from_pretrained(r"D:\Desktop\sun_rise\LSTMBERT\pretrain_model\bert-base-uncased", num_labels=4)
    tokenizer = BertTokenizer.from_pretrained(hyper_parameters['bert_base_tokenizer_path'] )
    model = BertForSequenceClassification.from_pretrained(hyper_parameters['bert_base_path'], num_labels=hyper_parameters['label_number'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    train_loader = DataLoader(NewsDataset(train_path, tokenizer, max_len=128), batch_size=16, shuffle=True)
    # train_loader = DataLoader(NewsDataset(r"D:\Desktop\rumor_detection_acl2017\code\merged_lora_train.json", tokenizer, max_len=128), batch_size=16, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    test_dataset = NewsDataset(test_path, tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    #

    for epoch in tqdm(range(epoch)):  # 假设训练4个epoch
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss}')

        # 每个epoch后进行测试
        if hyper_parameters['dataset_name'] == 'twitter16' or hyper_parameters['dataset_name'] == 'twitter15':
            accuracy, report = evaluate_4class(model, device, test_loader)
            print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy}")
            print(report)
            save_results_to_json(epoch + 1, accuracy, report, outpath, seed, mode, model_name)

        if hyper_parameters['dataset_name'] == 'isot' or hyper_parameters['dataset_name'] == 'weibo':
            accuracy, report = evaluate_2class(model, device, test_loader)
            print(f"Epoch {epoch + 1}, Test Accuracy: {accuracy}")
            print(report)
            save_results_to_json(epoch + 1, accuracy, report, outpath, seed, mode, model_name)


with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)
epoch = 50
dataprocesser = clm()
# mode = 'resnet'#bert模型使用resnet方案,测试训练集均为lora_xxxx(resnet结构)
# model_name = r'LLAMA'
model_name = r'qwen'
compare_path = fr"dataset/exp_cache/{hyper_parameters['dataset_name']}_llm"
index1 = [2024, 2025, 2026, 2027, 2028]
index2 = [['resnet', 'news'], ['resnet', 'suggestion'], ['resnet', 'resnet'], ['news', 'suggestion'],
          ['news', 'resnet'], ['news', 'news'], ['suggestion', 'news'], ['suggestion', 'resnet'],
          ['suggestion', 'suggestion']]

if model_name == 'LLAMA':
    suggestion_path = hyper_parameters['suggestion_path_llama']
elif model_name == 'qwen':
    suggestion_path = hyper_parameters['suggestion_path_qwen']
else:
    print("Model not found")
for seed in index1:
    set_seed(seed= seed)
    for mode in index2:
        data_path = fr"dataset/exp_cache/seed={seed}/data"
        outpath = fr'dataset/exp_cache/seed={seed}\result\{mode[0]}_{mode[1]}_{model_name}_evaluate.json'
        traindata_path = data_path + fr"\{mode[0]}_train_{model_name}.json"
        testdata_path = data_path + fr"\{mode[1]}_test_{model_name}.json"
        dataprocesser.run(compare_path + r"_trainset.json", suggestion_path + r"trainset\generated_predictions.jsonl", traindata_path, mode[0])
        dataprocesser.run(compare_path + r'_testset.json', suggestion_path + r"testset\generated_predictions.jsonl", testdata_path, mode[1])

        print('mode:',mode)
        train_and_evaluate_model(seed, mode, model_name, traindata_path, testdata_path, outpath, epoch)


