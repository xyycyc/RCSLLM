import json
import argparse


# dataset_name = 'twitter16'
dataset_name = 'weibo'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default=fr'{dataset_name}')
#public parameters
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--llama_model_path', type=str, default=r'pretrain_model/Meta-Llama-3-8B-Instruct')
parser.add_argument('--qwen_model_path', type=str, default=r'pretrain_model/Qwen2.5-7B-Instruct')
parser.add_argument('--bert-base_path', type=str, default=r'LLaMA-Factory/pretrain_model/bert-base-uncased')
parser.add_argument('--bert_large_path', type=str, default=r'LLaMA-Factory/pretrain_model/bert-large-uncased')
parser.add_argument('--bert_base_tokenizer_path', type=str, default=r'LLaMA-Factory/pretrain_model/bert-base-uncased/tokenizer')
parser.add_argument('--bert_large_tokenizer_path', type=str, default=r'LLaMA-Factory/pretrain_model/bert-large-uncased/tokenizer')
parser.add_argument('--label_number', type=int, default=2)
#data preprocess parameters
#format.py parameters.
parser.add_argument('--content_path_format', type=str, default=fr"dataset/{dataset_name}/source_tweets.txt")
parser.add_argument('--label_path_format', type=str, default=fr"dataset/{dataset_name}/label.txt")
parser.add_argument('--output_path_format', type=str, default=fr"dataset/exp_cache/formatted_{dataset_name}.json")

#data_format.py parameters
#json_path is '--output_path_format'
parser.add_argument('--outpath_data_format', type=str, default=fr"dataset/exp_cache/{dataset_name}_llm.json")
parser.add_argument('--outpath_slm_llm_data_format', type=str, default=fr"dataset/exp_cache/{dataset_name}_slm_llm.json")

#spilt_data.py parameters
# use 'outpath_slm_llm_data_format' and '--outpath_slm_llm_data_format' as input and auto output
parser.add_argument('--dataset_register', type=str, default=r"LLaMA-Factory/data/fakenews")
parser.add_argument('--data_info', type=str, default=r"LLaMA-Factory/data/dataset_info.json")


#dataset_info
parser.add_argument('--base_connent_train', type=str, default=fr"{dataset_name}_slm_llm_trainset")
parser.add_argument('--base_connent_test', type=str, default=fr"{dataset_name}_slm_llm_testset")
parser.add_argument('--lora_predoct_train', type=str, default=fr"{dataset_name}_llm_trainset")
parser.add_argument('--lora_predoct_test', type=str, default=fr"{dataset_name}_llm_testset")

#LLM base-content,lora-train,lora-predict
#yaml_path
parser.add_argument('--yaml_example', type=str, default=r"LLaMA-Factory\yaml_example\seed_test")
parser.add_argument('--yaml_name', type=str, default=fr"myyaml_{dataset_name}")
parser.add_argument('--seed_index', nargs='+', type=int, default=[2024, 2025, 2026, 2027, 2028],
                    help='A list of integers for the seed index.')
parser.add_argument('--max_samples', type=int, default='10000')
parser.add_argument('--epoch', type=float, default=10.0)
#LLM execution parameters
#cmd.py parameters




#bert train
parser.add_argument('--suggestion_path_llama', type=str, default=fr"LLaMA-Factory\saves\llama3-8b\base\content_{dataset_name}_")
parser.add_argument('--suggestion_path_qwen', type=str, default=fr"LLaMA-Factory\saves\qwen-7b\base\content_{dataset_name}_")







args = parser.parse_args()
with open('config.json', 'w') as f:
    json.dump(vars(args), f, indent= 4)

with open(r'LLaMA-Factory\config.json', 'w') as f:
    json.dump(vars(args), f, indent= 4)
# run(true_path,fake_path,embedding_vector_features = 128, dropout_rate = 0.3, lstm_units = 64, CNN_filters = 64, CNN_kernel_size = 4, batch_size = 32, epochs = 2, lr=1e-5)

