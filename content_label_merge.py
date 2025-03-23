import json
import os
class ContentLabelMerge:
    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def read_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                entry = json.loads(line.strip())
                data.append(entry['predict'])  # 假设每行JSON包含 'predict' 键
        return data

    def write_json(self, data, file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def merge_json_files(self,file_a, file_b_jsonl, output_file,mode):
        data_a = self.read_json(file_a)
        data_b = self.read_jsonl(file_b_jsonl)
        merged_data = []
        # 创建一个以input为键的字典，以便快速查找
        for index, item in enumerate(data_a):
            news = item['input']
            summary = data_b[index]
            # 找到相同input的元素
            if mode == 'suggestion':
                merged_item = {
                    "instruction": "Determine the category of the news based on the suggestion. Choose from: 1 (True), 2 (False), 3 (Unverified), 4 (Non-rumor). You just need answer 1,2,3 or 4 one letter",
                    'input': summary,
                    'output': item['output']
                }
                merged_data.append(merged_item)
            elif mode == 'news':
                merged_item = {
                    "instruction": "Determine the category of the news based on the news. Choose from: 1 (True), 2 (False), 3 (Unverified), 4 (Non-rumor). You just need answer 1,2,3 or 4 one letter",
                    'input': news,
                    'output': item['output']
                }
                merged_data.append(merged_item)
            elif mode == 'resnet':
               merged_item = {
                    "instruction": "Determine the category of the news based on the suggestion and news. Choose from: 1 (True), 2 (False), 3 (Unverified), 4 (Non-rumor). You just need answer 1,2,3 or 4 one letter",
                    'input': f"For news:{news}. Summary:{summary}",
                    'output': item['output']
                }
               merged_data.append(merged_item)
            else:
                print("Invalid mode")
    # 将合并后的数据写入新的JSON文件
            self.write_json(merged_data, output_file)

    def run(self, origin_file, suggestion_file, output_file, mode):
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        self.merge_json_files(origin_file, suggestion_file, output_file,mode)







# 使用示例
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\base\content_tw15\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\merged_output_test.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\base\content_tw15_trainset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\merged_output_train.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\lora\content_tw15_train\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\merged_lora_train.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\lora\content_tw15_test\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\merged_lora_test.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_testset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\lora\seed_control\seed=2024_content_testset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\lora_test.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_trainset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\lora\seed_control\seed=2024_content_trainset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\lora_train.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_testset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\lora\seed_control\seed=2024_10_testset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\lora_test.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_trainset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\lora\seed_control\seed=2024_10_trainset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\lora_train.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_testset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\base\content_tw15_testset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\lora_test.json')
# merge_json_files(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_trainset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\llama3-8b\base\content_tw15_trainset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\lora_train.json')
# test = ContentLabelMerge()
# test.run(r"D:\Desktop\rumor_detection_acl2017\code\twitter15_llm_trainset.json", r"D:\Desktop\finetune\LLaMA-Factory\saves\qwen-7b\base\content_tw15_trainset\generated_predictions.jsonl", r'D:\Desktop\rumor_detection_acl2017\code\seed=2024\qwen_train.json', 'news')