import json
import random

# Path to the JSON file (assuming the JSON structure is known)
with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)
random.seed(hyper_parameters['seed'])
json_path = hyper_parameters['output_path_format']
outpath = hyper_parameters['outpath_data_format']
outpath_slm_llm = hyper_parameters['outpath_slm_llm_data_format']
# Read the JSON file
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

random.shuffle(data)
data_list1 = []
data_list2 = []
# Assuming the JSON file structure is a list of dictionaries with 'title', 'text', and 'label'

for item in data:
    result = None

    if item['output'] == "true":
        result = 1
    elif item['output']== "false":
        result = 2
    elif item['output'] == "unverified":
        result = 3
    elif item['output'] == "non-rumor":
        result = 4
    else:
        print('error:', item['output'])
    if result:
        if hyper_parameters['dataset_name'] == 'twitter15' or hyper_parameters['dataset_name'] == 'twitter16':
            formatted_data = {
                "instruction": "Determine the category of the news based on the title and text. Choose from: 1 (True), 2 (False), 3 (Unverified), 4 (Non-rumor). You just need answer 1,2,3 or 4 one letter",
                "input": item['input'],
                "output": f'{result}'
            }
        else:
            formatted_data = {
                "instruction": "Determine the category of the news based on the title and text. Choose from: 1 (True), 2 (False). You just need answer 1 or 2 one letter",
                "input": item['input'],
                "output": f'{result}'
            }
        slm_data = {
            "instruction": "Given the text below, summarize the key information and synthesize the main points into a single sentence.",
                           # "For example, the output is'Helric Fredou, a police chief involved in the investigation of the Charlie Hebdo case, has died by suicide. "
                           # "Further details can be accessed via the provided URL.' when input is'police chief helric fredou, one of the police officers investigating the case #charliehebdo, commits suicide. "
                           # "URL'",
            "input": item['input'],
            "output": ''
        }
    else:
        print('error_result:', item['output'], result)
    data_list1.append(formatted_data)
    data_list2.append(slm_data)
# random.shuffle(data_list1)
# random.shuffle(data_list2)

# Save the shuffled data to a JSON file
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(data_list1, f, ensure_ascii=False, indent=2)

with open(outpath_slm_llm, 'w', encoding='utf-8') as f:
    json.dump(data_list2, f, ensure_ascii=False, indent=2)
