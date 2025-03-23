import os
import json
import random

# Load hyperparameters from config file
with open('config.json', 'r') as f:
    hyper_parameters = json.load(f)

# Define the directory paths
input_directory = r"dataset\weibo_oral"
output_file_path = r"dataset/exp_cache/formatted_weibo.json"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# Separate the files into true and fake lists
true_files = []
fake_files = []
data_list = []

# Iterate over each file in the input directory to separate true and fake
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        input_file_path = os.path.join(input_directory, filename)
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            try:
                data = json.load(infile)
                if "label" in data and "text" in data and "id" in data:
                    # Extract label
                    label = data["label"].lower()
                    if label == "real":
                        true_files.append(filename)
                    elif label == "fake":
                        fake_files.append(filename)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
                continue

# Set a random seed for reproducibility using hyperparameter
random.seed(hyper_parameters.get("seed", 2024))

# Sample the true files to match the number of fake files
sampled_true_files = random.sample(true_files, min(len(fake_files), len(true_files)))

# Combine the sampled true files and all fake files
selected_files = sampled_true_files + fake_files

# Shuffle the combined list of selected files
random.shuffle(selected_files)
# Process the selected files and save them to the output file
for filename in selected_files:
    input_file_path = os.path.join(input_directory, filename)
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        try:
            data = json.load(infile)
            if "label" in data and "text" in data and "id" in data:
                # Extract label, text, and id values
                label = data["label"].lower()
                text = data["text"]
                component_id = data["id"]

                # Convert label values from 'real' to 'true' and 'fake' to 'false'
                if label == "real":
                    label = 'true'
                elif label == "fake":
                    label = 'false'
                else:
                    continue  # Skip if the label is neither 'real' nor 'fake'

                # Create a new component containing label, text, and unique id
                new_component = {
                    "id": component_id,
                    "input": text,
                    "output": label
                }
                data_list.append(new_component)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {filename}")
            continue

# Save all the new components to the output JSON file
if not os.path.exists(os.path.dirname(output_file_path)):
    os.makedirs(os.path.dirname(output_file_path))
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, ensure_ascii=False, indent=4)