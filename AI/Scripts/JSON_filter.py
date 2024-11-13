import json


input_files = ['C:\\Users\\Jakub\\Downloads\\nsynth-train-all\\examples-valid-original.json', "C:\\Users\\Jakub\\Downloads\\nsynth-train-all\\examples-test-original.json", "C:\\Users\\Jakub\\Downloads\\nsynth-train-all\\examples-train-original.json"]
output_file = 'filtered_data.json'


merged_data = {}


for file_name in input_files:
    
    with open(file_name, 'r') as f:
        data = json.load(f)

    
    filtered_data = {key: value for key, value in data.items()
                     if value['instrument_family_str'] == 'keyboard' and
                     value['instrument_source_str'] in ['acoustic', 'synthetic']}

    
    merged_data.update(filtered_data)


with open(output_file, 'w') as f:
    json.dump(merged_data, f, indent=4)

print(f"Filtered and joined data saved in: '{output_file}'")
