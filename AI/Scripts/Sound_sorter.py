import os
import json
import shutil


sounds_folder = 'Sounds/'
bright_folder = 'bright/'
dark_folder = 'dark/'
generic_folder = 'generic/'


os.makedirs(bright_folder, exist_ok=True)
os.makedirs(dark_folder, exist_ok=True)
os.makedirs(generic_folder, exist_ok=True)


with open('filtered_data.json', 'r') as f:
    data = json.load(f)



def move_file_based_on_qualities(file_name, qualities):
    
    if qualities[0] == 1:
        destination_folder = dark_folder
    
    elif qualities[1] == 1:
        destination_folder = bright_folder
    
    else:
        destination_folder = generic_folder

    
    source_path = os.path.join(sounds_folder, file_name)

    
    if os.path.exists(source_path):
        shutil.move(source_path, os.path.join(destination_folder, file_name))
        print(f"File {file_name} moved to: {destination_folder}")
    else:
        print(f"File {file_name} doesn't exist in Sounds/")



for key, value in data.items():
    
    if 'qualities' in value:
        qualities = value['qualities']

        
        if 'keyboard_acoustic' in value['instrument_str'] or 'keyboard_synthetic' in value['instrument_str']:
            
            file_name = value['note_str'] + '.wav'  

            
            move_file_based_on_qualities(file_name, qualities)
