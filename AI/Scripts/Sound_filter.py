import os


folder_path = 'C:\\Users\\Jakub\\Downloads\\nsynth-train-all\\audio'


if os.path.exists(folder_path):
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        
        if os.path.isfile(file_path):
            
            if not (file_name.startswith('keyboard_acoustic') or file_name.startswith('keyboard_synthetic')):
                os.remove(file_path)
                print(f"Removed: {file_name}")
else:
    print(f"Folder {folder_path} doesn't exist.")
