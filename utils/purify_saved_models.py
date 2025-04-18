# purify_saved_models.py

import os
import re

def updated_saved_models():
    # Path to saved_models directory
    saved_models_dir = "./saved_models"
    print("Purifying saved models...")

    # Check if directory exists
    if not os.path.exists(saved_models_dir):
        print(f"Directory '{saved_models_dir}' does not exist.")
        return
    
    # Get all files in the directory
    files = os.listdir(saved_models_dir)

    # Count the number of files containing 'epoch', if it's not over 0, stop everything
    epoch_files = [file for file in files if 'epoch' in file.lower()]
    if len(epoch_files) == 0:
        print("No files containing 'epoch' found. Stopping everything.")
        return
    
    # First, delete files not containing 'epoch'
    for file in files:
        if 'epoch' not in file.lower():
            file_path = os.path.join(saved_models_dir, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")
    # Then rename files with epoch to their simplified form
    files = os.listdir(saved_models_dir)  # Get updated file list
    pattern = r'(poker_agent_Player_\d+)_epoch_\d+(.pth)'
    
    for file in files:
        match = re.match(pattern, file)
        if match:
            old_path = os.path.join(saved_models_dir, file)
            new_name = f"{match.group(1)}.pth"
            new_path = os.path.join(saved_models_dir, new_name)
            
            try:
                # If destination file exists, remove it first
                if os.path.exists(new_path):
                    os.remove(new_path)
                os.rename(old_path, new_path)
                print(f"Renamed: {file} -> {new_name}")
            except Exception as e:
                print(f"Error renaming {file}: {e}")

def purify_saved_models():
    # Path to saved_models directory
    saved_models_dir = "./saved_models"
    print("Deleting models containing 'epoch'...")

    # Check if directory exists
    if not os.path.exists(saved_models_dir):
        print(f"Directory '{saved_models_dir}' does not exist.")
        return

    # Delete all models containing epoch
    for file in os.listdir(saved_models_dir):
        if "epoch" in file.lower():
            file_path = os.path.join(saved_models_dir, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file}")
            except Exception as e:
                print(f"Error deleting {file}: {e}")

if __name__ == "__main__":
    input_value = input("Choose action:\n(p) Delete all models containing 'epoch'\n(u) Update model names (keep only latest epoch)\n> ")
    if input_value.lower() == "p":
        purify_saved_models()
    elif input_value.lower() == "u":
        updated_saved_models()
    else:
        print("Invalid option. Please use 'p' or 'u'")
    