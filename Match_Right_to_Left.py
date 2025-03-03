import os
import random

# Path to the Right_JPEGS folder
right_jpegs_folder = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Right_JPEGS"

# Step 1: Get a list of all files in the Right_JPEGS folder
right_files = [f for f in os.listdir(right_jpegs_folder) if f.endswith(".jpeg") or f.endswith(".jpg")]

# Step 2: Check how many files need to be removed
num_to_remove = len(right_files) - 774  # Adjust to match the Left_JPEGS count

if num_to_remove > 0:
    # Step 3: Randomly select files to remove
    files_to_remove = random.sample(right_files, num_to_remove)

    # Step 4: Remove the selected files
    for file in files_to_remove:
        file_path = os.path.join(right_jpegs_folder, file)
        os.remove(file_path)

    print(f"{num_to_remove} files have been removed from the Right_JPEGS folder to balance it with the Left_JPEGS folder.")
else:
    print("The Right_JPEGS folder already has 774 or fewer files. No changes needed.")
