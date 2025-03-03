import os
import shutil
import pandas as pd

# Paths to your files and directories
filtered_right_csv = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Balanced_Filtered_Right_Data.csv"
jpeg_folder_path = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Source_jpeg"
output_folder_path = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Right_JPEGS"

# Step 1: Read the Right-Hand WIDs
filtered_df = pd.read_csv(filtered_right_csv)
wid_column = 'wid'  
ids = filtered_df[wid_column].unique()

# Step 2: Create the Right_JPEGS folder
os.makedirs(output_folder_path, exist_ok=True)

# Step 3: Copy matching JPEG files to the new folder
for jpeg_file in os.listdir(jpeg_folder_path):
    if jpeg_file.endswith(".jpeg") or jpeg_file.endswith(".jpg"):  # Check for JPEG files
        for id_ in ids:
            if str(id_) in jpeg_file:  # Match the WID in the filename
                source_path = os.path.join(jpeg_folder_path, jpeg_file)
                dest_path = os.path.join(output_folder_path, jpeg_file)
                shutil.copy2(source_path, dest_path)  # Copy the file with metadata

print(f"All matching JPEG files have been copied to: {output_folder_path}")
