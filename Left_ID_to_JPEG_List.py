import pandas as pd
import os

# Paths to your files and directories
filtered_csv_path = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Filtered_Left_Data.csv"
jpeg_folder_path = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Source_jpeg"
output_csv_path = r"C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Left_ID_to_JPEG_List.csv"  # Updated filename

# Step 1: Read the IDs from Filtered_Left_Data.csv
filtered_df = pd.read_csv(filtered_csv_path)
id_column = 'wid'  # Replace with the actual name of the column containing IDs
ids = filtered_df[id_column].unique()

# Step 2: Initialize a dictionary to store IDs and their corresponding JPEG files
id_to_jpegs = {id_: [] for id_ in ids}

# Step 3: Search the Source_jpeg folder for matching JPEG files
for jpeg_file in os.listdir(jpeg_folder_path):
    if jpeg_file.endswith(".jpeg") or jpeg_file.endswith(".jpg"):  # Look for JPEG files
        # Extract the ID from the filename (assuming filenames contain IDs)
        for id_ in ids:
            if str(id_) in jpeg_file:  # Match the ID in the filename
                id_to_jpegs[id_].append(jpeg_file)

# Step 4: Convert the dictionary to a DataFrame and save it as a new CSV
output_df = pd.DataFrame({
    "ID": id_to_jpegs.keys(),
    "JPEG_Files": [", ".join(files) for files in id_to_jpegs.values()]
})

output_df.to_csv(output_csv_path, index=False)
print(f"JPEG list saved to: {output_csv_path}")
