import pandas as pd
import os
import shutil
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Load the Excel file
righties_df = pd.read_excel(r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trOCR\righties.xlsx')

# Define the source and destination directories
source_dir = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\Source_jpeg'
destination_dir = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trOCR\Right-Handed Samples'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get all filenames from the source directory
filenames = os.listdir(source_dir)

# Function to copy files based on partial matching
def copy_files(df, source, destination):
    for idx in df['0001']:  # Adjust this column name if needed
        # Find the best match for the given ID
        best_match, score = process.extractOne(str(idx), filenames, scorer=fuzz.partial_ratio)
        if score > 80:  # Adjust threshold as needed
            src_path = os.path.join(source_dir, best_match)
            if os.path.exists(src_path):
                shutil.copy(src_path, destination)
                print(f'Copied {best_match} to {destination}')
            else:
                print(f'File not found: {src_path}')
        else:
            print(f'No suitable match found for ID: {idx}')

# Copy the corresponding JPEG files
copy_files(righties_df, source_dir, destination_dir)

print('Files copied successfully!')
