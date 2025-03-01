import pandas as pd
import shutil
import os

# Load the combined CSV file
combined_df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/combined_filtered_classified_curvature.csv')

# Define source directories
left_hand_source_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/Left-Handed Processed'
right_hand_source_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/Right-Handed Processed'

# Define destination directory
destination_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set'
os.makedirs(destination_dir, exist_ok=True)

# Copy files from source directories to destination directory
for filename in combined_df['Filename']:
    # Determine the source directory based on handedness
    if 'Left' in filename:
        source_path = os.path.join(left_hand_source_dir, filename)
    else:
        source_path = os.path.join(right_hand_source_dir, filename)
    
    # Destination path
    destination_path = os.path.join(destination_dir, filename)
    
    # Copy file
    shutil.copy(source_path, destination_path)

print("Files copied successfully!")
