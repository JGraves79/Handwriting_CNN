import pandas as pd
import shutil
import os

# Define source directories for handwritten samples
source_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set'

# Define destination directories for training and testing samples
train_dest_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/samples for training'
test_dest_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/samples for testing'

os.makedirs(train_dest_dir, exist_ok=True)
os.makedirs(test_dest_dir, exist_ok=True)

# Function to safely convert filenames to strings and handle NaN values
def safe_str(val):
    return str(val) if pd.notna(val) else ''

# Copy files for training set
train_df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/train_set.csv')
print("Copying training set files...")
for filename in train_df['Filename'].apply(safe_str):
    if filename:  # Check if the filename is not empty
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(train_dest_dir, filename))
        else:
            print(f"File not found: {source_path}")
    else:
        print("Empty filename found in training set")

# Copy files for testing set
test_df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/test_set.csv')
print("Copying testing set files...")
for filename in test_df['Filename'].apply(safe_str):
    if filename:  # Check if the filename is not empty
        source_path = os.path.join(source_dir, filename)
        if os.path.exists(source_path):
            shutil.copy(source_path, os.path.join(test_dest_dir, filename))
        else:
            print(f"File not found: {source_path}")
    else:
        print("Empty filename found in testing set")

print("Files copied successfully!")
