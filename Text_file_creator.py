import os
import pandas as pd
from pathlib import Path

directory = r'C:\Users\X579430\OneDrive - Nissan Motor Corporation\Documents\Bootcamp\Handwriting_CNN'

# Directory containing JPEG files
jpeg_directory = "Source_jpeg"

# Path to the CSV file
csv_file_path = Path("C:/Users/X579430/OneDrive - Nissan Motor Corporation/Documents/Bootcamp/Handwriting_CNN/Source_Text.csv")

# Directory to save the new text files
output_directory = "Source_text"

# Function to load CSV file data into a DataFrame
def load_csv_file(csv_file_path):
    df = pd.read_csv(csv_file_path, encoding='latin-1', header=None)
    return df

# Function to match JPEG file names to DataFrame data and save corresponding text
def match_and_save_text(jpeg_directory, df, output_directory):
    for file_name in os.listdir(os.path.join(directory, jpeg_directory)):
        if file_name.endswith(".jpeg") or file_name.endswith(".jpg"):
            base_name = os.path.splitext(file_name)[0]
            matched_text = None

            # Look for a matching portion in the DataFrame
            for _, row in df.iterrows():
                if row[0] in base_name:
                    matched_text = row[1]
                    break

            # Save the corresponding text as a .txt file in the output directory
            if matched_text:
                txt_file_path = os.path.join(directory, output_directory, f"{base_name}.txt")
                with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(matched_text)
                print(f"Saved: {txt_file_path}")

# Load CSV file data
df = load_csv_file(csv_file_path)

# Ensure output directory exists
os.makedirs(os.path.join(directory, output_directory), exist_ok=True)

# Match JPEG files to DataFrame data and save the corresponding text
match_and_save_text(jpeg_directory, df, output_directory)

print("Task completed!")
