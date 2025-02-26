import pandas as pd
import os

# Load the CSV file
df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/Source_Data/CSAFE_Handwriting_Info.csv')

# Path to the directory containing PNG files
png_dir = 'C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/Source_Data/'

# Create a dictionary to map WID numbers to PNG files
wid_to_file = {}

# Assuming PNG files follow the pattern you've mentioned
for file in os.listdir(png_dir):
    if file.endswith('.png'):
        # Extracting the WID part from the file name
        parts = file.split('_')
        wid = parts[0][1:]  # Extract numerical part after 'w'

        # Map the extracted WID to the file name
        wid_to_file[wid] = file

# Function to get the corresponding PNG file for a given WID in the CSV
def get_png_file(wid):
    return wid_to_file.get(str(wid).zfill(4), 'File not found')  # Ensure WID has leading zeros if needed

# Adding a new column in the dataframe to store corresponding PNG file names
df['PNG_File'] = df['wid'].apply(get_png_file)

# Display the updated dataframe
print(df.head())
