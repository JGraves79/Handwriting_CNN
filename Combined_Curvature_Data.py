import pandas as pd

# Define file paths
left_csv = 'Left_JPEGS_File_Extraction.csv'
right_csv = 'Right_JPEGS_File_Extraction.csv'
combined_csv = 'Combined_Curvature_Data.csv'

# Load the CSV files
left_data = pd.read_csv(left_csv)
right_data = pd.read_csv(right_csv)

# Add a "Dataset" column to distinguish between left and right
left_data['Dataset'] = 'Left'
right_data['Dataset'] = 'Right'

# Combine the two datasets
combined_data = pd.concat([left_data, right_data], ignore_index=True)

# Save the combined dataset to a new CSV
combined_data.to_csv(combined_csv, index=False)

print(f"Combined dataset saved as {combined_csv}")
