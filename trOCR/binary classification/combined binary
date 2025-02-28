import pandas as pd

# Load the filtered right and left-handed classified curvature CSV files
filtered_right = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/filtered_right_handed_classified_curvature.csv')
filtered_left = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/filtered_left_handed_classified_curvature.csv')

# Combine the two DataFrames
combined_df = pd.concat([filtered_right, filtered_left])

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/combined_filtered_classified_curvature.csv', index=False)

# Print the first few rows to verify
print(combined_df.head())
