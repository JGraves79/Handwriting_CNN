import pandas as pd

# Load the classified curvature results from the specified path
df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/classified_curvature_right_handed_results.csv')

# Filter the DataFrame based on the conditions
filtered_df = df[(df['Handedness'] == 1) & (df['classified_curvature'] == 1)]

# Save the filtered DataFrame to a new CSV file in the desired location
filtered_df.to_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/filtered_right_handed_classified_curvature.csv', index=False)

# Print the first few rows to verify
print(filtered_df.head())
