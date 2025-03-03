import pandas as pd

# Load the combined dataset
combined_csv = 'Combined_Curvature_Data.csv'
combined_data = pd.read_csv(combined_csv)

# Calculate the threshold (midpoint of min and max Curvature)
threshold = (combined_data['Curvature'].min() + combined_data['Curvature'].max()) / 2

# Add a new column for Curvature Classification
combined_data['Curvature_Classification'] = combined_data['Curvature'].apply(lambda x: 0 if x <= threshold else 1)

# Save the updated dataset
#combined_data.to_csv('Updated_Curvature_Classification_Data.csv', index=False)

#print("Curvature Classification column added and dataset saved as 'Updated_Curvature_Classification_Data.csv'")

print(combined_data['Curvature_Classification'].value_counts())

