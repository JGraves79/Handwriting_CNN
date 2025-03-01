import pandas as pd
from sklearn.model_selection import train_test_split

# Load your combined dataset
df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/binary classification/combined_filtered_classified_curvature.csv')

# Define the split ratio (e.g., 80% training, 20% testing)
train_ratio = 0.8

# Perform the split (with randomization)
train_df, test_df = train_test_split(df, test_size=(1 - train_ratio), random_state=42)

# Save the training and testing sets to separate CSV files in the "training set" directory
train_df.to_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/train_set.csv', index=False)
test_df.to_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/test_set.csv', index=False)

# Print the first few rows of each set to verify
print("Training Set:")
print(train_df.head())
print("\nTesting Set:")
print(test_df.head())

