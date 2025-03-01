import pandas as pd

# Load your dataset
df = pd.read_csv('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/trOCR/left_feature_extraction_results.csv')

# Print column names to verify
print(df.columns)

# Calculate the mean for classification
curvature_mean = (0.698 + 0.721) / 2

def classify_number(value, threshold):
    return 1 if value >= threshold else 0

# Classify the Curvature values
if 'Curvature' in df.columns:
    df['classified_curvature'] = df['Curvature'].apply(lambda x: classify_number(x, curvature_mean))
    print(df.head())
else:
    print("'Curvature' column not found in the DataFrame")

df.to_csv('classified_curvature_results.csv', index=False)
