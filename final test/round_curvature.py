import pandas as pd

# Load the CSV file
df = pd.read_csv(r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\handwriting_analysis_results.csv')

# Define a function to round curvature values closer to 0 or 1
def round_curvature(Curvature):
    if Curvature < 0.7075:
        return 0  # Closer to 0 for left-hand classification
    else:
        return 1  # Closer to 1 for right-hand classification

# Apply the rounding function to the 'curvature' column
df['classification'] = df['Curvature'].apply(round_curvature)

# Display the first few rows of the dataframe to verify the results
print(df.head())

# Save the modified dataframe to a new CSV file
df.to_csv(r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\handwriting_analysis_results_classified.csv', index=False)
