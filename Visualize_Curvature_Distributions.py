import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to the CSV files for left and right datasets
left_csv = 'Left_JPEGS_File_Extraction.csv'
right_csv = 'Right_JPEGS_File_Extraction.csv'

# Load the datasets
left_data = pd.read_csv(left_csv)
right_data = pd.read_csv(right_csv)

# Plot histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # Left dataset histogram
sns.histplot(left_data['Curvature'], kde=True, color='blue', bins=30)
plt.title('Left Curvature Distribution')
plt.xlabel('Curvature')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)  # Right dataset histogram
sns.histplot(right_data['Curvature'], kde=True, color='green', bins=30)
plt.title('Right Curvature Distribution')
plt.xlabel('Curvature')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plot boxplots
plt.figure(figsize=(8, 6))
sns.boxplot(data=[left_data['Curvature'], right_data['Curvature']], 
            boxprops={'facecolor': 'blue'},
            medianprops={'color': 'green'})
plt.xticks([0, 1], ['Left', 'Right'])
plt.title('Boxplot of Curvature (Left vs. Right)')
plt.ylabel('Curvature')
plt.xlabel('Dataset')
plt.show()

# Calculate threshold for classification (midpoint between min and max)
left_threshold = (left_data['Curvature'].min() + left_data['Curvature'].max()) / 2
right_threshold = (right_data['Curvature'].min() + right_data['Curvature'].max()) / 2

print("Left Dataset Threshold: ", left_threshold)
print("Right Dataset Threshold: ", right_threshold)

with open("Curvature_Summary.txt", "a") as f:  # Append to the file
    f.write("\nThresholds:\n")
    f.write("Left Dataset Threshold: " + str(left_threshold) + "\n")
    f.write("Right Dataset Threshold: " + str(right_threshold) + "\n")
