from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
combined_csv = 'Updated_Curvature_Classification_Data.csv'  # Your combined file
data = pd.read_csv(combined_csv)

# Separate features (X) and target (y)
X = data[['Curvature']]  # Include other features if necessary
y = data['Curvature_Classification']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to training data
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Class distribution in y_train_smote:")
print(pd.Series(y_train_smote).value_counts())