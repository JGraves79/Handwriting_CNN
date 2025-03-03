import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
combined_csv = 'Updated_Curvature_Classification_Data.csv' 
data = pd.read_csv(combined_csv)

# Separate features and target
X = data[['Curvature']]  
y = data['Curvature_Classification']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Normalize the features (scale values between 0 and 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# Reshape for CNN input
X_train_smote = X_train_smote.reshape((X_train_smote.shape[0], X_train_smote.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the model
model = models.Sequential([
    layers.Flatten(input_shape=(X_train_smote.shape[1], 1)),  # Flatten input
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')  # Binary classification
])


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_smote, y_train_smote, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Generate predictions on the test set
predictions = model.predict(X_test).argmax(axis=1)

# Print classification metrics
print(classification_report(y_test, predictions))

# Save the trained model
model.save(r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trained_cnn_model2.keras')
