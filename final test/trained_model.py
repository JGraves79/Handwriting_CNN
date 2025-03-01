import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import pandas as pd
from PIL import Image
import os

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load images from a directory and assign labels based on the CSV
def load_images_from_folder(folder, csv_path):
    images = []
    labels = []
    df = pd.read_csv(csv_path)
    df['Filename'] = df['Filename'].str.strip()  # Ensure no leading/trailing spaces

    for filename in os.listdir(folder):
        if filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            img_array = np.array(img) / 255.0  # Normalize
            images.append(img_array)

            # Assign labels from the CSV
            if filename in df['Filename'].values:
                label = df.loc[df['Filename'] == filename, 'classification'].values[0]
                labels.append(label)
            else:
                print(f"Filename {filename} not found in CSV.")
    return np.array(images), np.array(labels)

# Load test images and labels from the CSV
folder_path = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\final_test\mnist'
csv_path = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\handwriting_analysis_results_classified.csv'
test_images, test_labels = load_images_from_folder(folder_path, csv_path)

# Reshape data to fit the CNN input requirements
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Load the pre-trained CNN model
model = models.load_model(r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trained_cnn_model.h5')

# Compile the model (if required)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
