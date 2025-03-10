import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to load images from a directory and assign labels manually
def load_images_from_folder(folder, label_value):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            img_array = np.array(img) / 255.0  # Normalize
            images.append(img_array)
            # Assign labels based on some logic or manually
            if 'w0' in filename:
                label = 0
            else:
                label = 1
            labels.append(label)
    return np.array(images), np.array(labels)

# Load testing data
test_images, test_labels = load_images_from_folder('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/Source_jpeg', label_value=1)

# Reshape data to fit the CNN input requirements
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
