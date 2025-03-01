import numpy as np
from sklearn import svm, metrics
from PIL import Image
import os

# Ensure reproducibility
np.random.seed(42)

# Function to load images from a directory and assign labels manually
def load_images_from_folder(folder, label_value):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpeg"):
            img_path = os.path.join(folder, filename)
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((28, 28))  # Resize to 28x28
                img_array = np.array(img).flatten() / 255.0  # Normalize and flatten
                images.append(img_array)
                labels.append(label_value)
    return np.array(images), np.array(labels)

# Load training and testing data
train_images_0, train_labels_0 = load_images_from_folder('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/samples for training/w0', label_value=0)
train_images_1, train_labels_1 = load_images_from_folder('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/samples for training/w1', label_value=1)

test_images_0, test_labels_0 = load_images_from_folder('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/samples for testing/w0', label_value=0)
test_images_1, test_labels_1 = load_images_from_folder('C:/Users/16154/vu/Group Project/project 4/Handwriting_CNN/training set/samples for testing/w1', label_value=1)

# Combine the data from both classes
train_images = np.concatenate((train_images_0, train_images_1))
train_labels = np.concatenate((train_labels_0, train_labels_1))

test_images = np.concatenate((test_images_0, test_images_1))
test_labels = np.concatenate((test_labels_0, test_labels_1))

# Ensure labels are integers
train_labels = train_labels.astype(int)
test_labels = test_labels.astype(int)

# Verify the labels
print(f'Training labels distribution: {np.bincount(train_labels)}')
print(f'Testing labels distribution: {np.bincount(test_labels)}')

# Ensure there are at least two classes
unique_classes = np.unique(train_labels)
if len(unique_classes) > 1:
    # Build and Train the SVM Model
    clf = svm.SVC(kernel='linear')
    clf.fit(train_images, train_labels)

    # Evaluate the Model
    predictions = clf.predict(test_images)
    accuracy = metrics.accuracy_score(test_labels, predictions)
    print(f'Test accuracy: {accuracy}')
    print(metrics.classification_report(test_labels, predictions))
    print(metrics.confusion_matrix(test_labels, predictions))
else:
    print("Error: The number of classes has to be greater than one.")
