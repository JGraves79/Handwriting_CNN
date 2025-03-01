from PIL import Image
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define paths
source_directory = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trOCR\Right-Handed Source'
save_directory = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trOCR\Right-Handed Processed'

# Ensure the save directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

def preprocess_image(img_path, save_path):
    # Load the image
    img = Image.open(img_path)

    # Convert to grayscale
    img = img.convert('L')

    # Resize to 512x512 pixels
    img = img.resize((512, 512))

    # Convert to numpy array
    img_array = np.array(img)

    # Apply Otsu's thresholding
    _, binary_img = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert the colors
    inverted_img = cv2.bitwise_not(binary_img)

    # Display the processed image
    plt.imshow(inverted_img, cmap='gray')
    plt.title('Preprocessed Image')
    plt.show()

    # Normalize the pixel values to [0, 1]
    img_array = img_array / 255.0

    # Save the array for further processing
    img_array = img_array * 255

    # Save the inverted image as JPEG
    cv2.imwrite(save_path, inverted_img)

    print(f"Image successfully saved as {save_path}")

    return img_array

def apply_additional_processing(img_path):
    # Load the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Apply thresholding with a different threshold value
    threshold_value = 128  # Try values between 0 to 255
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Ensure binary_img is a NumPy array
    if isinstance(binary_img, np.ndarray):
        # Display the binary image
        plt.imshow(binary_img, cmap='gray')
        plt.title(f'Binary Image with Threshold {threshold_value}')
        plt.show()
    else:
        print(f"Unexpected data type: {type(binary_img)}")

    # Apply Otsu's thresholding
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Ensure binary_img is a NumPy array
    if isinstance(binary_img, np.ndarray):
        # Display the binary image
        plt.imshow(binary_img, cmap='gray')
        plt.title('Binary Image with Otsu Thresholding')
        plt.show()
    else:
        print(f"Unexpected data type: {type(binary_img)}")

    # Invert the colors using bitwise NOT
    inverted_img = cv2.bitwise_not(binary_img)

    # Display the inverted image
    plt.imshow(inverted_img, cmap='gray')
    plt.title('Inverted Image')
    plt.show()

    # Apply Gaussian Blur
    blurred_img = cv2.GaussianBlur(binary_img, (5, 5), 0)

    # Display the blurred image
    plt.imshow(blurred_img, cmap='gray')
    plt.title('Blurred Image')
    plt.show()

    # Apply dilation followed by erosion (closing)
    kernel = np.ones((3, 3), np.uint8)
    morphed_img = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)

    # Display the morphed image
    plt.imshow(morphed_img, cmap='gray')
    plt.title('Morphed Image')
    plt.show()

    # Normalize the pixel values to [0, 1] again for model input
    final_img_array = morphed_img / 255.0

    print(final_img_array.shape)  # Should be (28, 28)
    return final_img_array

# Process all images in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        img_path = os.path.join(source_directory, filename)
        save_path = os.path.join(save_directory, f'processed_{filename}')
        
        preprocess_image(img_path, save_path)
        apply_additional_processing(img_path)
