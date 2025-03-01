import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import cv2  # OpenCV for additional image processing
import numpy as np
import pandas as pd  # Import pandas for saving results to CSV

# Initialize the processor and model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')

# Define the directory containing your handwriting samples
directory = r'C:\Users\16154\vu\Group Project\project 4\Handwriting_CNN\trOCR\Right-Handed Processed'

# Initialize a list to store results
results = []

# Function to analyze the slant of handwriting
def analyze_slant(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi for x1, y1, x2, y2 in lines[:, 0]]
        avg_angle = np.mean(angles) if len(angles) > 0 else 0
    else:
        avg_angle = 0
    return avg_angle

# Function to analyze stroke direction
def analyze_stroke_direction(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    directions = []
    for contour in contours:
        for i in range(1, len(contour)):
            x1, y1 = contour[i-1][0]
            x2, y2 = contour[i][0]
            direction = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            directions.append(direction)
    avg_direction = np.mean(directions) if len(directions) > 0 else 0
    return avg_direction

# Function to analyze curvature and smoothness
def calculate_curvature(x1, y1, x2, y2, x3, y3):
    try:
        curvature = np.abs((x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)) / np.sqrt(((x2 - x1)**2 + (y2 - y1)**2) * ((x3 - x2)**2 + (y3 - y2)**2))
        if not np.isfinite(curvature):
            curvature = 0
    except ZeroDivisionError:
        curvature = 0
    return curvature

def analyze_curvature_and_smoothness(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    curvatures = []
    smoothness = []
    for contour in contours:
        for i in range(2, len(contour)):
            x1, y1 = contour[i-2][0]
            x2, y2 = contour[i-1][0]
            x3, y3 = contour[i][0]
            curvature = calculate_curvature(x1, y1, x2, y2, x3, y3)
            curvatures.append(curvature)
            # Calculate smoothness
            distance = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
            smoothness.append(distance)
    avg_curvature = np.mean(curvatures) if len(curvatures) > 0 else 0
    avg_smoothness = np.mean(smoothness) if len(smoothness) > 0 else 0
    return avg_curvature, avg_smoothness

# Function to analyze pressure distribution
def analyze_pressure_distribution(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pressure_distribution = np.sum(thresh / 255, axis=0)
    avg_pressure = np.mean(pressure_distribution)
    return avg_pressure

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image

# Process each image in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(directory, filename)
        image = preprocess_image(image_path)
        
        # Analyze various handwriting features
        slant = analyze_slant(image)
        stroke_direction = analyze_stroke_direction(image)
        curvature, smoothness = analyze_curvature_and_smoothness(image)
        pressure = analyze_pressure_distribution(image)
        
        # Use TrOCR model to recognize text
        pixel_values = processor(images=image, return_tensors='pt').pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=50)  # Adjust max_new_tokens as needed
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Store results in a dictionary
        result = {
            'Filename': filename,
            'Transcription': transcription,
            'Slant': slant,
            'StrokeDirection': stroke_direction,
            'Curvature': curvature,
            'Smoothness': smoothness,
            'Pressure': pressure
        }
        results.append(result)

# Add a 'Handedness' column to the results
handedness_label = 1  # Change to 1 for right-handed samples

# Process each image in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        image_path = os.path.join(directory, filename)
        image = preprocess_image(image_path)
        
        # Analyze various handwriting features
        slant = analyze_slant(image)
        stroke_direction = analyze_stroke_direction(image)
        curvature, smoothness = analyze_curvature_and_smoothness(image)
        pressure = analyze_pressure_distribution(image)
        
        # Use TrOCR model to recognize text
        pixel_values = processor(images=image, return_tensors='pt').pixel_values
        generated_ids = model.generate(pixel_values, max_new_tokens=50)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Store results in a dictionary
        result = {
            'Filename': filename,
            'Transcription': transcription,
            'Slant': slant,
            'StrokeDirection': stroke_direction,
            'Curvature': curvature,
            'Smoothness': smoothness,
            'Pressure': pressure,
            'Handedness': handedness_label
        }
        results.append(result)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('right_feature_extraction_results.csv', index=False)

print("Right feature extraction completed and results saved")
