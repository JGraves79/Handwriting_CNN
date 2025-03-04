# Handwriting_CNN - Handwriting Analysis using Convolutional Neural Network

Final Project - Vanderbilt University Data Analytics Bootcamp

# Objective

To develop a Convolutional Neural Network (CNN) capable of determining whether an image of handwriting was produced by a left- or right-handed person. The analysis will include Microsoft’s TrOCR for feature extraction and CNN to classify handwriting into left-handedness or right-handedness.

# Data Source

CSAFE Handwriting database

Crawford, Amy; Ray, Anyesha; Carriquiry, Alicia; Kruse, James; Peterson, Marc (2019): CSAFE Handwriting Database. Iowa State University. Dataset. https://doi.org/10.25380/iastate.10062203.v1

The data consists of over 12,000 handwritten messages written by 90 volunteers over a period of time. Each person wrote the same three paragraphs multiple times to provide samples of handwriting for review. This dataset was converted from PNG to JPEG and reduced to a single series of over 4000 files.

pngtojpeg_all.ipynb
Text_file_creator.py

## PREPROCESSING
Create two separate lists as csv files, pull 'Left' and 'left' into Left_ID_to_JPeg.csv and 'Right' and 'right' into Right_ID_JPEG.csv from Source_jpeg folder.

Left list will have 774 items and Right will have 786.

Use Balanced_Filtered_Right_Data.py so the Right list is equal to the Left.

Match_Right_to_Left.py so both will have 774 images

Next, Preprocessing_Left_JPEGS.py will make new file Preprocessed_Left_JPEGS.

Preprocessed_Right_JPEGS.py will create new file Preprocessed_Right_JPEGS.

Data is processed into digital data and analyzed starting with grayscale conversion for removing color, Otsu's method for making the handwriting stand out by converting image to binary format. Then color inversion inverts it back to black writing on white background. Gaussian blur for smoothing the image and reduces noise and then morphological operations removes small artifacts and joins broken lines.

## Microsoft TROCR

Transformer-based Optical Character Recognition was developed to recognize text or images in scanned documents
This program records the digitized images to classify various metrics of the writing including, slant, stroke patterns, curvature, smoothness, and pressure.

Left_JPEGS_Feature_Extraction.py and Right_JPEGS_Feature_Extraction.py 

Curvature was determined to have the highest correlation to handedness.

Visualize_Curvature_Distributions.py for visualizing bar plots for left and right.

The handedness was classified to each individual as 0 for left-handed and 1 for righ-handed and rounded to the closest to make a determination.

Updated_Curvature_Classification.py

# SMOTE

Synthetic Minority Oversampling Technique to balance the data. 

SMOTE.py

#CNN

CNN_Training_with_Smote.py

# Results

The training program was able to use the sample data and predict handedness with 98% accuracy.

# Installation

Running this program requires to use of the folowing programs

pip install os PIL tensoflow numpy pandas transformers cv2 matplotlib imblearn random shutil pathlib 

# Contributors

Contributors to this project include Jordan Graves, Jeff Gary, Toni Akande, Paige Manguiat

# License

This project is licensed under the MIT License - see the LICENSE file for details.
