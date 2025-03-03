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

## MNIST

# preprocessing, augmentation, normalization

Preprocessing

JPEG dataset is resized and converted to png. 
preprocessing.py

Augmentation

Data is processed into digital data and analyzed starting with the letter identification. Binary image with threshold converts the image to negative and shows the pressure used in each stroke.

Normalization

Additional adjustments and augmentation is used to normalize each document to digitize the data.

## Microsoft TROCR

Transformer-based Optical Character Recognition was developed to recognize text or images in scanned documents
This program records the digitized images to classify various metrics of the writing including, slant, stroke patterns, curvature, smoothness, and pressure.

Curvature was determined to have the highest correlation to handedness.

The handedness was classified to each individual as 0 for left-handed and 1 for righ-handed and rounded to the closest to make a determination.

# Model Architecture

Using np.random.seed(42) and tf.random.set_seed(42) ensures consistent results across runs. Reproducibility!
Conv2d extracts spatial features, MaxPooling2D reduces spatial dimensions to prevent overfitting, Flatten converts 2D to 1D, and Dense fully connects layers
Input shape 28, 28, 1. 28 pixel square and 1 indicates single channel which is grayscale
5 epochs for computational efficiency and prevents overfitting

cnn_train.py

# Results

The training program was able to use the sample data and predict handedness with 100% accuracy.

# Installation

Running this program requires to use of the folowing programs

pip install os PIL tensoflow numpy pandas transformers cv2 matplotlib

# Contributors

Contributors to this project include Jordan Graves, Jeff Gary, Toni Akande, Paige Manguiat

# License

This project is licensed under the MIT License - see the LICENSE file for details.
