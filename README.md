# BeauSkin: Skin Type and Acne Detection Image Classification

This repository contains a Jupyter notebook that demonstrates how to build, train, convert, and use a machine learning model for skin type and acne detection using TensorFlow and Keras. The model classifies images into categories such as oily, dry, and normal skin types, and detects whether the skin is acne-prone.

## Table of Contents
- [Project Description](#project-description)
- [Setup and Installation](#setup-and-installation)
- [Notebook Overview](#notebook-overview)
  - [Importing Libraries](#importing-libraries)
  - [Data Preprocessing](#data-preprocessing)
  - [Building the Model](#building-the-model)
  - [Training the Model](#training-the-model)
  - [Model Conversion](#model-conversion)
  - [Making Predictions](#making-predictions)
- [Files](#files)

## Project Description
This project focuses on classifying skin types (oily, dry, and normal) and detecting acne-prone skin using Convolutional Neural Networks (CNN) implemented in TensorFlow and Keras. After training, the model is saved in `.h5` format, making it ready for deployment. The notebook also includes code for making predictions on new images, providing a complete pipeline from data preparation to model inference.

## Setup and Installation

### Clone the Repository:
To get started, clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/beauskin.git
cd beauskin
```

### Create a Virtual Environment and Activate It:
Set up a virtual environment to manage the dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install the Required Dependencies:
Install all the necessary dependencies by running:
```bash
pip install -r requirements.txt
```

## Notebook Overview

### Importing Libraries
The notebook begins by importing the necessary libraries for TensorFlow, Keras, and image processing. This includes modules for loading and preprocessing images, defining the CNN model architecture, and performing data augmentation to improve the model’s ability to generalize.

### Data Preprocessing
Paths to the training and validation datasets are defined, and the `ImageDataGenerator` class is used to perform data augmentation on the training data. This augmentation includes transformations like rotation, zoom, and horizontal flips, which help improve the robustness of the model. For validation data, only rescaling is applied to standardize the image pixel values.

### Building the Model
A Convolutional Neural Network (CNN) is created using Keras' Sequential API. The model is composed of several convolutional layers followed by max-pooling layers, which are used to extract important features from the images. The model then flattens the extracted features and feeds them through dense (fully connected) layers for classification. The final output layer uses a softmax activation function for multi-class classification (skin types: oily, dry, normal) and a sigmoid activation function for binary classification (acne detection).

### Training the Model
The model is compiled with a loss function suited for categorical classification, an optimizer, and performance metrics. The training process fits the model to the training dataset while validating on the validation dataset. Once training is complete, the model’s weights are saved in `.h5` format for future use or fine-tuning.

### Model Conversion
After training, the Keras model is saved in `.h5` format for deployment. Additionally, a separate object detection model for acne detection is saved in `.pt` format, making it compatible with PyTorch-based deployment systems.

### Making Predictions
The notebook includes code to load the saved model and make predictions on new skin images. The images are preprocessed to match the input size expected by the model. The model predicts the skin type (oily, dry, or normal) and whether the skin is acne-prone, displaying the result along with the confidence level. Class labels are stored in a text file.

## Files
The repository contains the following files:
- `acne_grade.ipynb` - Jupyter notebook for training and predicting acne grades.
- `acne_types_detection.ipynb` - Jupyter notebook for acne type detection model.
- `chatbot.ipynb` - Jupyter notebook for the skincare chatbot model.
- `combine.ipynb` - Jupyter notebook that combines the results from the skin type and acne detection models.
- `datasets.md` - Documentation file for dataset download links.
- `requirements.txt` - Required dependencies for the project.
- `skin_types_classification.ipynb` - Jupyter notebook for training and predicting skin types (oily, dry, normal).
