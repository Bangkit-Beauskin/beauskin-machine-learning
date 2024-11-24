# BeauSkin: Skin Type and Acne Detection Image Classification

This repository contains Jupyter notebooks that demonstrate how to build, train, convert, and use a machine learning model for skin type classification and acne detection (grades and types) using TensorFlow and Keras. The model classifies skin types into categories such as oily, dry, and normal skin, and detects acne grades and types. Additionally, a chatbot for skin care guidance is included.

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
This project focuses on classifying skin types (oily, dry, and normal) and detecting acne grades and types using Convolutional Neural Networks (CNNs) implemented in TensorFlow and Keras. The notebook provides a complete pipeline, from data preprocessing to model deployment. The model is trained to classify skin types, detect acne grades, and identify acne types, with the model's weights saved in `.h5` format for skin type classification and in `.pt` format for acne type detection. Additionally, a chatbot is integrated for skincare recommendations.

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

Alternatively, if you're using Google Colab, you can install the dependencies directly within the Colab environment using:
```python
!pip install -r requirements.txt
```

## Notebook Overview

### Importing Libraries
The notebook begins by importing the necessary libraries for TensorFlow, Keras, and image processing. This includes modules for loading and preprocessing images, defining the CNN model architecture, and performing data augmentation to improve the model’s ability to generalize.

### Data Preprocessing
Paths to the training and validation datasets are defined, and the `ImageDataGenerator` class is used to perform data augmentation on the training data. This augmentation includes transformations like rotation, zoom, and horizontal flips, which help improve the robustness of the model. For validation data, only rescaling is applied to standardize the image pixel values.

### Building the Model
A Convolutional Neural Network (CNN) is created using Keras' Sequential API for skin type classification and acne grade detection. The model is composed of several convolutional layers followed by max-pooling layers, which are used to extract important features from the images. The model then flattens the extracted features and feeds them through dense (fully connected) layers for classification. Both for skin type classification and acne grade detection, the model uses a softmax activation function for multi-class classification.

For the chatbot component, an LSTM (Long Short-Term Memory) network is used to handle the sequential nature of text data. This allows the chatbot to process and generate appropriate responses based on the input text.

### Training the Model
The model is compiled with different loss functions, optimizers, and performance metrics based on the task at hand. For skin type classification, the model uses Stochastic Gradient Descent (SGD) with Nesterov Accelerated Gradient (NAG) as the optimizer. For acne grade detection, the Adam optimizer is used. The training process fits the model to the training dataset while validating on the validation dataset. Once training is complete, the model’s weights are saved in .h5 format for skin type classification and in .pt format for acne type detection.

### Model Conversion
After training, the Keras model for skin type classification is saved in `.h5` format, which is ideal for deployment and future use. Additionally, the acne type detection model is saved in `.pt` format to make it compatible with PyTorch-based deployment systems.

### Making Predictions
The notebook includes code to load the saved models and make predictions on new skin images. The images are preprocessed to match the input size expected by the models. The models predict the skin type (oily, dry, or normal), acne grades, and acne types, displaying the results along with the confidence levels. Class labels are stored in a text file.

## Files
The repository contains the following files:
- `acne_grade.ipynb` - Jupyter notebook for training and predicting acne grades.
- `acne_types_detection.ipynb` - Jupyter notebook for acne type detection model, saved as `.pt`.
- `chatbot.ipynb` - Jupyter notebook for the skincare chatbot model that provides personalized skincare recommendations.
- `combine.ipynb` - Jupyter notebook that combines the results from the skin type and acne detection models.
- `datasets.md` - Documentation file for dataset download links.
- `requirements.txt` - Required dependencies for the project.
- `skin_types_classification.ipynb` - Jupyter notebook for training and predicting skin types (oily, dry, normal).
