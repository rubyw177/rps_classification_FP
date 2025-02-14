# Machine Learning Final Project

## Overview
This project implements a machine learning model for classifying images in the Rock-Paper-Scissors dataset. The dataset consists of labeled images representing hand gestures of rock, paper, and scissors. The goal is to train a deep learning model using TensorFlow to recognize and classify these gestures.

## Features
- Downloads and preprocesses the Rock-Paper-Scissors dataset
- Implements a Convolutional Neural Network (CNN) using TensorFlow
- Trains the model on the dataset and evaluates performance
- Uses data augmentation techniques to enhance model robustness
- Provides accuracy metrics and classification reports

## Dataset
The dataset is obtained from Dicoding and consists of images of hand gestures for rock, paper, and scissors. It is automatically downloaded and extracted using the following command:

```python
!wget --no-check-certificate \
  "https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip" \
  -O /tmp/rockpaperscissors.zip
```

The dataset is then extracted and stored in `/tmp` for processing.

## Dependencies
Ensure you have the following Python packages installed:

```bash
pip install tensorflow numpy matplotlib zipfile
```

## Model Implementation
1. **Import necessary libraries:**
   ```python
   import tensorflow as tf
   import zipfile, os
   import numpy as np
   import matplotlib.pyplot as plt
   ```
2. **Download and extract dataset:**
   ```python
   loc_zip = "/tmp/rockpaperscissors.zip"
   zip_ref = zipfile.ZipFile(loc_zip, "r")
   zip_ref.extractall("/tmp")
   zip_ref.close()
   ```
3. **Preprocess and augment data:**
   - Data is split into training and validation sets
   - Data augmentation is applied to improve model generalization
4. **Build and compile CNN model:**
   ```python
   model = tf.keras.models.Sequential([
       tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
       tf.keras.layers.MaxPooling2D(2,2),
       tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
       tf.keras.layers.MaxPooling2D(2,2),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(512, activation='relu'),
       tf.keras.layers.Dense(3, activation='softmax')
   ])
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   ```
5. **Train the model and evaluate accuracy.**
6. **Make predictions on new images and visualize results.**

## Evaluation
The model's performance is evaluated using accuracy metrics and loss curves. The final accuracy of the model depends on hyperparameters and training duration.
