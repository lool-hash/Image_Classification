# Project: Image Classification with Deep Learning & Transfer Learning

## Overview

This notebook is a comprehensive guide for building image classification models using **Transfer Learning** and **Deep Neural Networks**. It covers data preparation, training, and evaluation with popular pre-trained architectures.

**Main Goal:** Classify images into **26 different classes** using state-of-the-art techniques.

---

## Table of Contents

1. Libraries & Imports
2. Data Preparation & Cleaning
3. Data Generators & Augmentation
4. Baseline CNN Model
5. Transfer Learning with VGG16/VGG19
6. Data Balancing
7. Pre-trained Models
8. Model Evaluation

---

## 1. Libraries & Imports

```python
# Data & Numerical
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3, VGG16, VGG19
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
```

**Key Libraries:**

* TensorFlow/Keras – deep learning framework
* NumPy & Pandas – data manipulation
* Matplotlib & Seaborn – visualization
* ImageDataGenerator – data augmentation

---

## 2. Data Preparation & Cleaning

* **Train Set:** `train/` directory with 26 classes
* **Test Set:** `test/` directory with 26 classes

### Steps:

1. Verify class consistency between train and test sets
2. Remove unwanted classes (e.g., `Class_Nothing`, `Class_Space`)
3. Clean checkpoint directories

---

## 3. Data Generators & Augmentation

```python
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 26
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
```

**Augmentation Techniques:**

* Random rotations (±15 degrees)
* Zoom
* Width & height shifts
* Brightness adjustment
* Horizontal flips
* Normalization to [0,1]

**Dataset Split:**

* Train: 90% (32,432 images)
* Validation: 10% (3,587 images)
* Test: 253 images

---

## 4. Baseline CNN Model

* 3 convolutional blocks + pooling
* Flatten layer
* Dense layers with dropout
* Softmax output for 26 classes

Baseline for comparison with transfer learning models.

---

## 5. Transfer Learning with VGG16/VGG19

Steps:

1. Load pre-trained VGG16/VGG19 (ImageNet)
2. Freeze base layers
3. Add custom dense layers
4. Train top layers
5. Optional fine-tuning with low learning rate

---

## 6. Data Balancing

Techniques:

* Count images per class
* Augmentation for underrepresented classes
* Undersampling for oversized classes
* Target: equal samples per class (e.g., 389)

---

## 7. Pre-trained Models

* **VGG16/VGG19:** Simple, 224x224 input, easy for transfer learning
* **ResNet50:** Skip connections, better generalization, 224x224, Test Accuracy: 95.26%
* **InceptionV3:** Multi-scale features, input 299x299
* **EfficientNetB0:** Efficient scaling, suitable for mobile/edge deployment

---

## 8. Model Evaluation

Metrics:

* Accuracy
* Confusion Matrix
* Classification Report
* Training history plots

Model selection based on high test accuracy, low overfitting, balanced class performance.

---

## Key Concepts

* **Transfer Learning:** Use pre-trained weights, freeze early layers, train top layers
* **Data Augmentation:** Rotation, zoom, shifts, brightness, flips
* **Class Imbalance:** Handle with augmentation, undersampling, or class weights

---

## Best Practices

1. Start simple (VGG16) and progress to ResNet50
2. Use data augmentation
3. Freeze base layers initially
4. Monitor validation set for overfitting
5. Fine-tune carefully
6. Balance dataset
7. Save models and checkpoints

---

## Notes

* Standard image size: 224x224 (except InceptionV3: 299x299)
* 26 classes used, adaptable to more
* ImageNet pre-trained weights for faster training and better performance

---

## Summary

Pipeline includes:

* Data preparation & cleaning
* Data augmentation & generators
* Baseline CNN
* Transfer learning
* Data balancing
* Model training & evaluation
* Model comparison & recommendations
