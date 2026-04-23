# Image Classification with Deep Learning & Transfer Learning

## Overview

This notebook is a comprehensive guide for building image classification models using **Transfer Learning** and **Deep Neural Networks**. It covers everything from data preparation to model training and evaluation using popular pre-trained architectures.

**Main Goal:** Classify images into **26 different classes** using state-of-the-art deep learning techniques.

---

## Table of Contents

1. [Libraries & Imports](#1-libraries--imports)
2. [Data Preparation & Cleaning](#2-data-preparation--cleaning)
3. [Data Generators & Augmentation](#3-data-generators--augmentation)
4. [Baseline CNN Model](#4-baseline-cnn-model)
5. [Transfer Learning with VGG16/VGG19](#5-transfer-learning-with-vgg16vgg19)
6. [Data Balancing](#6-data-balancing)
7. [Pre-trained Models (ResNet, InceptionV3, EfficientNetB0)](#7-pre-trained-models)
8. [Model Evaluation](#8-model-evaluation)

---

## 1. Libraries & Imports

The notebook starts by importing essential libraries:

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

### Key Libraries:
- **TensorFlow/Keras** - Deep learning framework
- **NumPy & Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **ImageDataGenerator** - Data augmentation and preprocessing

---

## 2. Data Preparation & Cleaning

### Dataset Structure

The notebook works with a dataset split into:
- **Train Set:** `train/` directory with 26 classes
- **Test Set:** `test/` directory with 26 classes

### Data Cleaning Steps

#### 2.1 Verify Class Consistency
```python
train_classes = sorted(os.listdir("train/"))
test_classes = sorted(os.listdir("test/"))

assert train_classes == test_classes, "Classes must be identical"
```

#### 2.2 Remove Unwanted Classes
Classes like `Class_Nothing`, `Class_nothing`, `Class_Space`, and `Class_space` are removed:
```python
REMOVE_CLASSES = ["Class_Nothing", "Class_nothing", "Class_Space", "Class_space"]

def remove_classes(base_dir, remove_list):
    for cls in os.listdir(base_dir):
        if cls in remove_list:
            shutil.rmtree(os.path.join(base_dir, cls))
```

#### 2.3 Clean Checkpoint Files
Jupyter checkpoint directories are removed:
```python
for split in ["train/", "test/"]:
    checkpoint_path = os.path.join(split, ".ipynb_checkpoints")
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
```

---

## 3. Data Generators & Augmentation

### 3.1 General Settings

```python
IMG_SIZE = 224           # Standard size for ResNet & EfficientNet
BATCH_SIZE = 32
NUM_CLASSES = 26
TRAIN_DIR = "train/"
TEST_DIR = "test/"
```

### 3.2 Training Data Augmentation

To improve model generalization, the training data is augmented with:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,                    # Normalize pixel values to [0, 1]
    rotation_range=15,                 # Random rotation
    zoom_range=0.1,                    # Random zoom
    width_shift_range=0.1,             # Horizontal shift
    height_shift_range=0.1,            # Vertical shift
    brightness_range=[0.8, 1.2],       # Brightness adjustment
    horizontal_flip=True,              # Random flip
    fill_mode='nearest'                # Fill missing pixels
)
```

### 3.3 Test Data (No Augmentation)

```python
test_datagen = ImageDataGenerator(rescale=1./255)
```

### 3.4 Data Generators with Validation Split

```python
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'  # 90% for training
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'  # 10% for validation
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)
```

### Dataset Split:
- **Train:** 32,432 images (90% of original train set)
- **Validation:** 3,587 images (10% of original train set)
- **Test:** 253 images

### 3.5 Batch Preview

Visualize sample images from the training set to verify data loading:

```python
images, labels = next(train_generator)
# Display first 6 images with their labels
```

---

## 4. Baseline CNN Model

A simple custom CNN is built as a baseline to compare against transfer learning models:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Training

```python
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator
)
```

### Evaluation

```python
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
```

---

## 5. Transfer Learning with VGG16/VGG19

### 5.1 Load Pre-trained Base Model

```python
from tensorflow.keras.applications import VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Or use VGG19:
# base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

### 5.2 Add Custom Dense Layers

```python
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])
```

### 5.3 Freeze Base Layers

To prevent overfitting and reduce training time, freeze the pre-trained weights:

```python
for layer in base_model.layers:
    layer.trainable = False
```

### 5.4 Compile & Train

```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)
```

### 5.5 Fine-tuning (Optional)

Unfreeze the last few layers for fine-tuning:

```python
for layer in base_model.layers[-4:]:
    layer.trainable = True
```

---

## 6. Data Balancing

### 6.1 Check Class Distribution

Analyze the number of images per class:

```python
def count_images_per_class(folder_path):
    class_counts = {}
    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)
        if os.path.isdir(class_path):
            images = [img for img in os.listdir(class_path) 
                     if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)
    return class_counts

train_counts = count_images_per_class("train/")
```

### 6.2 Balance Dataset Using Data Augmentation

If some classes have fewer images, generate synthetic samples:

```python
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

for class_name in sorted(os.listdir("train/")):
    class_path = os.path.join("train/", class_name)
    images = os.listdir(class_path)
    current_count = len(images)
    TARGET = 389

    if current_count < TARGET:
        needed = TARGET - current_count
        for i in range(needed):
            img = load_img(os.path.join(class_path, images[i % len(images)]))
            x = img_to_array(img)
            x = datagen.random_transform(x)
            save_img(os.path.join(class_path, f"aug_{current_count}.jpg"), x)
            current_count += 1
```

### 6.3 Reduce Oversized Classes

For classes with too many images, randomly sample:

```python
for class_name in os.listdir("train/"):
    class_path = os.path.join("train/", class_name)
    images = os.listdir(class_path)
    if len(images) > TARGET_COUNT:
        images_to_keep = random.sample(images, TARGET_COUNT)
        for img in images:
            if img not in images_to_keep:
                os.remove(os.path.join(class_path, img))
```

---

## 7. Pre-trained Models

The notebook demonstrates transfer learning with multiple architectures:

### Available Models:
1. **VGG16 / VGG19** - Good starting point, simpler architecture
2. **ResNet50** - More powerful, better performance
3. **InceptionV3** - Complex, requires careful training
4. **EfficientNetB0** - Efficient and balanced

### Example: Loading Different Models

```python
from tensorflow.keras.applications import (
    VGG16, VGG19, ResNet50, InceptionV3, EfficientNetB0
)

# ResNet50 Example
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# EfficientNetB0 Example
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# InceptionV3 Example
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
```

---

## 8. Model Evaluation

### 8.1 Visualize Training History

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### 8.2 Test Set Evaluation

```python
loss, acc = model.evaluate(test_generator)
print(f"Test Accuracy: {acc*100:.2f}%")
```

### 8.3 Visualize Class Distribution

```python
plt.figure(figsize=(12, 5))
plt.bar(df["Class"], df["Train Images"], color='skyblue')
plt.xticks(rotation=90)
plt.ylabel("Number of Images")
plt.title("Train Images per Class After Balancing")
plt.show()
```

---

## Key Concepts

### Transfer Learning
- Use pre-trained weights from ImageNet
- Freeze early layers (learned general features)
- Train only the top layers for your specific task
- Reduces training time and improves performance

### Data Augmentation
- Increases dataset size artificially
- Improves model generalization
- Prevents overfitting on small datasets

### Class Imbalance
- Different classes may have different amounts of data
- Balance using augmentation or undersampling
- Ensures fair training across all classes

---

## Best Practices

1. **Start with VGG16/VGG19** - Easy to implement and understand
2. **Use Data Augmentation** - Improves generalization
3. **Freeze Base Layers Initially** - Preserves pre-trained knowledge
4. **Use Validation Set** - Monitor overfitting during training
5. **Fine-tune Last Layers** - Adapt to your specific task
6. **Balance Dataset** - Equal representation of all classes

---

## Summary

This notebook provides a complete pipeline for image classification:
- ✅ Data preparation and cleaning
- ✅ Data augmentation and generators
- ✅ Baseline CNN model
- ✅ Transfer learning with multiple architectures
- ✅ Data balancing techniques
- ✅ Model training and evaluation

The approach is flexible and can be adapted for different image classification tasks by adjusting the number of classes, image size, and model architecture.

---

## Notes

- Image size of **224×224** is standard for ResNet and EfficientNet
- **InceptionV3** requires **299×299** images
- The notebook uses **26 classes** but can be modified for any number of classes
- Pre-trained weights from **ImageNet** provide excellent starting features
#   I m a g e _ C l a s s i f i c a t i o n  
 