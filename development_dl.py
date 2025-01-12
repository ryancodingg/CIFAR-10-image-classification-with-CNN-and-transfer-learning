# -*- coding: utf-8 -*-
"""development_DL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11sVo1s8De25nvXg2fSGtEPlTSV6Ep4Be

#**Import**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from IPython.display import Image
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout,Input, MaxPooling2D,RandomFlip, RandomRotation, RandomZoom, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix
import scipy.io
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToPILImage

"""## Preprocessing the dataset"""

# Preprocessing function
def map_image(image, label):
    # Normalize the image
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    return image, label  # Return the preprocessed image and label

# Parameters
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024

# Load and preprocess the CIFAR-10 training dataset
train_dataset = tfds.load('cifar10', as_supervised=True, split="train")

# Preprocess the dataset using the `map_image()` function
train_dataset = train_dataset.map(map_image)

# Shuffle and batch the dataset
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

# Load and preprocess the CIFAR-10 test dataset
test_dataset = tfds.load('cifar10', as_supervised=True, split="test")

# Preprocess the test dataset using the `map_image()` function
test_dataset = test_dataset.map(map_image)

# Batch the test dataset
test_dataset = test_dataset.batch(BATCH_SIZE)

# Count samples in train and test datasets
train_count = len(list(train_dataset))
test_count = len(list(test_dataset))

# Load CIFAR10 dataset
train_dataset = tfds.load('cifar10', as_supervised=True, split="train")
test_dataset = tfds.load('cifar10', as_supervised=True, split="test")

# Count samples in train and test datasets
train_count = len(list(train_dataset))
test_count = len(list(test_dataset))

print(f"Number of training images: {train_count}")
print(f"Number of test images: {test_count}")

"""## **Formulate the problem:**
- **Input**:
  - Images: 32x32 RGB images (3 channels).
  - Shape: (32, 32, 3).

- **Output**:
  - Classes: One of 10 categories (airplane, automobile, ..., truck).
  - Output Format: A probability distribution over 10 classes.
  - Activation: The output layer uses a softmax activation function to produce class probabilities.

- **Objective**:
  - Minimize a suitable loss function for classification, such as categorical cross-entropy or sparse categorical cross-entropy.

- **Evaluation Metrics**:
  - Primary: Classification Accuracy.
  - Optional: Precision, Recall, F1-score, and Confusion Matrix to analyze per-class performance.

- Deep Learning Framework: TensorFlow, PyTorch (pre-trained model)

## **Data Augmnetation & Partitioning**
"""

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),           # Randomly flip images horizontally
    RandomRotation(0.1),                # Random rotation by ±10%
    RandomZoom(0.1)                     # Random zoom by ±10%
])

# Apply data augmentation during preprocessing
def augment_image(image, label):
    image = data_augmentation(image)    # Apply augmentation
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    return image, label

# Apply to training dataset only
train_dataset = train_dataset.map(augment_image).shuffle(1024).batch(128)
test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).batch(128)

# Data Partitioning
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Print dataset shapes
print(f"Training Images Shape: {train_images.shape}")
print(f"Training Labels Shape: {train_labels.shape}")
print(f"Test Images Shape: {test_images.shape}")
print(f"Test Labels Shape: {test_labels.shape}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize original images
plt.figure(figsize=(8, 8))
for i in range(16):  # Display 16 images
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[i])  # Plot original training images

    # Extract the scalar value for the label
    label = train_labels[i][0] if len(train_labels[i]) > 0 else train_labels[i]

    # Set the title with class names
    plt.title(class_names[label], fontsize=8)  # Add class names
    plt.axis('off')  # Remove axes for better visuals

plt.tight_layout()
plt.suptitle("", fontsize=16)
plt.show()

# Define image transformations (augmentation and normalization)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images
    transforms.RandomCrop(32, padding=4),  # Randomly crop with padding
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize image to [-1, 1]
])

# Load CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(root='./data', train=True,
                                 transform=transform, download=True)

# Create DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Get a batch of augmented images
data_iter = iter(train_loader)
images, labels = next(data_iter)

# Plot augmented images
plt.figure(figsize=(8, 8))
for i in range(16):  # Display 16 images
    plt.subplot(4, 4, i + 1)
    image = images[i].permute(1, 2, 0)  # Convert CHW to HWC format
    image = (image * 0.5 + 0.5).numpy()  # Denormalize the image
    plt.imshow(image)
    plt.title(class_names[labels[i].item()])  # Get class name for the label
    plt.axis('off')

plt.tight_layout()
plt.show()

# Define the augmentation pipeline with resizing
aug_transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert tensor to PIL image
    transforms.Resize(40),  # Resize to a size larger than 32x32 to allow cropping
    transforms.RandomCrop(32, padding=4),  # Crop to 32x32 with padding
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation
    transforms.ToTensor(),  # Convert back to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Apply augmentations to the batch of images
aug_images = torch.stack([aug_transform(images[i]) for i in range(len(images))])  # Apply to each image in the batch

# Compare original vs augmented images
plt.figure(figsize=(10, 10))
for i in range(8):  # Display 8 pairs of original and augmented images
    # Original image
    plt.subplot(8, 2, 2 * i + 1)
    orig_image = images[i].permute(1, 2, 0)  # Convert CHW to HWC
    orig_image = (orig_image * 0.5 + 0.5).numpy()  # Denormalize
    plt.imshow(orig_image)
    plt.title(f"Original: {class_names[labels[i].item()]}")
    plt.axis('off')

    # Augmented image
    plt.subplot(8, 2, 2 * i + 2)
    aug_image = aug_images[i].permute(1, 2, 0)  # Convert CHW to HWC
    aug_image = (aug_image * 0.5 + 0.5).numpy()  # Denormalize
    plt.imshow(aug_image)
    plt.title(f"Augmented")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Split the training data into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42
)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

# Shuffle, batch, and prefetch datasets for efficient training
BATCH_SIZE = 128
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Verify dataset sizes
print(f"Train Dataset: {len(list(train_dataset))} batches")
print(f"Validation Dataset: {len(list(val_dataset))} batches")
print(f"Test Dataset: {len(list(test_dataset))} batches")

# Split the training data into training and validation sets (90-10 split)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, random_state=42
)

# Print dataset shapes
print(f"New Training Images Shape: {train_images.shape}")
print(f"Validation Images Shape: {val_images.shape}")
print(f"Training Labels Shape: {train_labels.shape}")
print(f"Validation Labels Shape: {val_labels.shape}")

# Define the model
model = Sequential()

# Downsampling layers (Convolution + MaxPooling)
model.add(Conv2D(64, input_shape=(32, 32, 3), kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))

# Flatten and Dense layers for classification
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Regularization
model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()

## Train the model
history = model.fit(train_images, train_labels,
                    epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model on both training and test sets
train_loss, train_accuracy = model.evaluate(train_images, train_labels, verbose=2)
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)

print(f'Training Accuracy: {train_accuracy:.3f}')
print(f'Test Accuracy: {test_accuracy:.3f}')

# Get predictions on the test set
test_predictions = model.predict(test_dataset)
predicted_classes = np.argmax(test_predictions, axis=1)

# Get true labels from the test dataset
true_labels = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(true_labels, predicted_classes, target_names=[
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]))

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Get a batch of test images and labels
test_images_batch, test_labels_batch = next(iter(test_dataset.unbatch().batch(16)))

# Get model predictions for this batch
predictions = model.predict(test_images_batch)
predicted_classes = np.argmax(predictions, axis=1)

# Plot some test images with their true and predicted labels
plt.figure(figsize=(8, 8))
for i in range(16):  # Display 16 images
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_images_batch[i].numpy())

    # Convert labels to scalars
    true_label = class_names[test_labels_batch[i].numpy().item()]  # Extract scalar
    predicted_label = class_names[predicted_classes[i]]

    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()

"""## **Model 2: CNN Model using Pytorch**

"""

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations for CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize with CIFAR-10 stats
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split training data into training and validation sets (90% train, 10% validation)
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Data loaders
BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Load pre-trained ResNet18 and modify for CIFAR-10
model = models.resnet18(pretrained=True)  # Load pre-trained ResNet18
model.fc = nn.Linear(model.fc.in_features, 10)  # Replace the final layer for CIFAR-10 (10 classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = outputs.max(1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Print epoch metrics
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Print epoch metrics in the desired format
print(f"{len(train_loader)}/{len(train_loader)} - {int(1000 / len(train_loader))}ms/step - accuracy: {train_accuracy:.3f} - loss: {train_loss:.4f}")
print(f"{len(val_loader)}/{len(val_loader)} - {int(1000 / len(val_loader))}ms/step - accuracy: {val_accuracy:.3f} - loss: {val_loss:.4f}")
print(f"Epoch {epoch + 1}/{num_epochs}, Train Accuracy: {train_accuracy:.3f}, Val Accuracy: {val_accuracy:.3f}")


# Evaluation on test dataset
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Ensure the model is in evaluation mode
model.eval()

# Initialize lists to store predictions and true labels
all_predictions = []
all_true_labels = []

# Collect predictions and true labels
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_true_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_predictions = np.array(all_predictions)
all_true_labels = np.array(all_true_labels)

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_true_labels, all_predictions)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Classification report
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
print("Classification Report:")
print(classification_report(all_true_labels, all_predictions, target_names=class_names))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Get a batch of test images and labels
data_iter = iter(test_loader)
test_images_batch, test_labels_batch = next(data_iter)

# Move the images to the device (GPU/CPU)
test_images_batch = test_images_batch.to(device)

# Get model predictions for the batch
model.eval()
with torch.no_grad():
    outputs = model(test_images_batch)
    predicted_classes = torch.argmax(outputs, dim=1)

# Convert images and labels to CPU for visualization
test_images_batch = test_images_batch.cpu()
test_labels_batch = test_labels_batch.cpu()
predicted_classes = predicted_classes.cpu()

# Plot the test images with their true and predicted labels
plt.figure(figsize=(8, 8))
for i in range(16):  # Display 16 images
    plt.subplot(4, 4, i + 1)
    image = test_images_batch[i].permute(1, 2, 0)  # Convert from CHW to HWC format
    image = (image * 0.5 + 0.5).numpy()  # Denormalize the image
    plt.imshow(image)

    # Convert labels to class names
    true_label = class_names[test_labels_batch[i].item()]
    predicted_label = class_names[predicted_classes[i].item()]

    plt.title(f"True: {true_label}\nPred: {predicted_label}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()

class_accuracies = {}
for i, class_name in enumerate(class_names):
    class_correct = (all_predictions[all_true_labels == i] == i).sum()
    class_total = (all_true_labels == i).sum()
    class_accuracy = 100 * class_correct / class_total
    class_accuracies[class_name] = class_accuracy
    print(f"Accuracy for {class_name}: {class_accuracy:.2f}%")