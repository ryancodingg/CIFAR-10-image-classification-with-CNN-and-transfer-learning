# CIFAR-10 Image Classification with CNN and Transfer Learning

This project focuses on classifying images from the CIFAR-10 dataset using deep learning models. The CIFAR-10 dataset contains 60,000 32x32 RGB images in 10 categories, with 50,000 training images and 10,000 test images. The goal of this project is to train deep learning models to classify these images with high accuracy.

## Project Overview

### Steps Involved:
1. **Dataset Loading and Preprocessing**:
   - The CIFAR-10 dataset is loaded using TensorFlow Datasets (TFDS) and preprocessed by normalizing the pixel values to a range of 0 to 1.
   - Data augmentation is applied to the training set to improve model generalization.
   - The dataset is split into training and test sets, with batching and shuffling applied to optimize training.

2. **Model Development**:
   - A **Convolutional Neural Network (CNN)** model is developed using TensorFlow/Keras, with convolutional layers, pooling layers, and dense layers for classification.
   - **ResNet18**, a deep residual network pre-trained on ImageNet, is used for transfer learning. The final layers are fine-tuned to output 10 classes corresponding to the CIFAR-10 categories.

3. **Evaluation**:
   - The models are evaluated using **classification accuracy**, **precision**, **recall**, and **F1-score**.
   - A **confusion matrix** is used to analyze the performance of the model across different categories, particularly focusing on visually similar categories like "cat" vs. "dog".

4. **Results**:
   - The CNN model achieved **90.9% accuracy** during training but experienced overfitting, with the test accuracy reaching only **73.4%**.
   - The **ResNet18 model** demonstrated significant improvements, with **training accuracy of 96.14%** and **validation accuracy of 81.06%**.
   - Misclassifications mostly occurred between classes with similar features, such as cats, birds, and deer.

### Key Observations:
- Categories with clear, structured features (e.g., **ship**, **automobile**, **truck**) performed well.
- Visually similar categories (e.g., **cats**, **birds**, **frogs**) resulted in frequent misclassifications.
- Techniques like **data augmentation**, **class-specific fine-tuning**, and **feature extraction** are necessary for improving performance on challenging classes.

### Conclusion:
This project demonstrates how deep learning models, particularly with the help of transfer learning using **ResNet18**, can significantly improve classification performance on the CIFAR-10 dataset. While the model achieved strong performance on well-defined classes, there is room for improvement in distinguishing between similar categories. By incorporating advanced techniques like **data augmentation** and **fine-tuning**, the model’s ability to generalize to these challenging categories can be enhanced.

## Getting Started

### Prerequisites:
- Python 3.x
- TensorFlow, Keras, and PyTorch
- Other necessary Python libraries (NumPy, Matplotlib, etc.)

### Installation:
1. Clone the repository:
   ```bash
   git clone https://github.com/ryancodingg/CIFAR-10-image-classification-with-CNN-and-transfer-learning
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the project:
* Open the project in your preferred IDE or Jupyter Notebook.
* Follow the instructions in the provided code to train and evaluate the model.
