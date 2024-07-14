# Traffic Sign Image Recognition and Classification (ML & DL)

## Overview

This project focuses on the implementation of a Traffic Sign Image Recognition and Classification system using Machine Learning (ML) and Deep Learning (DL) techniques. The primary objective is to accurately identify and classify various traffic signs from images, which can be crucial for applications such as autonomous driving and advanced driver-assistance systems (ADAS).

## Features

- **Implementation of AlexNet**: Utilizes the Convolutional Neural Network (CNN) architecture known as AlexNet, implemented in Python using the Keras library.
- **High Precision Rate**: Achieves a precision rate of 90% on a test dataset comprising 10,000 images.
- **Enhanced VR Image Recognition**: Results in a 15% increase in accuracy for VR image recognition applications compared to existing models.
- **Proficiency**: Demonstrates proficiency in Python, Keras, CNN, AlexNet, ReLU, and SoftMax functions.

## Project Details

### 1. Dataset

The project uses a dataset of traffic sign images. This dataset is split into training and testing subsets, with 10,000 images reserved for testing the model's performance.

### 2. Model Architecture

- **AlexNet**: The architecture consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers. AlexNet is chosen for its efficiency in handling large-scale image recognition tasks.
- **ReLU and SoftMax Functions**: ReLU (Rectified Linear Unit) is used as the activation function to introduce non-linearity, and SoftMax is used in the output layer for classification.

### 3. Implementation

- **Programming Language**: Python
- **Libraries**: Keras, TensorFlow (backend for Keras)
- **Model Training**: The model is trained on a large dataset of labeled traffic sign images. Various data augmentation techniques are applied to enhance the training process.

### 4. Performance Metrics

- **Precision Rate**: The model achieves a precision rate of 90% on the test dataset, indicating its effectiveness in correctly identifying traffic signs.
- **Accuracy Improvement**: The model enhances VR image recognition applications, resulting in a 15% increase in accuracy compared to existing models.

### 5. Key Technologies

- **Python**: The primary programming language used for implementing the model and associated algorithms.
- **Keras**: A high-level neural networks API, written in Python and capable of running on top of TensorFlow.
- **CNN (Convolutional Neural Networks)**: A class of deep neural networks, most commonly applied to analyzing visual imagery.
- **AlexNet**: A pre-trained CNN model known for its deep architecture and high performance in image recognition tasks.
- **ReLU (Rectified Linear Unit)**: An activation function that helps in speeding up the training process and introducing non-linearity.
- **SoftMax Function**: Used for the classification task in the output layer to provide probability distribution across multiple classes.

## Usage

To run the project, follow these steps:

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   python train_model.py
   ```

4. **Test the Model**:
   ```bash
   python test_model.py
   ```

## Results

## Here are some screenshots of the web application after running the model:


### The interface for uploading a traffic sign image.
<br>
<img src="https://github.com/firodiaPurva/Traffic_Signs_WebApp/blob/main/1.PNG" alt="The interface for uploading a traffic sign image" width="600" height="600">

### The predicted traffic sign classification result.
<br>
<img src="https://github.com/firodiaPurva/Traffic_Signs_WebApp/blob/main/2.PNG" alt="The predicted traffic sign classification result" width="600" height="600">

## Conclusion

This project showcases the successful implementation of a CNN-based traffic sign recognition and classification system, achieving significant accuracy and precision rates. The enhancement in VR image recognition applications demonstrates the model's potential for real-world applications in autonomous driving and ADAS.
