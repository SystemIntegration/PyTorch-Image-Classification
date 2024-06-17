# Deep Learning Model for Image Classification 🖼️

Welcome to the Image Classification project! This project demonstrates the use of a deep learning model to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Overview 📄

This project is aimed at building and evaluating a CNN model for image classification. The CIFAR-10 dataset, consisting of 10 different classes of images, is used for training and testing the model.

### Dataset 📚

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Training 🏋️‍♂️

The training script is designed to load the CIFAR-10 dataset, define the CNN architecture, and train the model. Key steps include:

   1. Loading Data: Using PyTorch's Dataset and DataLoader classes to handle data loading and batching.
   2.  Model Architecture: Defining the CNN using nn.Module with layers such as Conv2d, ReLU, MaxPool2d, and Linear.
   3.  Loss Function: Using Cross-Entropy Loss to measure the model's performance.
   4.  Optimizer: Using optimizers like SGD or Adam to update the model's parameters.
   5.  Training Loop: Iterating over the dataset, computing the loss, performing backpropagation with loss.backward(), and updating the model parameters.

## Training Results

- After training for 80 epochs, the model achieved:

   -  Validation Accuracy: 89.41%
   -  Class-wise Accuracy:
        1. Plane: 91.40%
        2. Car: 94.90%
        3. Bird: 85.20%
        4. Cat: 79.00%
        5. Deer: 89.50%
        6. Dog: 82.30%
        7. Frog: 92.00%
        8. Horse: 91.60%
        9. Ship: 94.10%
        10. Truck: 94.10%

## Model Testing 🧪

The testing script loads the trained model and evaluates it on new images from the CIFAR-10 dataset. Steps include:

  1. Loading the Trained Model: Loading the model parameters saved in the "cnn.pth" file.
  2. Preprocessing: Preprocessing the input images to match the model's expected input.
  3. Prediction: Using the model to predict the class of the input images.

### Testing Output

Example output:

    - Predicted class: Cat
    - Actual class: Cat

## Improving Model Accuracy 📈

To improve the accuracy of the training model, consider the following strategies:

  - Increase Number of Epochs: Training for more epochs can help the model learn better.
  - Change Optimizer: Experiment with different optimizers like Adam, RMSprop, or AdaGrad.
  - Add More Hidden Layers: Increase the depth of the network to capture more complex patterns.
  - Increase Filters: Use more filters in convolutional layers to capture detailed features.
  - Adjust Batch Size and Learning Rate: Fine-tune these hyperparameters for better performance.
  - Add Padding and Stride: Adjusting padding and stride can help in preserving spatial dimensions and capturing finer details.

## PyTorch Framework 🔧

PyTorch is the main framework used in this project for building and training the CNN model. It provides several important functions and classes:

  - Dataset: Handles loading and preprocessing of data.
  - DataLoader: Efficiently batches and shuffles data for training and evaluation.
  - nn.Linear: Applies a linear transformation to the incoming data.
  - Optimizer: Adjusts model parameters based on gradients.
  - loss.backward(): Computes the gradient of the loss with respect to model parameters.
  - Loss Function: Measures the model's performance (e.g., Cross-Entropy Loss).
  - zero_grad: Clears old gradients to prevent accumulation.
  - Activation Function - ReLU: Introduces non-linearity into the model.
  - Cross Entropy: Combines LogSoftmax and Negative Log Likelihood Loss.
  - Softmax Function: Converts logits into probabilities.
  - Max-Pooling: Reduces the spatial dimensions of the feature maps.

## Getting Started 🚀

To get started with the Image Classification project:

  1. Clone the repository to your local machine.
  2. Install dependencies using pip install -r requirements.txt.
  3. Run the training script to train the model on the CIFAR-10 dataset.
  4. Save the trained model parameters in a file (e.g., "cnn.pth").
  5. Run the testing script to evaluate the model on new images.

## Support and Feedback 📧

- For any issues, feedback, or queries related to the Image Classification project, please contact info@systemintegration.in.