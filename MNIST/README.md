# MNIST Classification using Convolutional Neural Network with PyTorch 
In this sample demo project, we perform `multi label classification` on the famous MNIST dataset which contains 10 classes of handwritten digits. We are going to make use of neural networks, specifically convolutional neural networks using PyTorch.

<img src="https://github.com/navti/DLWithPyTorch/blob/main/MNIST/assets/mnist.png?raw=true">

## Table of Contents
- [Introduction](#introduction) 
- [About the Dataset](#about-the-dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [The Network](#the-network)
- [How To Use](#how-to-use)
- [License](#license)
- [Get in touch](#get-in-touch)

## Introduction
The MNIST (Modified National Institute of Standards and Technology) dataset is a widely used benchmark in machine learning, particularly in the field of image recognition and classification. 1  It consists of a collection of handwritten digits (0-9) represented as 28x28 pixel grayscale images. 1  The dataset is divided into a training set of 60,000 images and a test set of 10,000 images, making it a valuable resource for researchers and practitioners to train and evaluate their models. 1  MNIST's simplicity and well-defined nature have made it an ideal starting point for exploring various machine learning algorithms, from simple linear classifiers to deep neural networks. 2  Due to its extensive use and well-documented characteristics, MNIST serves as a common ground for comparing different approaches and understanding the performance of various machine learning models.  

## About the Dataset
- Large Dataset: Contains 70,000 images (60,000 training, 10,000 testing) providing ample data for model training and evaluation.
- Simple Images: 28x28 pixel grayscale images, making them computationally manageable and suitable for various machine learning algorithms.
- Well-Defined Task: Clear objective of classifying handwritten digits (0-9), allowing for straightforward performance measurement.
- Standardized Format: Consistent image size and format across the dataset, simplifying data preprocessing and model development.
- Widely Used: A benchmark in machine learning, making it easy to compare results with other researchers and algorithms.
- Easy to Access: Readily available from various sources, including libraries like TensorFlow and PyTorch.
- Suitable for Beginners: Its simplicity makes it an excellent starting point for learning about image recognition and machine learning concepts.
- Preprocessed Data: Images are preprocessed and centered, reducing the need for extensive data cleaning.

- Here are a few samples of digits from the training dataset with their respective labels.

<img src="https://github.com/navti/DLWithPyTorch/blob/main/MNIST/assets/mnist_samples.png?raw=true">


## Evaluation Criteria

### The Loss Function  
Since we are doing a multi label classification, one appropriate loss function to consider is the Cross Entropy Loss. Cross-entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross-entropy loss increases as the predicted probability diverges from the actual label. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

In binary classification, where the number of classes M equals 2, cross-entropy can be calculated as:
$$-{(y\log(p) + (1 - y)\log(1 - p))}$$

If M>2 (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.
$$-\sum_{c=1}^My_{o,c}\log(p_{o,c})$$

### Performance Metric: Accuracy

Accuracy is a fundamental performance metric used to evaluate the overall correctness of a machine learning model, particularly in classification tasks. It measures the proportion of correctly predicted instances to the total number of instances in the dataset.

$$Accuracy = \frac{(Number\ of\ Correct\ Predictions)}{(Total\ Number\ of\ Predictions)} * 100%$$

## The Network
- The network contains two parts: Convolution layers and dense layers.
- The Conv layers make use of 3X3 kernels and Max Pool with stride 2.
- The 3x3 convolutions along with strided max pooling quickly downsample the image while extracting useful features. The last conv layer is then flattened and sent to the dense part of the network.
- The dense layer is a simple fully connected network with 576 and 64 length tensors. These are the defaults. The size can change depending on the no. of filters used during convolution.
- The dense layer terminates into a classification head with 10 neurons that output the class probabilities for the input digits.

<img src="https://github.com/navti/DLWithPyTorch/blob/main/MNIST/assets/network.png?raw=true">

- Network is trained and validated for ten epochs using the `CrossEntropyLoss` function and `Adam` optimizer with a learning rate of 0.01.
- We keep track of training and validation losses. A sample plot of the train and validation loss curves are shown below.

<img src="https://github.com/navti/DLWithPyTorch/blob/main/MNIST/assets/loss_plot.png?raw=true">

## How To Use
Ensure the below-listed packages are installed
- `matplotlib`
- `torch`
- `torchvision`
- `pathlib`
- `argparse`

### To run the training:
```python
python main.py --lr 0.01 --batch-size 64 --nf 32 --epochs 30 --save-model
```

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

# Get in touch
[![email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:navintiwari08@gmail.com)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/navti/)

[Back To The Top](#MNIST-Classification-using-Convolutional-Neural-Network-with-PyTorch)
