
# MNIST Handwritten Digit Generation using GAN

This project implements a Generative Adversarial Network (GAN) to generate images of handwritten digits, trained on the MNIST dataset. The GAN consists of a generator model that learns to create realistic digit images and a discriminator model that learns to distinguish between real MNIST digits and fake (generated) digits.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Prerequisites](#prerequisites)  
- [Setup](#setup)  
  - [1. Clone Repository (Optional)](#1-clone-repository-optional)  
  - [2. Create Virtual Environment (Recommended)](#2-create-virtual-environment-recommended)  
  - [3. Install Dependencies](#3-install-dependencies)  
- [Usage](#usage)  
  - [1. Configure Parameters (Optional)](#1-configure-parameters-optional)  
  - [2. Run the Training Script](#2-run-the-training-script)  
- [Model Architecture](#model-architecture)  
  - [Generator](#generator)  
  - [Discriminator](#discriminator)  
  - [Combined GAN Model](#combined-gan-model)  
- [Training Process](#training-process)  
- [Outputs](#outputs)  
- [File Description](#file-description)  
- [Customization](#customization)  
- [License](#license)  

---

## Project Overview

Generative Adversarial Networks (GANs) are a class of machine learning frameworks where two neural networks — a generator and a discriminator — compete in a zero-sum game. The generator creates fake data resembling real data, while the discriminator learns to distinguish real from fake. This project implements a basic GAN to generate 28x28 grayscale images of handwritten digits using the MNIST dataset.

---

## Features

- **MNIST Dataset:** Uses the standard handwritten digits dataset provided by Keras.  
- **Generator Model:**  
  - Takes random noise (latent vector) as input.  
  - Uses dense and convolutional transpose layers to upsample to 28x28 images.  
  - LeakyReLU activations with sigmoid output for pixel values between 0 and 1.  
- **Discriminator Model:**  
  - Binary classifier to distinguish real vs. fake images.  
  - Uses convolutional layers with LeakyReLU and dropout for regularization.  
  - Outputs a probability with sigmoid activation.  
- **Combined GAN Model:**  
  - Stacks generator and discriminator for training generator while freezing discriminator weights.  
- **Training Loop:** Alternates training discriminator and generator with appropriate labels.  
- **Performance Monitoring:** Prints losses, discriminator accuracy, saves generated image plots and model weights periodically.

---

## Dataset

The project uses the [MNIST dataset](https://keras.io/api/datasets/mnist/) of handwritten digits. Keras automatically downloads it if not found locally.

---

## Prerequisites

- Python 3.7 or higher  
- pip package manager  
- Git (optional, if cloning the repo)  

---

## Setup

### 1. Clone Repository (Optional)

If you have a Git repository for this project:

```bash
git clone <your-repository-url>
cd <your-repository-name>
Otherwise, just save the Python script locally (e.g., mnist_gan.py).

2. Create Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
3. Install Dependencies
Create a requirements.txt file with the following content:

nginx
Copy
Edit
tensorflow
numpy
matplotlib
Then install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Usage
1. Configure Parameters (Optional)
You can modify these parameters at the top or end of the Python script:

latent_dim — dimension of noise vector (default: 100)

n_epochs — number of training epochs (default: 100)

n_batch — batch size (default: 256)

Adam optimizer parameters like learning rate and beta_1 are defined inside model functions.

2. Run the Training Script
bash
Copy
Edit
python mnist_gan.py
The MNIST dataset will be automatically downloaded if not available.

Training progress (losses, accuracy) will be printed batch-wise.

Every 10 epochs:

Discriminator accuracy on real and fake samples is printed.

A plot of 100 generated images is saved as generated_plot_e<epoch_number>.png.

The generator model is saved as generator_model_<epoch_number>.h5.

Model Architecture
Generator
Input: latent noise vector of dimension latent_dim (e.g., 100).

Dense layer projecting input to 7x7x128 nodes.

LeakyReLU activation.

Reshape to (7, 7, 128).

Two Conv2DTranspose layers with LeakyReLU upsampling to (14, 14, 128) then (28, 28, 128).

Conv2D with sigmoid activation outputs grayscale image (28, 28, 1).

Discriminator
Input: (28, 28, 1) grayscale image.

Two Conv2D layers (64 filters, 3x3 kernel, stride 2) with LeakyReLU and Dropout (0.4).

Flatten.

Dense layer with sigmoid activation outputs probability real/fake.

Optimized with Adam (learning_rate=0.0002, beta_1=0.5).

Loss function: binary crossentropy.

Combined GAN Model
Stacks generator and discriminator models.

Discriminator weights frozen during generator training.

Optimized with Adam (learning_rate=0.0002, beta_1=0.5).

Loss: binary crossentropy.

Training Process
For each batch:

Train discriminator on half real samples and half fake samples (generated images).

Train generator via combined model with labels pretending generated images are real.

Periodically evaluate and print discriminator accuracy.

Save generated image grids and generator model weights every 10 epochs.

Outputs
Console Logs: Training progress including losses and accuracy.

Image Plots: Files named generated_plot_e<epoch_number>.png showing 100 generated digits.

Saved Models: Generator models saved as generator_model_<epoch_number>.h5.

File Description
mnist_gan.py: Main Python script containing model definitions, training loop, and utilities.

requirements.txt: Python package dependencies.

generated_plot_e*.png: Generated image grid files.

generator_model_*.h5: Saved Keras generator models.

Customization
Modify architecture: add BatchNormalization, change filter sizes, experiment with activation functions.

Adjust hyperparameters: latent_dim, learning rate, batch size, epochs.

Use other datasets by adjusting input/output shapes and data loading functions.

Add advanced GAN evaluation metrics like Inception Score (IS) or Fréchet Inception Distance (FID).

