# Monotype-Assignment
This project implements a Generative Adversarial Network (GAN) to generate images of cats. The GAN architecture consists of a Generator and a Discriminator, both implemented using PyTorch. The model is trained on a dataset of cat images, and the performance is monitored through the convergence of the training and validation loss.

## Dataset

The model is trained on a dataset of cat images. You can use the [Cat Dataset from Kaggle](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models/data), or any other dataset of your choice.
The dataset should be structured in a way that the images are stored in folders, which is suitable for the `ImageFolder` class in PyTorch.

## Model Architecture

### Generator

The Generator is designed to take random noise as input and generate a 64x64 RGB image. The architecture includes several transposed convolutional layers, batch normalization, and ReLU activation.

### Discriminator

The Discriminator is a binary classifier that takes an image as input and outputs a probability that the image is real or fake. The architecture includes several convolutional layers, batch normalization, and LeakyReLU activation.

### Enhancements

- **Spectral Normalization**: Applied to the Discriminator to stabilize training.
- **Dropout**: Added to the Generator to prevent overfitting and improve the diversity of generated images.
- **Label Smoothing**: Used for the real labels to make the Discriminator less confident, helping the Generator learn more effectively.

## Training

The model is trained for a maximum of 10 epochs. The training process involves alternating updates to the Discriminator and Generator. Losses for both models are monitored and plotted to evaluate the performance.

## Generated Images
<img width="695" alt="Screenshot 2024-08-09 at 4 29 02 PM" src="https://github.com/user-attachments/assets/6794c1e3-312d-43ed-8c8f-72547fa28f94">


```python
# Key Training Parameters
batch_size = 128
image_size = 64
nz = 100  # Size of the latent z vector (input to generator)
ngf = 64  # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 11
lr = 0.0002
beta1 = 0.5
