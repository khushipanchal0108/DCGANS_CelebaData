# DCGANS for image generation

## overview
this project implements a deep convolutional generative adversarial network (dcgan) using pytorch. the goal is to generate realistic images by training the model on either the celebA faces dataset or the lsun bedrooms dataset.

## dataset
- celebA: a large-scale dataset of celebrity faces
- lsun bedrooms: a dataset containing bedroom images
- the images are resized to 64x64 pixels and normalized to [-1,1] for stable training

## architecture
### generator:
- input: 100-dimensional noise vector
- uses fractionally-strided convolutions (also called transposed convolutions) to upsample the noise
- batch normalization and relu activations are used
- final layer uses tanh activation to output a 64x64x3 image

### discriminator:
- input: 64x64x3 image
- series of convolutional layers with leaky relu activation
- batch normalization used to stabilize training
- final layer outputs a probability score using sigmoid activation

## training details
- loss function: binary cross-entropy loss (bce)
- optimizer: adamw with learning rate 0.0005 and betas (0.5, 0.999)
- batch size: 64
- epochs: 25
- training is done using real and fake images, updating both generator and discriminator iteratively

## running the code
1. install dependencies: `pip install torch torchvision matplotlib`
2. run the script in google colab or a local environment with gpu support
3. generated images are saved periodically for visualization

## expected results
- early epochs: blurry, unrealistic images
- mid-training: images start forming recognizable features
- final epochs: realistic images resembling dataset samples

## acknowledgments
- based on "unsupervised representation learning with deep convolutional generative adversarial networks" by radford et al. (2015)
- pytorch tutorials on gans

