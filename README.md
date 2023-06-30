# GestureGAN

A project where I train a generative adversarial network, or GAN (more specifically, a deep convolutional generative adversarial network, or DCGAN) to generate images of certain hand gestures.

# Data Collection

A total of 1500 image samples were collected, according to the following format:

- 300 photos taken over 5 locations.
- In each location, 50 samples were allocated for each gesture, with 100 being control samples (no gesture).
- For the control samples, 25 had a face, 25 had a hand, 25 had neither, and 25 had both.
- For the gestures, 25 were recorded for each hand in various positions, with approximately half (12-13) of the 25 including faces, with the other half not including any face.

A few sample images are shown below:

![First Sample Image](./IMG_16640.png)
![Second Sample Image](./IMG_02570.png)
![Third Sample Image](./IMG_10920.png)

Photos were taken using an iPhone 13 Pro in the HEIC format, with 2316x3088 pixel resolution. When ported to a Windows PC, their resolution changed, with some images being 756x1008 and some being 579x772.

# Issues around the data collection

The largest issue is that a single person (myself) was used to take the photos; the model has learned to recognize gestures from people with my hand size, structure, and skin color. For example, if someone with darker skin and a larger hand than mine performed gestures, the model may struggle to recognize their gestures. I predict a more complex architecture than the one ultimately chosen would need to be used to effectivly capture these differences.

Another issue is the size of the dataset; there are only 1500 samples collected, and even after preprocessing (described in the next section), 12000 samples are available. This is far less than datasets used to train state-of-the-art models (such as ImageNet), which contain millions of samples.

Finally, the size of the images; the photos are far larger than popular image dataset sizes (MNIST, for instance, uses 28x28 photos [1]).

# Preprocessing

Preprocessing the data involves a few steps: we first convert the HEIC images to PNG. To ensure all images are the same size, we pad them to a size of 64x64, which also is intended to match the image size in the paper we reference. Then, to increase the size of our dataset, we will create more training examples by taking transformations of the images. 

In this case, we take 90, 180 and 270 degree rotations, then apply Gaussian noise to the images. This results in 8 images generated from 1 original sample, for a total of 12000 images generated from the original data.

All images are scaled down on each side by a factor of 8, giving is 126x126 sized images.

# Generator and Discriminator Architecture

We will be using the generator and discriminator architectures from the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" [2]

# References

[1] https://en.wikipedia.org/wiki/MNIST_database

[2] https://arxiv.org/abs/1511.06434