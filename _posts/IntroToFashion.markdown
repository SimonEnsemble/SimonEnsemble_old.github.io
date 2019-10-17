---
layout: post
title: My First Step into the Fashion Industry
excerpt: "An introduction to convolutional autoencoders using the Fashion-MNIST dataset."
comments: false
categories: articles
share: false
tags: [Machine Learning, Autoencoders, Fashion]
---

## Introduction
We're experiencing a Machine Learning (ML)/Artificial Intelligence (AI) boom. Silicon Valley is a hot pot of amateurs and experts in Machine Learning striving to create the next Google, Amazon, [insert an online powerhouse]. The field is being catapulted forward at an intense rate by researchers and it can be difficult keeping up.
The internet contains a huge amount of resources for learning ML, from a beginners tutorial to an in-depth paper about the interpretability of convolutional layers. That being said, I've not come across many easily readable and accessible resources for unsupervised learning.

This post is aimed at people taking their first steps in unsupervised learning, the interested expert or whoever has a few minutes to burn. Some prior knowledge about Python 3 is required but I'll do my best to describe each step of the way. I'll briefly talk about the goal of unsupervised learning and how it differs from supervised learning and how it works. Then I'll introduce an autoencoder, which can be split up into an encoder and a decoder. I'll show an example of a convolutional autoencoder in Python 3 using a deep learning library called Keras. The dataset we'll be working on is called *Fashion-MNIST*, and contains 10 classes of fashionable clothing. Using the autoencoder we'll be able to generate our own pieces of clothing and see relations between different types of clothing (but more on that later). I'll end on some concluding comments, and point the reader towards further resources.

If you're already informed about the details of machine learning, you're safe to skip straight to the autoencoder section.

## Machine Learning
The idea of machine learning and neural networks has been around for a long time. The idea that machines could be able to learn (and possibly become sentient!) goes back to the early days of the 20th century. The simplistic *Multilayer Perceptron* (MLP) was described in the 1950s and the artificial intelligence defense network **SkyNet** made its' debut in the 1984 thriller *The Terminator*.

But what exactly is machine learning? The term is very broad, and not easy to describe concisely. Arthur Samuel (1901-1990) was the one who coined the term machine learning, and in 1959 he described it as
> Field of Study that gives computers the ability to learn without being explicitly programmed.

Machine learning can be divided into many more categories, but the general gist of it is that we are able to train a network to see patterns and connections, and learn from them without being told explicitly to look for them.

There's one classification of machine learning that we'll mention: supervised and unsupervised machine learning. These two categories describe how our network learns from the dataset we provide. In supervised learning we provide the network with labels telling us what the output should be.

An example of supervised learning would be a neural network that could tell if something was a hot dog or not. We would have to train the network with a dataset containing a lot of pictures and a label telling us if that picture contains a hot dog or not. After the training we would be able to input just a picture and from the connections and patterns the network picked up on from our dataset, it could confidently tell us if our new picture contains a hot dog or not. This was perfectly showcased in the fourth season of Silicon Valley on HBO (WARNING: Explicit language)

[![Hot dog or not](https://img.youtube.com/vi/ACmydtFDTGs/0.jpg)](https://www.youtube.com/watch?v=ACmydtFDTGs)

In contrast to supervised learning, unsupervised learning doesn't provide the network with any labels, and we rely on the network to find patterns with no prior knowledge of the dataset. One use of unsupervised learning is clustering the data based on the patterns the network learns.

One can imagine a dataset consisting of pictures of cats and dogs. The network won't be able to learn that one group corresponds to dogs and the other corresponds to cats, but it might pick up on that cats and dogs are different and separately cluster them without knowing exactly what they are.

Here we'll use an unsupervised autoencoder to cluster the Fashion-MNIST data into separate clusters and try to investigate what patterns our network picked up on.

## The Dataset

In 2017 Han Xiao, Kashif Rasul and Roland Vollgraf published a paper describing their new fashion-centered dataset **[1]**, the *Fashion-MNIST*.
The motivation behind their work was to give Machine Learning and Data Science enthusiasts to get an alternative dataset to test their algorithms on. Previously, the MNIST handwritten-digits dataset had been ubiquitous in benchmarking algorithms.
Not only does the *Fashion-MNIST* dataset provides a more complex alternative to the old handwritten MNIST dataset, but it will improve your fashion sense significantly (not guaranteed).
![Example of clothings from *Fashion-MNIST*](../images/FashionAE/zalando10x10.jpeg)

## Machine Learning background
For the sake of brevity, I will leave out the description of optimization and regularization methods (for further reading on those, see the resources section at the end). This post is not meant to provide a comprehensive description of neural networks, but rather provide the information needed to understand autoencoders and hopefully put them to use.

We'll start with describing the multilayer perceptron to get an idea of how networks learn, and then we'll make things a bit more complicated with convolutional layers.


### The Multilayer Perceptron (MLP)
Before we start describing our autoencoder, we'll start with the basics: The multilayer perceptron (MLP). Here we have three layers, the **input**, the **output** and what we'll call the **hidden layer** (this name comes from the fact that both the input and output are directly observable, while the hidden layer is only an intermediate layer).

![A simple multilayer perceptron](../images/FashionAE/mlp.png)

We can imagine that each node in the input layer contains a single number. The arrows coming out of the input layer show us that each node in the input layer is connected to every node in the hidden layer. This connection is controlled by a set of weights, *w*. Lets zoom in on one node in the hidden layer to get a better idea what's going on:

![Zoomed in MLP](../images/FashionAE/mpl2.png)

We see that we have a weight corresponding to each connection. One problem we see here is that the outcome is linear. To make our networks able to describe more complex systems, usually the outcomes of each node (excluding the input nodes) is put through an *activation function*, which is usually non-linear. There are many activation functions available, but among the most common are the ReLU (Rectified Linear Unit), the sigmoid function and the hyperbolic tangent (tanh) function. Going back to our figure:

![Non-linear MLP](../images/FashionAE/mlp3.png)

Although we're only looking at the input nodes and a single hidden layer node, this applies to every node in the MLP network. Each node outputs a weighted sum of the inputs to that node, put through a non-linear function.

The amount of weights grows very fast with as we increase the amount of nodes in the layers. Adding more nodes usually allows us to model systems better (as long as we don't overfit the data). Another option is to add more hidden layers. In our figure we only included one hidden layer, but we could add as many as we want within reason. When we have more than one hidden layer, we usually call it a **Deep Neural Network**.

Before we wrap up our brief introduction of the multilayer perceptron, I'll talk a bit about the input data. Here each input node corresponds to a single number. We can stack these numbers into a single *N x 1* column vector, where *N* is the number of datapoints in the input. We can see that the input is one dimensional, but in a lot of cases we want to deal with higher dimensional data. A great example of 2D data is a picture. You have pixels stretching along the height and width of the picture, describing the color intensity at each point. A MLP can work with pictures, but it requires some pre-processing of the input data. The 2D data would have to be flattened down into a 1D vector. This is far from the ideal way to model a neural network capable of working with pictures. A much better way would be to introduce convolutional layers. Now we'll try to get an idea about why convolutional neural networks might be more advantageous than the MLP.


### Convolution
A convolutional layer is based off the mathematical operation [convolution](https://en.wikipedia.org/wiki/Convolution) **[2]**. For the purpose of this post, you don't need to know the math behind it, but know that due to this operation we're able to reuse a set of weights for multiple input nodes, rather than having a specific weight for each connection like for the MLP. This set of weights we'll call a **kernel**.

The kernel will learn to see some feature (horizontal/vertical line, circle, etc.) in the picture and output a signal telling us if that feature is located at some point in the picture. A simple feature it could learn is to see if a horizontal line is located in the picture.

Lets look at a picture of a cat, divided into a `5x5` grid. In reality, we don't form a grid but rather sample the pixel values, but for simplicity's sake I decided to visualize convolution using a grid system.

![A good looking cat](../images/FashionAE/conv1c.png)

Lets say we've trained a kernel to recognize cat ears. The kernel size is `3x3`, so it samples every `3x3` grid on our `5x5` picture. We move the kernel to sample every `3x3` grid on our image. Every time the kernel sees a cat eye, it outputs a 1, while it outputs a 0 if there's no cat eye. A big advantage is that there was only one set of weights used to train this kernel. Another advantage of convolutional layers is that our network becomes invariant to translation. That is, it doesn't matter where in the picture the cat eye is located because the kernel samples all possible locations on the picture.

![Convolution](https://media.giphy.com/media/5ttSu2Thrzw22QWCeG/giphy.gif)

You can see that the resulting matrix outputs a one for the bottom two lines, meaning that the kernel saw cat eyes at the bottom of the image and did not see any at the top. Usually we have a lot of different kernels, each looking for different features, and when we put those all together the network is able to see the bigger picture.

Another thing to note is that our data went from a `5x5` grid to a `3x3` grid. A lot of times we want the output to have the same dimensions as the input. To do this we'll introduce **zero-padding**. Zero-padding adds zeros around the picture which only serve to change the dimensions of the output. In a lot of neural network libraries, we can specify *same*-padding which means that regardless of the size of the kernel or the image, the library will make sure the output has the same dimensions as the input. The alternative is *valid*-padding which means no padding at all.

![Same-Padding](https://media.giphy.com/media/1yMf42rljjvnqvN4i8/giphy.gif)

We can have a lot of different kernels, all learning a different feature. That way we get a lot of different representations of our input picture. The output of each kernel is called a feature map, and it tells us where on the picture we see the feature that specific kernel learned.

Starting with a `28x28` grayscale image, we have a total of `28*28=784` datapoints. Lets say we use 10 kernels, all giving us a new `28x28` representation of our picture, we're left with `28x28x10=7840` datapoints. So rather than having a whole bunch of nodes in our hidden layer, like for the MLP, we're making a bunch of feature maps using different kernels.


## The Autoencoder
This work is heavily influenced by [this Keras blog post](https://blog.keras.io/building-autoencoders-in-keras.html). They model a few different autoencoders using the handwritten MNIST dataset. Hopefully this post can expand on the convolutional portion of the Keras blogpost and provide additional insights.


An autoencoder is a neural network that's used for unsupervised learning, that is, we don't provide the dataset with any labels for the training. The autoencoder takes an input $x$ and aims to recreate that input by putting it through a neural network. That problem sounds ideal, and one wouldn't even need a neural network to do that (it's just the identity function!).

What makes autoencoders useful is the way they recreate the input. The dimensions of the input are continuously being decreased by putting the input through specific layers. These layers can be dense layers, like the ones in the MLP, or convolutional layers. At a point in the autoencoder, we decide that we've decreased the dimensions of the input enough and start increasing the dimensions again, until we've regained the dimensions of the original input.

![A simplistic graph of an autoencoder](../images/FashionAE/myae_notransp.png)

I'll introduce two terms that are widely used when talking about autoencoders: **the encoder** and **the decoder**.

The encoder is the first part of the autoencoder, which reduces the dimensions of the input data and effectively *encodes* the data. The encoder is suppressing the information contained in the input to a low-dimensional space. This low-dimensional space is usually called the **latent space**.

The decoder takes the low-dimensional data from the latent space and tries to decode it and recreate the input. Because we're putting the input data through the latent space, which acts like a bottleneck, recreating the data is not so easy anymore.

Essentially, the autoencoder is trying to learn the most important features of the input data and representing it in the latent space. Then it tries to recreate the data from those important features.

Making an autoencoder is a fine balance between keeping the output similar to the input data, and keeping the dimensions of the latent space low. If we would have the dimensions of the latent space high, we wouldn't be forming a bottleneck and no meaningful features would be extracted from the data.

Using the low-dimensional representation of the input data, we can plot that representation onto a graph, meaningful to a human observer (that is, 2D- or 3D-plot) and cluster similar data points together. That way we may be able to see new relationships, previously unknown to the observer, and utilize that in one way or another.

# Autoencoder model

To start off, we'll import a few things in our python script. Matplotlib allows us to visualize our findings easily, Numpy is a library that adds support to large, multi-dimensional arrays and matrices, and Keras is our machine learning library.
```
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import fashion_mnist
```

Lets also import the layers and functions from keras we'll need moving onwards
```
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, ZeroPadding2D
from keras.models import Model
```

Now we can load in the *Fashion-MNIST* dataset straight from Keras. We'll rescale it so it goes from 0 - 1 and we'll reshape it as a matrix (or a 2D image).
We omitted the labels that usually go with the dataset because they're no use to us in unsupervised learning.

```
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
```

The shape of the matrix holding our dataset is now `(N, 28, 28, 1)`, where `N` is the number of images, `28x28` is the size of each image, and `1` is the number of feature maps. We'll see the feature maps increase when we start using the convolutional layers, but right now, the input images only have 1 "feature map".

The encoder and decoder of the autoencoder consist of 2 convolutional layers (with their corresponding max-pooling/up-sampling layer) and one dense layer each. The encoder and decoder are symmetrical about the bottleneck layer, which I've chosen to have 2 dimensions. That allows us to plot the latent space easily on a 2D plot.

Lets start by defining the encoder
```
myinput = Input(shape = (28, 28, 1))
# Shape = (28, 28, 1)
conv1 = Conv2D(32, (5, 5), activation = 'relu', padding = 'same')(myinput)
# Shape = (28, 28, 32)
pool1 = MaxPooling2D((2,    2))(conv1)
# Shape = (14, 14, 32)
conv2 = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(pool1)
# Shape = (14, 14, 16)
pool2 = MaxPooling2D((2,2))(conv2)
# Shape = (7, 7, 16)
conv3 = Conv2D(64, (7,7), activation = 'relu', padding = 'valid')(pool2)
# Shape = (1, 1, 64)
conv4 = Conv2D(2, (1,1), activation = 'relu', padding = 'valid')(conv3)
# Shape = (1, 1, 2)
btlnck = Flatten()(conv4)
# Shape = (2,)
encoder_model = Model(input, btlnck)
```
Lets break down this snippet:
`myinput = Input(shape = (28, 28, 1))` defines an input to the autoencoder. We have to specify the shape of each image, which is `(28, 28, 1)`.

Next we define a convolutional layer with 32 different kernels, a kernel size of `5x5`, a ReLU activation and *same* padding: `conv1 = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(input)`. Notice that at the end of the `Conv2D` call, we specify what the input to that layer is.

Following the convolutional layer, we define a max-pooling layer with a pool size of `2x2`: `pool1 = MaxPooling2D((2,2))(conv1)`.

After going through another convolutional layer (now with a kernel size of `3x3`) and a max-pooling layer, we put the data through a different convolutional layer. This layer has 64 kernels and the same kernel size as the dimensions of the data (`7x7`) and *valid* padding. This results in the output shape `(1, 1, 64)`, which can be thought of as a one dimensional array (similar to dense layers). The *valid* padding makes sure we're not adding any padding.

Next we make the bottleneck layer with a similar convolutional layer. The kernel size is `1x1` and we have two different kernels. We flatten the layer (not necessary). The `Flatten()(conv4)` layer doesn't require us to put in any parameters. We'll only specify the input layer.

Finally we define the model with `encoder_model = Model(inputs = myinput, outputs = btlnck)`

We'll construct our decoder slightly differently. The main reason is that we need to connect the bottleneck layer to the encoder, but for us to generate new fashionable clothing, we need to connect a new input to the encoder as well. Through this new input, we can throw in whatever numbers we want and see how the encoder does. But the following method was the best way I could connect a new input to an intermediate layer in our autoencoder.

We'll form layers without specifying which layer is the input. This way, we can reuse the layer for multiple models (and keep the weights)!
```
conv5 = Conv2D(64, (1,1), activation = 'relu', padding = 'valid')
# Shape = (1, 1, 64)
pad = ZeroPadding2D(padding = (6,6))
# Shape = (7, 7, 64)
conv6 = Conv2D(16, (7,7), activation = 'relu', padding = 'valid')
# Shape = (7, 7, 16)
ups1 = UpSampling2D((2,2))
# Shape = (14, 14, 16)
conv7 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')
# Shape = (14, 14, 32)
ups2 = UpSampling2D((2,2))
# Shape = (28, 28, 32)
conv8 = Conv2D(1, (5,5), activation = 'relu', padding = 'same')
# Shape = (28, 28, 1)
```

The layers try to keep the encoder and decoder as symmetrical as possible. The new additions to the system are the `ZeroPadding2D` and `UpSampling2D` layers. The `ZeroPadding2D` will add custom zero-padding to the input layer. I use it here to regain the `(7,7,x)` shape I need to keep the symmetry intact. The `ZeroPadding2D` layer introduces no weights or biases. `UpSampling2D` doesn't introduce any either, and it serves as a "reverse max-pooling", except it will increase the dimensions by repeating the numbers over and over again. An example would be `[1,2,3] -> [1,1,2,2,3,3]`. In our case, we're using a 2D version of this operation.

We've specified all the layers we need. Notice that we regain the original shape of the input image in the end. This is crucial if we want to compare our output to the input. We optimize the model based on how close our output image is to the input image.

Like I mentioned previously, we haven't specified any inputs to the decoder layers. Lets connect them to two different models: First, the autoencoder model, and second, the decoder model (called the generator model from hereon).

```
# AutoEncoder
ae = conv5(conv4)
ae = pad(ae)
ae = conv6(ae)
ae = ups1(ae)
ae = conv7(ae)
ae = ups2(ae)
decoded = conv8(ae)
autoencoder_model = Model(inputs = myinput, outputs = decoded)
```
Here we connect the first layer in the decoder to the last convolutional layer in the encoder (note that we don't connect it to the flattened layer).

Lets make our decoder as well:

```
g_input = Input(shape=(1,1,2,))
g = conv5(g_input)
g = pad(g)
g = conv6(g)
g = ups1(g)
g = conv7(g)
g = ups2(g)
generated = conv8(g)
generator_model = Model(inputs = gen_input, outputs = generated)
```

Again, the generator model is identical to the latter half of the autoencoder model, except that instead of being directly connected to the autoencoder, we can specify the input to the generator model and generate new images.

Now we've defined our model, and we just need to train it. We have to instruct the model what optimizer and loss function to use. We'll be using an optimizer called *RMSprop* **[3]** and a mean squared error loss function.
```
autoencoder_model.compile(optimizer='rmsprop', loss='mse', metrics = ['accuracy'])
autoencoder_model.fit(X_train, X_train, epochs = 10, batch_size = 128,
                        shuffle = True, validation_data = (X_test, X_test))
```
The `fit` method instructs the model to train on the training set specified. This could take a few minutes depending on the machine you're using. If it runs out of memory consider decreasing the `batch_size`.

The first two parameters represent the input data, and what the output should be. In our case, the input data should be the same as the output. These two values define the training set. `Epochs` are the number of iterations we go through the dataset and `batch_size` are the number of images we're working with at a time (to keep the memory required within reason). We also shuffle the data and specify a validation set to compare our results with and make sure we're not just memorizing the data in our training set (More information on training/validation sets in the resources below).

Congratulations! If you managed to get through this, you've made a convolutional autoencoder! Lets plot the results and see how we did.

# Figures
## Clustering
Lets see how well our model is able to cluster different clothings from the dataset. Here I've put 10,000 different clothing images into the encoder and I've plotted the 2D latent space.
```
fig = plt.figure(figsize = (10, 7.5))
ax = fig.add_subplot(111)
colors = ["#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe"]
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

cluster_data = encoder_model.predict(X_train[0:10000])
for (i, point) in enumerate(cluster_data):
    plt.scatter(point[0], point[1], c = colors[Y_train[i]], alpha = 0.5)
plt.show()
```

![Clusters](../images/FashionAE/clusterpic.png) ![Clothing legend](../images/FashionAE/Classes.png)

We can see that Sandals, Sneakers and Ankle Boots are clustered together. We can also see that Trousers and Dresses are on the left, and Bags are spread out. What's interesting as well is that our autoencoder has a hard time differentiating between Coats, Pullovers, T-shirts or Shirts. Looking at a few images of those classes, they look very similar, and a human observer would maybe even make a few mistakes classifying those images.

## Predicting fashion
Lets give our decoder an arbitrary input and see if it comes up with some fashionable clothing!
```
N = 10
fig2 = plt.figure(figsize = (10, 10))
xrange = np.linspace(0,5,N)
yrange = np.linspace(5,0,N)
X,Y = np.meshgrid(xrange, yrange)

for k in range(0, N):
    XX = np.stack((X[k,:], Y[k,:]))
    XX = np.transpose(XX)
    XX = np.reshape(XX, (N, 1, 1, 2))

    fashion = generator_model.predict(XX)

    for j in range(0,N):
        ax2 = plt.subplot(N, N, j + 1 + k * N)
        plt.imshow(fashion[j].reshape(28,28))
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
plt.show()
```
![Predicted pieces of clothing](../images/FashionAE/Predicted_Fashion.png)

We see that a lot of the predicted pieces of clothing are very similar to the ones we trained our network on. That's no coincidence because our autoencoder was only able to pick up the patterns and styles of those clothes.

# Future Directions
Unfortunately I won't be leaving my career to get on the catwalk with these results any time soon, but it demonstrated that autoencoders can pick on features and styles relatively easy! Our bottleneck (or latent space) only had two values, and from those two numbers we were able to recreate a lot of the clothing we observed in the data. That's pretty exciting!

Compared to other networks and autoencoders out there, this is not very interesting though. I decided on keeping this example simple, but with enough complexity to get some cool results.
More complexity could improve the autoencoder, granted we won't overfit the data. I've intentionally skipped talk about overfitting (there's only so much I can fit in a single blogpost), but know that if we have a very complex autoencoder it can memorize the training set, and won't do good with predictions. It's similar to fitting a straight line with a 10th order polynomial. Sure, you fit the data you got, but it gives you a really poor description of whatever you're trying to model.

 Other methods to improve it could be to add something called *Dropout*. *Dropout* removes some of the weights during each iteration of the training. This forces the remaining weights to learn something substantial. Another thing is introducing regularization, such as *l1 regularization* to some layers. *L1 regularization* adds a term to the loss function which tries to minimize the values in the regularized layers. Both *Dropout* and *l1 regularization* introduce something called *sparsity* to the autoencoder. This means that many of the outputs of the autoencoder are zero, and the ones that aren't will have to learn some important feature to make up for the other outputs that aren't contributing anything to the final result.

# Further resources

[The keras blogpost that inpired a lot of this post](https://blog.keras.io/building-autoencoders-in-keras.html)

[The keras documentation. If there are any questions about how a function behaves, this is the place to look](https://keras.io/)

[Tensorflow. A machine learning framework Keras uses as a backend. It's has more options but is more complex](https://www.tensorflow.org/)

[A really good overview of gradient descent optimization algorithms by Sebastian Ruder](http://ruder.io/optimizing-gradient-descent/index.html)

[Andrew Ng's 5 course series in Deep learning. It is behind a paywall but you can use a 7-day free trial.](https://www.coursera.org/specializations/deep-learning)

[A statsexchange post about regularization](https://stats.stackexchange.com/a/18765)

[Another statsexchange post about the difference between different sets (training/validation/test)](https://stats.stackexchange.com/a/19051)

[Distill. A free online publication about machine learning. Distill put's a lot of effort into being clear, by presenting their subjects in a dynamic and vivid way](https://distill.pub/)

# References

**[1]** [https://arxiv.org/abs/1708.07747] - Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms

**[2]** [https://en.wikipedia.org/wiki/Convolution] - Convolution, Accessed on April 30th, 2018

**[3]** Tieleman, Tijmen, and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural networks for machine learning 4.2 (2012): 26-31.
