# @carvanamask

### U-net Image Segmentation Kaggle Competition

## Tags

`image segmentation`, `deep learning`, `unet`, `python`, `keras`, `neural networks`, `computer vision`, `tensorflow`, `opencv`, `image processing`

## Introduction

In computer vision, image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as super-pixels). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images. More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

In this project, a system that automatically removes cars from the photo studio background is developed. The project is designed as a part of [Kaggle competition](https://www.kaggle.com/c/carvana-image-masking-challenge) for online used car startup called Carvana. Such a model will allow Carvana to superimpose cars on a variety of backgrounds. 
 
<p align="center">
  <img src="/uploads/7755da0db1e284f5e6239818e99a9175/carvana.jpg" width="600px" >
</p>

The result of a competition is an ensembled model of different approaches made in a group of 3 persons.  

The Public Leaderboard place: 10-th.  
The Private Leaderboard place: 51-th (top 7%).

The Private score was lowered due to the presence of exactly one car in the Private set on which the model crashed. In all other respects, the overall model is very efficient. The mistake made will be a good experience for us.

## Getting Started

These instructions allow you to reproduce the project and run it on your local machine for development and testing purposes. 

### Prerequisites

The following software was used in this project:

* PyCharm: Python IDE for Professional Developers by JetBrains;
* Anaconda 4.4.0 Python 2.7 version;
* Keras: The Python Deep Learning library with Tensorflow backend;
* OpenCV >= 3.2 for Python;
* Other python modules can be installed by `pip install`.

### Project structure

    ├── data                                    # data files
        ├── ...
    ├── src                                     # project source
        ├── modules                             # additional modules of a common purpose
            ├── ...
        ├── segmentation                        # segmentation approach
            ├── logs                            # tensorflow logs
            ├── models                          # trained models
            ├── results                         # prediction results
            ├── ...
    ├── config.py
    ├── main.py
    ├── ...    
    
`/src` has to be marked in PyCharm as sources root folder. 

### Glossary

* [Kaggle](https://www.kaggle.com/) - a platform for predictive modelling and analytics competitions in which companies and researchers post data and statisticians and data miners compete to produce the best models for predicting and describing the data.
* [Carvana](https://www.carvana.com/) - a successful online used car startup, has seen opportunity to build long term trust with consumers and streamline the online buying process.
An interesting part of their innovation is a custom rotating photo studio that automatically captures and processes 16 standard images of each vehicle in their inventory. While Carvana takes high quality photos, bright reflections and cars with similar colors as the background cause automation errors, which requires a skilled photo editor to change.

### Data

The dataset contains a large number of car images (as `.jpg` files with 1918x1280 resolution). Each car has exactly 16 images, each one taken at different angles. Each car has a unique id and images are named according to id_01.jpg, id_02.jpg ... id_16.jpg.  
For the training set `.gif` files are provided that contain the manually cutout mask for each image.

Train images are supposed to be kept in `/data/train` folder while train masks are kept in `/data/train_masks` folder.

<p align="center">
    <img src="/uploads/0a6759f4761a262ecf0774218b32a25b/train_images.jpg">
</p>

In order to make submission for the competition `/data/test` folder is used.

All the data can be downloaded [here](https://www.kaggle.com/c/carvana-image-masking-challenge/data).  

## Approach

### U-Net Convolutional Networks for Image Segmentation

The [u-net](https://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a/) is convolutional network architecture for fast and precise segmentation of images. Up to now it has outperformed 
the prior best method (a sliding-window convolutional network) on the ISBI challenge for 
segmentation of neuronal structures in electron microscopic stacks. It has won the Grand 
Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, 
and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted 
light microscopy categories (Phase contrast and DIC microscopy) by a large margin.

<p align="center">
  <img src="/uploads/21250e3003ccc09fc09f7c1463fafe78/unet.jpg" width="600px" >
</p>

### Model parameters

In this project, a u-net model with 6 double downsample/triple upsample levels is used. The initial kernel size is 16. 

The main image resize choice is 1024x1024. But other options like 1344x896 or 1280x1280 were also used in the ensembled model.

* Optimizer: RMSprop with lr = 0.0001;
* Keras ReduceLROnPlateau(): {factor=0.1, patience=4};
* Batchsize: 2;
* Batch Normalization is used;
* Validation: one hold out 15% data.

There were attempts of using Batch Renormalization during the execution of the tests, but, unfortunately, it does not improve the results.

### Loss function

Most prediction errors are focused on the boundaries of cars. Therefore, a loss function is proposed, which allows  to overestimate the importance of such pixels.  
<p align="center">
  <img src="/uploads/8985b37e4c6559a4759e150f184c1b68/loss_function.jpg" width="350px">
</p>
Boundary pixels can be both background and foreground. Different coefficients are used for ensembling.

<p align="center">
  <img src="/uploads/48dc5c1b30c4ee69ed2c9bdbf83cce35/boundaries.jpg" width="600px" >
</p>

### Data augmentation

In order to combat the high expense of collecting thousands of training images, image augmentation has been developed in order to generate training data from an existing dataset. Image Augmentation is the process of taking images that are already in a training dataset and manipulating them to create many altered versions of the same image. This both provides more images to train on, but can also help expose classifier to a wider variety of lighting and coloring situations so as to make classifier more robust. 

<p align="center">
  <img src="/uploads/50c18140c8bf26aadf671165ecd88c01/augmentation.jpg" width="600px" >
</p>

Augmentation techniques used:

* Random hue-saturation-value shifting: 

```
hue_shift_limit=(-50, 50)
sat_shift_limit=(-5, 5)
val_shift_limit=(-15, 15)
```

* Random shifting and scaling:

```
shift_limit=(-0.0625, 0.0625)
scale_limit=(-0.1, 0.1)
```

* Random horizontal flipping;

It also possible to use adaptive histogram equalization but the time required for its execution outweighs the insignificant improvement of the model performance.

In order to increase prediction performance the images with/without flopping are used together for averaging the final prediction.

### Adjusting threshold

Different binarizing threshold were checked. The threshold which gives the best score on validation set is 0.485.

### Pseudo labeling

One of the best contributors of improving score is training on pseudo labeling data. Pseudo labeling contribute so much because of the amount of test data, and it is possible to get predictions close ground truth.


## Running the project

The main file is `main.py` located in the root folder. 

It expects training images and masks downloaded from [Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

### Usage

Training the new model:

```
python main.py train 
```

Training the pretrained model:

```
python main.py train --model model_name
```

Evaluate trained model on random images:

```
python main.py eval --model model_name --size num_images
```

Create Kaggle submission and save predicted masks:

```
python main.py submit --model model_name --save True
```

### Examples

<details> 
  <summary>Example 1</summary>
    <p align="center">
  		<img src="/uploads/cf592a0d8a247023d268942a74b88b60/example01.jpg">
	</p>
</details>

<details> 
  <summary>Example 2</summary>
    <p align="center">
  		<img src="/uploads/827c2fd1dc1109b76a8c165d73603532/example02.jpg">
	</p>
</details>

<details> 
  <summary>Example 3</summary>
    <p align="center">
  		<img src="/uploads/eb939370a299e06fc1d6799e6e6c1448/example03.jpg">
	</p>
</details>

<details> 
  <summary>Example 4</summary>
    <p align="center">
  		<img src="/uploads/34059fb5630b5dc8d384f930e7951a19/example04.jpg">
	</p>
</details>

<details> 
  <summary>Example 5</summary>
    <p align="center">
  		<img src="/uploads/3650e4ae95cbf2b9ac2a012448139311/example05.jpg">
	</p>
</details>

## Fate played a cruel joke

Being on the 10th place in the evening before summarizing the results of the competition, we hoped to be in the TOP-20 on the Private Leaderboard.

<p align="center">
	<img src="/uploads/0ce6efc46d8b341731fae3bcc3dd711f/public_leaderboard.jpg" width="600px">
</p>

In the morning we found out that we were thrown to 51st place on the Private Leaderboard. After some analysis, we realized that the error was covered in only one car, which was absent in the public set, but present in the private one.

<p align="center">
	<img src="/uploads/9417f40f6c028c6b677c2600355a4981/minivan.jpg" width="600px">
</p>

Most competition participants solved this problem either manually or directly checking the image with this car and applying simple transformations. We only had to worry in advance about this situation.

As a result, we corrected the error and found out that we could be on 6th place. A great lesson for life.

<p align="center">
	<img src="/uploads/a74c18bfb5e17e77aa70323344c6e357/private_leaderboard.jpg" width="600px">
</p>