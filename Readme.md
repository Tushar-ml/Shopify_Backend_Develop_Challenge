# Similar Image Searching using Resnet50

## Introduction
Given a batch of images, the program tries to Search similar images using Resnet50 based feature vector extraction.

## Folder Structure
* Images Folder will contain all other images, your input image will be compared.(Currently Dogs and Cats)
* Uploads Folder will contain images that you upload or paste in them.

## Usage
``Put your Images in Uploads Folder (example Dog and Cat)``
``python kreas_resnet50.py`` will compare all the images present in ``images`` folder and ``upload`` folder with each other and provide the most similar image for every image. 

## Pre-Requisites
* Download [Anaconda](https://www.anaconda.com/download/#linux)
* Make the downloaded shell script executable and install
* ``conda -V`` to check that installation was successfull. 
* ``conda update conda`` and ``conda update anaconda``
* ``conda update scikit-learn``
* ``conda install theano``
* ``conda install -c conda-forge tensorflow``
* ``pip install keras``
* ``export MKL_THREADING_LAYER=GNU``

