## Project Dependencies

This project was executed on google cloud. The following script will install all the dependencies according to the system we had on GCP. We used python3.6 for development

``` pip3 install requirements.txt ```

Alternatively, if you opt to use Google Cloud for your research, you can use **Intel® optimized Deep Learning Image: TensorFlow 1.13.1 m25 (with Intel® MKL-DNN/MKL and CUDA 10.0)** image as boot disk in your compute engine to avoid installing all the requirements.

## Source Code (Developed using Keras with Tensorflow backend)

The following is a brief description of the sub-directories present in this directory.

* **data_and_exploration**: 
    - Contains the dataset used for training and testing the constructed models. The dataset which we are referencing here is a textfile containing a split of the original dataset over which various architectures have provided their benchmarks. Using the same split helped us compare our results with the already established architectures.
    You can download the videos from [here](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar).
    - *extract_images.py*: In order to work on videos, we performed preprocessing part where we stored all the frames, their optical flows and some relevant information on disk. This file helped us with that pre-processing part.
    - *generateDataset.py*: This file was used for shuffling and creating custom datasets with equal distribution over all classes using the predefined splits.

* **papers**: These are some reference papers which we referred during the course of this project

* **random_scripts**: This directory contains some random scripts which were used for generating plots with intermediate results and relevant error checking.

* **models**: In all, we implemented 3 architectures in this project. This directory contains 3 sub-directories namely *I3D*, *C3D*, *TSN*. Each of these directories contains the code for the respective model.

* **trained_model_weights**: This directory contains the saved weights after the training of the respective models. These weights are further used for prediction

* **training_logs**: This directory contains the outputs of CSVLogger for all the models during the training phase.

* **results**: 
	- *Video_Outputs*: This directory contains some videos with our models' results embedded in them. The embedded text shows the action predicted by C3D model alongwith the confidence the model has for that action.
	- It also contains a loss curve for studying TSN's behavior and archived results text documents for reference.

## Use of Google Cloud Platform 

Our biggest challenge was to work on such a big dataset with limited resources we had. So, we migrated our code to [google cloud](https://cloud.google.com) and were able to use compute-intensive GPUs for training purpose. Our pre-processed videos are still stored on the disk(~110GBs). Let us know if you would like to use them for further research.


## Running the models' scripts

As mentioned before, we implemented three models. Out of which, we were able to train two models successfully given the timeframe we had and multiple implementation issues we faced, the reasons of which can be found in our project report[ADD REPORT IN REPO] [here](../Report.pdf).

### C3D

Implementation of C3D consists of two files:
* *models.py* : This file contains the architecture of the model
* *vidc3d.py* : This file is the driver program for training/testing the model. It expects one command-line argument which is to mentioned whether to *'train'* or *'test'* the model.

In order the run this file for training, it expects a pre-defined weights file which we have provided [here](trained_model_weights/c3d). Once, you have the weights in your local system, give appropriate location in the script. After that, you also have to specify the location of your pre-processed videos, the splitfile to use for training in order to create a proper data generator for training. Once, all these steps are done, you can train the model by the following command.

 ``` python3 vidc3d.py train ```

In order to test the model over any specific video, you have to specify two things in the *test* function.
   * which weights to use for testing - can be found [here](trained_model_weights/c3d).
   * which videofiles to test - Can be a single video or can be multiple. Uncommenting specific lines will do the task.

Once you do the above mentioned steps, you can test the model by the following command.

 ``` python3 vidc3d.py test ``` 


### I3D

Our I3D implementation is adapted from [Keras Kinetics I3D](https://github.com/dlpbc/keras-kinetics-i3d). We use pre-trained weights for this model that can be downloaded from within the script. Please read script comments to follow that. Specifically, our model used this pre-trained weight: 'rgb_imagenet_and_kinetics'. The same model is used for both Frame training and flow, we can use the argument as described below to proceed further.
Implementation of I3D consists of two files:
* *i3d_inception.py* : This file contains the architecture of the model
* *i3d.py* : This file is the driver program for training/testing the model. It expects two command-line argument:
	*  First argument is to mention whether to *'train'* or *'test'* the model.
	* Second argument is optional in case of testing. For training,  you need to specify what part of video to train on ie. *'FLOW'* or *'FRAME'*.

As mentioned above in the implementation of C3D, you will want to check for proper weights' directory and splitfile directory before training/testing the model. For testing or training further, the weights are provided [here](trained_model_weights/i3d).

For training the model, you can use

``` python3 i3d.py train FLOW```
or
``` python3 i3d.py train FRAME```

For testing the model, you can use

``` python3 i3d.py test ```

### TSN

Implementation of TSN is consists of 3 files:
* *inception.py* : This file contains the implementation of InceptionV2ResNet model in Keras. Reason of using the source code of this model and not the Keras Functional API is to stack up multiple inception models as per our need for TSN model.
* *pretrained-models.py* : This file contains the implementation of TSN model.
* *tsn_implementation.py* : This file is the driver program for training/testing the model. It expects one command-line argument which is to mentioned whether to *'train'* or *'test'* the model.


As mentioned above in the implementation of C3D, you will want to check for proper weights' directory and splitfile directory before training/testing the model. For training the model, the weights will be downloaded automatically. For testing or training further, the weights are provided [here](trained_model_weights/tsn).

For training the model, you can use

``` python3 tsn_implementation.py train ```

For testing the model, you can use

``` python3 tsn_implementation.py test ```

One thing to note here is in Keras, the implementation of Batch Normalization layer is not providing correct behavior as it should have. This issue is addressed in this blog [here](https://blog.datumbox.com/the-batch-normalization-layer-of-keras-is-broken/). So, the model is trained by keeping the layer's training property as False (which is equivalent to NOT use Batch Normalization later at all). So, a better option to implement this model is to shift the architecture to some other framework ie PyTorch or Caffe (which is listed as future work on our part). 
