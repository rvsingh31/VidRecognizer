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

