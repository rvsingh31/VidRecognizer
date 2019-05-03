This directory is related to the dataset used in the project.
The original dataset(group of videos and train/test splits) can be found [here](https://www.crcv.ucf.edu/data/UCF101.php).

For the task of training the models in a simplified way, we pre-processed the videos and saved the frames and 
optical flows for each video in an hierarchy as mentioned below.

The hierarchy which we used is explained using a sample dataset provided in this directory.

All the details about the script used for this purpose will be followed after.

mytest: This is the name of the split being used for the project.
We used trainlist01 and separated some part of it as training and validation and named it as CUSTOM3TRAIN, CUSTOM3VAL and CUSTOM3TEST.(can be found [here](split_files))

Each of these text files contains the name of the videos to be used in this split.

In 'mytest' folder, you will find different directories. These directories are distinguished based on the actions we have in the split file.
On cloud, we have 101 folders corresponding to 101 classes. 

Each of this folder will have n number of folders (one for each video under this class).

Following is a hierarchy for one video.
    - Frames(DIR) : It contains sequential frames for that video.
    - Flows(DIR) : It contains sequential flows for both X and Y directions for that video.
    - info.txt : it contains information about total frames and class ID for that video


To generate such hierarchy and use it for running the models, 
we have a script wherein you need to mention 2 things:
    1. SPLITFILE to be used to fetch videos and pre-process them.
    2. LOCATION where all the videos from original dataset are stored.

The script will store all the information in a folder named after the splitfile in a directory "FRAMESFLOWS". (You can always change it in the script)

The script can be found [here](\src\data_and_exploration\extract_images.py)


The script expects 3 command-line arguments:
    1. Start Index: The index(line number) from where you want to read the file (in order to skip earlier ones). Usually, the value is 0.
    2. End Index: The index(line number) till which you want to read the file. If you want to read till the end, you can use any value greater than the length of total videos.
    3. Tracker file: The scripts logs the line number and video file which is currently being processed to keep a track of the work done. You can specify any text file of your choice.

You can run the script using the following command (below mentioned is a sample command).

```python3 extract_images 0 9500 tmp.txt```