# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2018_07_31_19_35_57_978.jpg "Center camera image"
[image3]: ./examples/left_2018_08_03_00_06_47_814.jpg "Recovery Image"
[image4]: ./examples/right_2018_08_03_00_06_45_635.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup.md summarizing the results
* video.mp4 shows a vehicle driving on a track autonomously using the trained model.

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The code also contains weblinks for quick references to various technical topics and should be helpful for understanding the code.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried to setup sequential model using Keras similar to Nvidia deep-learning self-driving-car CNN (https://devblogs.nvidia.com/deep-learning-self-driving-cars/) and made changes to include few layers as listed below.

My model starts with a convolution neural network layer with 3x3 filter sizes and depths between 24 and 64 (model.py lines 98-135). In between the convolution layers pooling is performed for down sampling.

The model starts with preprocessing steps of normalization and cropping of images (model.py lines 101-106). The data is normalized in the model using a Keras lambda layer (code line 101).

The model includes RELU layers for each of the convolution in order to introduce nonlinearity.


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 129 and 131).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 140).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I intentionally drove the vehicle in a zig-zag fashion. I collected data driving the vehicle on right turning track. Also, I used the second track to collect additional data.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build something similar to familiar architecture that have solved this problem. With that in mind and various reading and project references I chose to mimic Nvidia CNN. https://developer.nvidia.com/discover/convolutional-neural-network

I got started with only few layers (Convolution2D, Relu activation, flattening, dense). With limited layers, I got confirmation that my workflow is atleast correct and vehicle tends to move in a right direction. However, things got out of control very fast as the vehicle could steer timely manner and go over the road edges especially on turns.

As the vehicle was moving, I decided to split my image and steering angle data into a training and validation set to get into the expected project output.

After plotting the mean squared error (MSE) used for the loss function per epoch,  I found that the model had a low MSE on the training set but a high on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model to include droput layer. The final step was to run the simulator to see how well the car was driving around track one.

However the vehicle was not driving well. So I almost doubled the number of layers adding extra Convolution2D, Max Pooling, Dropout. I even experimented with different activation functions settling on LeakyRELU for one layer. With I managed to get first two turns on the track successfully. After the second turn, the vehicle would cross over to flat ground instead of continuing with the turn. That's when I realized the image (RGB vs BGR) issue reading over Slack channel. After fixing that majority of the issues got resolved, the model was training better.  

There were couple of spots where the vehicle was going close to edge of the road and in one case climbing up but somehow managing to return to the track.To improve the driving behavior in these cases, I experimented a lot with various parameters. I tried changing values to various layers (% value for dropout, number of filters on Convolution2D layer, number of epochs etc). I also adjusted the image cropping. Additionally, I collected data for those specific turns where the vehicle maneuvering was not good. That helped to improve the model training and eliminated the overfitting visible in the plot.

Final adjustment to the steering angle values on left, right image did the trick and allowed the vehicle to run through the line markings all throughout the track.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 98-135) consisted of a convolution neural network with the following layers and layer sizes.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 320x160x3 RGB image   					|
| Convolution 2D     	| 3x3 stride, valid padding, 24 filters 	|
| RELU					|												|
| Max pooling 2D		   	| 2x2 stride 					|
| Convolution 2D     	| 1x1 stride, valid padding, 48 filters 	|
| RELU				|												|
| Max pooling		   	| 1x1 stride 					|
| Convolution 2D     	| 1x1 stride, valid padding, 64 filters 	|
| Flatten			    |    									|
| Dropout		| 25%							|
| Fully connected		| Output 100							|
| Dropout		| 40%							|
| Fully connected		| Output 50							|
| LeakyRELU					| 0.3												|
| Fully connected		| Output 10							|
| Fully connected		| Output 1							|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded just over two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle hard turns and recovering from those turning back to the center of the road. These images show what a recovery looks like starting from left and right side :

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points for two specific turns where the vehicle had trouble.

To improve learning I gather data from the second track.

After the collection process, I had 20 thousand number of data points. I then preprocessed this data by normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by observing the overfitting plot of MSE vs number epochs, finally deciding to use 5. By keeping everything same and just changing the the number of epochs to 6 would result in overfitting on the last epoch.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
