**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./examples/center_2017_05_20_23_08_33_486.jpg "Center driving"
[image3]: ./examples/right_2017_05_20_23_31_08_759.jpg "Recovery Image"
[image4]: ./examples/right_2017_05_20_23_31_09_236.jpg "Recovery Image"
[image5]: ./examples/right_2017_05_20_23_31_09_857.jpg "Recovery Image"
[image6]: ./examples/right_2017_05_20_22_50_34_308.jpg "Reverse driving"
[image7]: ./examples/right_2017_05_20_22_50_41_648.jpg "Reverse driving"

## Rubric Points


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the model in NVIDIA paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)

Some preprocessing of cropping, resize and normalization has been added at input; for cropping, the top and bottom 22 pixels are removed since they are basically sky or hood of the car.
The final network architecture looks like this:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 116, 320, 3)       0
_________________________________________________________________
lambda_1 (Lambda)            (None, 116, 320, 3)       0
_________________________________________________________________
lambda_2 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 100, 24)       1824
_________________________________________________________________
spatial_dropout2d_1 (Spatial (None, 33, 100, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 50, 36)        21636
_________________________________________________________________
spatial_dropout2d_2 (Spatial (None, 17, 50, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 25, 48)         43248
_________________________________________________________________
spatial_dropout2d_3 (Spatial (None, 9, 25, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 7, 23, 64)         27712
_________________________________________________________________
spatial_dropout2d_4 (Spatial (None, 7, 23, 64)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 7, 23, 64)         36928
_________________________________________________________________
spatial_dropout2d_5 (Spatial (None, 7, 23, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 10304)             0
_________________________________________________________________
dropout_1 (Dropout)          (None, 10304)             0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               1030500
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dropout_2 (Dropout)          (None, 10)                0
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 1,167,419
Trainable params: 1,167,419
Non-trainable params: 0
_________________________________________________________________
```
The model consists of 5 convolution layers with 5x5 and 3x3 filter sizes and depths between 24 and 64 (details in model.py); followed by 3 fully connected layer with output size 100, 50 and 1;

The model includes ELU layers to introduce nonlinearity (more details for [ELU](http://www.picalike.com/blog/2015/11/28/relu-was-yesterday-tomorrow-comes-elu/)), and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

SpatialDropout2D and Dropout are also introduced in between layers to prevent overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. Apart from teh provided data, I augmented the dataset with the trainning mode in the simulator:
* Simulator runs with 640 x 480 resolution and fastest graphics.
* Use mouse to drive; keyboard inputs are too responsive and not ideal for control.
* Drive at different speed to allow different density of samples for each location and steering.
* Add correction to the steering for left and right angled pictures.
* Drive reversely on the track. The track is a circle so driving only in one direction is inclined to generate more samples for left / right turn.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use convolutional neural network to solve the regression problem given the image input predict the steering angle to keep the car on track.

My first step was to use a convolution neural network model similar to one in the NVIDIA paper. I thought this model might be appropriate because it is proven to be working to solve the same problem and it is not too complex to train.

To prevent overfitting, dropout layers are added between each two convolution and fully connected layers.

Then I experimented a little bit to find the best batch size and number of epochs. The total number of csv entries generated are 5850 so that is > 15k images since I use all the left, center, right angle inputs for training. Loss and valid_loss both went under 0.02 after 7 epochs with batch size 128 on a 1080 Ti.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I manully generated more training data on those sites to bring the car back from the point it failed.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture (note: dropout layers are not included)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from :

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also drive reversely images and angles thinking that this would generate more balanced data set w.r.t left / right turns and objects on either sides of the road. For example, here is an image that captured when driving reversely:

![alt text][image6]
![alt text][image7]

After the collection process, I had 5850 x 3 of data points with left and right steering corrected for positive and negative 0.3; I then preprocessed this data by cropping the top and bottom 22 pixels. normalization and resize to fit the input size of 66 x 200.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by experiments. I used an adam optimizer so that manually training the learning rate wasn't necessary.
