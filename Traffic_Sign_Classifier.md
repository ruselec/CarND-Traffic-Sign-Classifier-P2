# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/signs_example.png "Signs example"

[image2]: ./examples/visualization.png "Visualization"

[image3]: ./examples/augment.png "Augmentation"

[image4]: ./examples/augment_chart.png "Visualization"

[image5]: ./examples/gray_and_histeq.png "Preprocessing"

[image6]: ./examples/1.png "Traffic Sign 1"

[image7]: ./examples/2.png "Traffic Sign 2"

[image8]: ./examples/3.png "Traffic Sign 3"

[image9]: ./examples/4.png "Traffic Sign 4"

[image10]: ./examples/5.png "Traffic Sign 5"

[image11]: ./examples/6.png "Traffic Sign 6"

[image12]: ./examples/predict.png "Softmax prediction"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here is an visualization one of the classes. The code for the visualization is contained in the third code cell of the IPython notebook. 

![alt][image1]

It is a bar chart showing number of training examples in each class. The code for the bar chart is contained in the fourth code cell of the IPython notebook.

![alt][image2]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to augment the images because the dataset is unbalancing. I used translation image for this. The code for the augmentation of data is contained in the fifth and sixth code cells of the IPython notebook. 

Next image shows examples of augmentation.

![alt][image3]

It is a bar chart showing number of training examples in each class after augmentation. The code for the bar chart is contained in the seventh code cell of the IPython notebook.

![alt][image4]

Then, I converted the images to grayscale because it decreased network size and time of training. I applied histogram equalization because for network training it would be better if signs have the same brightness. The code for the preprocessing of data is contained in the eigth code cell of the IPython notebook. 

Here is an example of a traffic sign images after grayscaling and histogram equalization.

![alt][image5]

As a last step, I normalized the image data because if the data image is in range [-1; 1] the network training process is less time consuming. The code for the normalizing of data is contained in the ninth code cell of the IPython notebook. 

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for augementation the data for training set is contained in the sixth code cell of the IPython notebook.  

To validate my model, I used the original validation set without splitting it from training set. My final training set had 84326 number of images. My validation set and test set had 4410 and 12630 number of images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

I used modified LeNet Model Architecture from Sermanet/LeCunn paper. My final model consisted of the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 grayscale image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x16 	|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x16 				|
| RELU					             |												|
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16 				|
| RELU					             |												|
| Convolution 5x5	      | 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					             |												|
| Flatten1					         |	input 5x5x16,  outputs 400 										 |
| Flatten2					         |	input 1x1x400,  outputs 400 										|
| Fully connected		     | input 400+400,  outputs 800   									|
| Dropout			            |         									|
| Fully connected		     | input 800,   outputs 43  									|

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the thirteenth cell of the ipython notebook. 

To train the model, I used:
* an Adam optimizer
* batch size: 256
* epochs: 20
* starting learning rate: 0.001
* mu: 0
* sigma: 0.1
* dropout keep probability: 0.5

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the forthteenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.973 
* test set accuracy of 0.951

An iterative approach was chosen:
1. The first architecture that was tried: LeNet Model
2. The problem was that test accuracy was not achieve 0.93.
3. Then modified LeNet Model Architecture was chosen because it was described in the [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) as the good choice for traffic signs recognition.
4. Architecture was adjusted by:
* type of activation (tanh, relu). Relu was chosen because it is not vanishing gradient and training with relu is going faster.
* convolution padding (same and valid). Valid was chosen because it decrease size of network.
* dropout layer was added as hidden layer before last fully connected to avoid over fitting.
* convolution first layer's input was increased from 32x32x6 to 32x32x16 because last of the six new images was predicted false. The visualization of first convolutional layer show that "Right-of-way at the next intersection" sign and  "Beware of ice/snow" sign have similar feature maps. So new features could be trained in neural network by increasing  input size.
5. Parameters were tuned:
* batch size was setted to 256 to maximazr generalization.
* number of epochs was setted to 20 because loss of validation not decrease with new epochs.
* dropout keep probability was chosen to avoid over fitting.
6. The final model's accuracy on the training, validation and test sets show that model is trained enough to recognize similar traffic signs with accuracy about 0.95 that was proved later on new images.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt][image6] ![alt][image7]  ![alt][image8] 
![alt][image9] ![alt][image10] ![alt][image11]

The last image might be difficult to classify because of snow masked some parts of it.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the sixteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			                                     |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      	 | Right-of-way at the next intersection   									| 
| No passing     			                           | No passing 										                            |
| Priority road				                            | Priority road										                          |
| Dangerous curve to the left      		          | Dangerous curve to the left					 			           	 |
| Beware of ice/snow			                        | Beware of ice/snow                             		|
| Beware of ice/snow			                        | Beware of ice/snow                             		|

The model was able to correctly guess 6 of the 6 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of original dataset.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the eighteenth cell of the Ipython notebook.

For all images, the model correct predicted signes with probability of 1.0. The top five soft max probabilities were shown for six images on follow bar charts:

![alt][image12]
